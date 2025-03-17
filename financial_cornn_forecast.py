import torch
import torch.nn as nn
import numpy as np
from financial_cornn import coRNN1DCell

"""
Enhanced 1D CoRNN model for financial time series forecasting.
Supports both single-step and multi-step forecasting with various readout strategies.
"""

class FinancialForecastModel(nn.Module):
    def __init__(self, seq_length, c_in, c_mid, c_out, forecast_horizon=1, 
                 min_iters=10, max_iters=100, dt=0.5, T=50, readout="last"):
        """
        Args:
            seq_length: Length of input time series
            c_in: Number of input features (e.g., OHLCV data)
            c_mid: Number of channels in middle layers
            c_out: Number of output channels in hidden state
            forecast_horizon: Number of future time steps to predict
            min_iters: Minimum iterations
            max_iters: Maximum iterations
            dt: Time step for ODE
            T: Number of RNN unroll steps
            readout: Readout method ("last", "mean", "fft", "linear")
        """
        super(FinancialForecastModel, self).__init__()

        self.seq_length = seq_length
        self.c_in = c_in
        self.c_out = c_out
        self.K = int((max_iters - min_iters) / 2) - 1
        self.forecast_horizon = forecast_horizon
        self.readout = readout

        self.forecaster = CoRNN1DForecastModel(
            seq_length,
            c_in,
            c_mid, 
            c_out,
            forecast_horizon=forecast_horizon,
            dt=dt,
            T=T,
            readout=readout,
            persistent_state=True  # Enable persistent state for online learning
        )
        
    def forward(self, x, hidden_state=None, return_hidden_state=False):
        """
        Forward pass with optional persistent hidden state
        
        Args:
            x: Input tensor of shape (batch_size, features, time_steps)
            hidden_state: Optional tuple of previous (hy, hz) states for online learning
            return_hidden_state: Whether to return the updated hidden state
            
        Returns:
            output: Model predictions (batch_size, forecast_horizon)
            y_seq: Sequence of hidden states
            new_hidden_state: Updated hidden state for next prediction (only if return_hidden_state=True)
        """
        result = self.forecaster(x, hidden_state=hidden_state)
        
        # If return_hidden_state is True, return all three values
        # Otherwise, only return the first two (predictions and hidden states sequence)
        if return_hidden_state:
            return result
        else:
            return result[0], result[1]


class CoRNN1DForecastModel(nn.Module):
    """
    An enhanced 1D CoRNN model for financial time series forecasting:
      - Input:  (B, features, time_steps)
      - Output: (B, forecast_horizon) for multi-step forecasting
    """
    def __init__(self, 
                 seq_length,
                 c_in,
                 c_mid,
                 c_out,
                 forecast_horizon=1,  # Number of future steps to predict
                 dt=0.5,              # ODE time step
                 T=50,                # Number of RNN unroll steps
                 readout="last",
                 fc_readout=None,
                 persistent_state=False):
        super().__init__()

        self.seq_length = seq_length
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.forecast_horizon = forecast_horizon
        self.dt = dt
        self.T = T
        self.n_hid = seq_length  # flattened sequence length
        self.K = self.T//2 + 1
        self.readout_type = readout
        self.fc_readout = fc_readout
        self.persistent_state = persistent_state

        # ====== 1) Encoders for omega, alpha, and init hy  ======
        #  A) omega_encoder (1D conv)
        self.omega_encoder = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        #  B) alpha_encoder (1D conv)
        self.alpha_encoder = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        #  C) hy_encoder (for initial hidden y)
        self.hy_encoder = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(c_mid, c_out, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )

        # ====== 2) The coRNNCell  ======
        self.cell = coRNN1DCell(channels=c_out, seq_length=seq_length, dt=dt)

        # ====== 3) Readout networks ======
        # Create readout networks based on the readout type and forecast horizon
        self._create_readout_networks()

    def _create_readout_networks(self):
        """Create appropriate readout networks based on readout type and forecast horizon"""
        
        if self.fc_readout is None:
            if self.readout_type == "linear":
                # Linear readout with time projection
                self.fc_time = nn.Linear(self.T, self.K, bias=False)
                
                # For multi-step forecasting, output forecast_horizon values
                self.fc_readout = nn.Sequential(
                    nn.Linear(self.K * self.c_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.forecast_horizon)
                )
                
            elif self.readout_type in ["last", "mean"]:
                # For last or mean readout, use the final hidden state
                self.fc_readout = nn.Sequential(
                    nn.Linear(self.c_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.forecast_horizon)
                )
                
            elif self.readout_type == "fft":
                # FFT-based readout for detecting cycles
                # First half of FFT components (magnitude only)
                fft_features = min(self.T // 2 + 1, 8)  # Limit number of FFT components
                
                self.fc_readout = nn.Sequential(
                    nn.Linear(fft_features * self.c_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.forecast_horizon)
                )
                
            else:
                raise ValueError(f"Invalid readout type: {self.readout_type}")

    def forward(self, x, hidden_state=None):
        """
        x: (B, features, time_steps) financial time series input
        hidden_state: Optional tuple of (hy, hz) for persistent state
        returns: 
            - (B, forecast_horizon) predictions for future time steps
            - sequence of hidden states
            - new hidden state tuple (hy, hz) if persistent_state=True
        """
        B, C, L = x.shape
        
        # 1) Precompute omega, alpha from input x
        omega = self.omega_encoder(x)     # shape (B, c_out, L)
        alpha = self.alpha_encoder(x)     # shape (B, c_out, L)
        
        # 2) Initialize or use provided hidden states (hy, hz)
        if hidden_state is not None and self.persistent_state:
            hy, hz = hidden_state
        else:
            hy_init = self.hy_encoder(x)  # shape (B, c_out, L)
            hy = hy_init
            hz = torch.zeros_like(hy)     # start velocity at 0

        # 3) Run dynamics
        y_seq = []
        for t in range(self.T):
            hy, hz = self.cell(
                x_t=x,  # optional usage
                hy=hy, 
                hz=hz, 
                omega=omega, 
                alpha=alpha
            )
            y_seq.append(hy)  # each is (B, c_out, L)
            
        y_seq = torch.stack(y_seq, dim=1)  # (B, T, c_out, L)

        # 4) Readout based on selected strategy
        if self.readout_type == "linear":
            predictions = self._linear_readout(y_seq, B, L)
        elif self.readout_type == "last":
            predictions = self._last_readout(y_seq, B, L)
        elif self.readout_type == "mean":
            predictions = self._mean_readout(y_seq, B, L)
        elif self.readout_type == "fft":
            predictions = self._fft_readout(y_seq, B, L)
        else:
            # Default to last readout if unknown type
            predictions = self._last_readout(y_seq, B, L)
        
        if self.persistent_state:
            return predictions, y_seq, (hy, hz)
        else:
            return predictions, y_seq

    def _linear_readout(self, y_seq, B, L):
        """
        Linear readout that projects through time dimension first
        y_seq: (B, T, c_out, L)
        """
        y_seq = y_seq.reshape(B, self.T, self.c_out, -1)
        y_seq = y_seq.transpose(1, 3)  # (B, L, c_out, T)
        fft_vals = self.fc_time(y_seq)  # (B, L, c_out, K)
        fft_mag = fft_vals.transpose(1, 3)  # (B, K, c_out, L)
        fft_mag = fft_mag.reshape(B, self.K * self.c_out, -1)  # (B, K*c_out, L)
        fft_mag = fft_mag.mean(dim=2)  # (B, K*c_out)
        predictions = self.fc_readout(fft_mag)  # (B, forecast_horizon)
        return predictions

    def _last_readout(self, y_seq, B, L):
        """
        Use only the last hidden state for prediction
        y_seq: (B, T, c_out, L)
        """
        hy = y_seq[:, -1]  # (B, c_out, L)
        hy = hy.mean(dim=2)  # (B, c_out)
        predictions = self.fc_readout(hy)  # (B, forecast_horizon)
        return predictions

    def _mean_readout(self, y_seq, B, L):
        """
        Use mean of hidden states across time for smoother predictions
        y_seq: (B, T, c_out, L)
        """
        hy = y_seq.mean(dim=1)  # (B, c_out, L)
        hy = hy.mean(dim=2)  # (B, c_out)
        predictions = self.fc_readout(hy)  # (B, forecast_horizon)
        return predictions

    def _fft_readout(self, y_seq, B, L):
        """
        Use FFT of hidden states to capture frequency patterns
        y_seq: (B, T, c_out, L)
        """
        # Reshape for FFT processing
        y_seq = y_seq.mean(dim=3)  # Average over sequence length: (B, T, c_out)
        
        # Compute FFT over time dimension
        fft_output = torch.fft.rfft(y_seq, dim=1)  # (B, T//2+1, c_out)
        
        # Get magnitude of FFT components (absolute value)
        fft_mag = torch.abs(fft_output)  # (B, T//2+1, c_out)
        
        # Take only the first few components (most significant frequencies)
        fft_components = min(self.T // 2 + 1, 8)
        fft_mag = fft_mag[:, :fft_components, :]  # (B, fft_components, c_out)
        
        # Flatten for the readout network
        fft_features = fft_mag.reshape(B, fft_components * self.c_out)  # (B, fft_components*c_out)
        
        # Pass through readout network
        predictions = self.fc_readout(fft_features)  # (B, forecast_horizon)
        
        return predictions


# Utility functions for multi-step forecasting

def create_forecast_dataset(data, seq_length, forecast_horizon, train_ratio=0.8):
    """
    Create sliding window dataset for multi-step forecasting
    
    Args:
        data: Tensor of shape (time_steps, features)
        seq_length: Length of input sequence
        forecast_horizon: How many steps ahead to predict
        train_ratio: Ratio of data to use for training
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon, 0])  # Only predict first feature (price)
    
    X = torch.stack(X)
    y = torch.stack(y)
    
    # Transpose X to get (batch, features, time_steps)
    X = X.transpose(1, 2)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test

def online_forecast(model, data_stream, window_size, forecast_horizon, step_size=1):
    """
    Perform online multi-step forecasting on streaming financial data
    
    Args:
        model: Trained FinancialForecastModel
        data_stream: Tensor of shape (time_steps, features)
        window_size: Size of sliding window
        forecast_horizon: Number of steps to forecast
        step_size: How many steps to move forward for each prediction
        
    Returns:
        Predictions for each window
    """
    model.eval()
    predictions = []
    hidden_state = None
    
    for i in range(0, len(data_stream) - window_size - forecast_horizon + 1, step_size):
        window = data_stream[i:i+window_size]
        window = window.transpose(0, 1).unsqueeze(0)  # (1, features, window_size)
        
        with torch.no_grad():
            if hidden_state is not None:
                output, _, hidden_state = model(window, hidden_state=hidden_state, return_hidden_state=True)
            else:
                output, _, hidden_state = model(window, return_hidden_state=True)
            
        predictions.append(output.squeeze(0))
    
    return torch.stack(predictions)