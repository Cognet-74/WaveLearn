import torch
import torch.nn as nn
import numpy as np

"""
1D CoRNN model for financial time series forecasting.
Adapted from the 2D CoRNN segmentation model.
"""

class FinancialModel(nn.Module):
    def __init__(self, seq_length, c_in, c_mid, c_out, output_size, min_iters, max_iters, dt, T, readout="linear", fc_readout=None):
        """
        Args:
            seq_length: Length of input time series
            c_in: Number of input features (e.g., OHLCV data)
            c_mid: Number of channels in middle layers
            c_out: Number of output channels in hidden state
            output_size: Size of output (1 for regression, 2 for binary classification)
            min_iters: Minimum iterations
            max_iters: Maximum iterations
            dt: Time step for ODE
            T: Number of RNN unroll steps
            readout: Readout method
            fc_readout: Optional pre-defined readout network
        """
        super(FinancialModel, self).__init__()

        self.seq_length = seq_length
        self.c_in = c_in
        self.c_out = c_out
        self.K = int((max_iters - min_iters) / 2) - 1
        self.output_size = output_size
        self.readout = readout

        self.forecaster = CoRNN1DModel(
            seq_length,
            c_in,
            c_mid, 
            c_out,
            output_size=output_size,
            dt=dt,
            T=T,
            readout=readout,
            fc_readout=fc_readout,
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
            output: Model predictions
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


class CoRNN1DModel(nn.Module):
    """
    A 1D CoRNN model for financial time series:
      - Input:  (B, features, time_steps)
      - Output: (B, output_size) for regression or classification
    """
    def __init__(self, 
                 seq_length,
                 c_in,
                 c_mid,
                 c_out,
                 output_size=1,  # 1 for regression, 2+ for classification
                 dt=0.5,         # ODE time step
                 T=50,           # Number of RNN unroll steps
                 readout="linear",
                 fc_readout=None,
                 persistent_state=False):
        super().__init__()

        self.seq_length = seq_length
        self.c_in = c_in
        self.c_out = c_out
        self.output_size = output_size
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

        # ====== 3) Readout network ======
        if fc_readout is None:
            if readout == "linear":
                self.fc_time = nn.Linear(self.T, self.K, bias=False)
                self.fc_readout = nn.Sequential(
                    nn.Linear(self.K * c_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_size)
                )
            elif readout == "last" or readout == "mean_time" or readout == "max_time":
                self.fc_readout = nn.Sequential(
                    nn.Linear(c_out, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_size)
                )
            else:
                raise ValueError(f"Invalid readout type: {readout}")
        else:
            self.fc_readout = fc_readout

    def forward(self, x, hidden_state=None, return_fft=False):
        """
        x: (B, features, time_steps) financial time series input
        hidden_state: Optional tuple of (hy, hz) for persistent state
        returns: 
            - (B, output_size) predictions
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

        # 4) Readout
        if self.readout_type == "linear":
            logits = self._linear_readout(y_seq, B, L)
        elif self.readout_type == "last":
            logits = self._last_readout(y_seq, B, L)
        elif self.readout_type == "mean_time":
            logits = self._mean_time_readout(y_seq, B, L)
        elif self.readout_type == "max_time":
            logits = self._max_time_readout(y_seq, B, L)
        
        if self.persistent_state:
            return logits, y_seq, (hy, hz)
        else:
            return logits, y_seq

    def _linear_readout(self, y_seq, B, L):
        # y_seq: (B, T, c_out, L)
        y_seq = y_seq.reshape(B, self.T, self.c_out, -1)
        y_seq = y_seq.transpose(1, 3)  # (B, L, c_out, T)
        fft_vals = self.fc_time(y_seq)  # (B, L, c_out, K)
        fft_mag = fft_vals.transpose(1, 3)  # (B, K, c_out, L)
        fft_mag = fft_mag.reshape(B, self.K * self.c_out, -1)  # (B, K*c_out, L)
        fft_mag = fft_mag.mean(dim=2)  # (B, K*c_out)
        logits = self.fc_readout(fft_mag)  # (B, output_size)
        return logits

    def _last_readout(self, y_seq, B, L):
        # y_seq: (B, T, c_out, L)
        hy = y_seq[:, -1]  # (B, c_out, L)
        hy = hy.mean(dim=2)  # (B, c_out)
        logits = self.fc_readout(hy)  # (B, output_size)
        return logits

    def _mean_time_readout(self, y_seq, B, L):
        # y_seq: (B, T, c_out, L)
        hy = y_seq.mean(dim=1)  # (B, c_out, L)
        hy = hy.mean(dim=2)  # (B, c_out)
        logits = self.fc_readout(hy)  # (B, output_size)
        return logits

    def _max_time_readout(self, y_seq, B, L):
        # y_seq: (B, T, c_out, L)
        hy, _ = y_seq.max(dim=1)  # (B, c_out, L)
        hy = hy.mean(dim=2)  # (B, c_out)
        logits = self.fc_readout(hy)  # (B, output_size)
        return logits

    def _init_encoders(self):
        with torch.no_grad():
            # Initialize omega encoder conv layers
            for i, layer in enumerate(self.omega_encoder):
                if isinstance(layer, nn.Conv1d):
                    nn.init.dirac_(layer.weight)
                    nn.init.zeros_(layer.bias)

            # Initialize alpha encoder conv layers
            for i, layer in enumerate(self.alpha_encoder):
                if isinstance(layer, nn.Conv1d):
                    nn.init.constant_(layer.weight, 0.00)
                    nn.init.constant_(layer.bias, 0.00)

            # Initialize hy encoder conv layers
            for i, layer in enumerate(self.hy_encoder):
                if isinstance(layer, nn.Conv1d):
                    if i < len(self.hy_encoder) - 1:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    else:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
                    nn.init.zeros_(layer.bias)


class coRNN1DCell(nn.Module):
    def __init__(self, channels, seq_length, dt):
        """
        channels: number of input/output channels
        seq_length: length of the time series sequence
        dt: ODE timestep
        """
        super(coRNN1DCell, self).__init__()
        self.channels = channels
        self.dt = dt
        self.seq_length = seq_length

        # Local (1D) coupling: learnable Laplacian kernel
        self.Wy = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False,
            groups=1  # Each channel gets its own independent kernel
        )

        # 1D Laplacian kernel [1, -2, 1]
        laplacian_kernel = torch.tensor([1., -2., 1.]).reshape(1, 1, 3)

        # Initialize each channel's kernel with the 1D Laplacian
        with torch.no_grad():
            nn.init.constant_(self.Wy.weight, 0)
            for i in range(channels):
                self.Wy.weight[i:i+1, i:i+1].copy_(laplacian_kernel)

    def forward(self, x_t, hy, hz, omega, alpha):
        """
        x_t: [batch_size, channels, seq_length] for time step t
        hy:  [batch_size, channels, seq_length] hidden 'position'
        hz:  [batch_size, channels, seq_length] hidden 'velocity'
        omega, alpha: each [batch_size, channels, seq_length], precomputed from the initial frame
        """
        B = x_t.shape[0]

        # Apply the local 1D convolution
        spring_force = torch.tanh(self.Wy(hy))  # shape (B, channels, seq_length)
        
        # ODE: 
        #   hz_{t+1} = hz_t + dt*( spring_force - omega*hy - alpha*hz )
        #   hy_{t+1} = hy_t + dt*hz_{t+1}
        new_hz = hz + self.dt * (spring_force - omega * hy - alpha * hz)
        new_hy = hy + self.dt * new_hz

        return new_hy, new_hz


# Utility functions for financial time series

def create_financial_dataset(data, seq_length, forecast_horizon=1, train_ratio=0.8):
    """
    Create sliding window dataset from financial time series
    
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
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    
    X = torch.stack(X)
    y = torch.stack(y)
    
    # Transpose to get (batch, features, time_steps)
    X = X.transpose(1, 2)
    
    # For regression, we want to predict the next value
    # For classification, we can convert to binary (up/down)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test

def create_binary_labels(data, forecast_horizon=1):
    """
    Convert regression targets to binary classification (price up/down)
    
    Args:
        data: Tensor of shape (batch, forecast_horizon, features)
        forecast_horizon: How many steps ahead were predicted
        
    Returns:
        Binary labels (1 for price up, 0 for price down/same)
    """
    # Assuming the first feature is the price
    price_feature = data[:, :, 0]
    
    # Calculate if the forecasted price is higher than the last known price
    last_known_price = data[:, 0, 0]
    forecasted_price = price_feature[:, -1]
    
    # 1 if price went up, 0 if it went down or stayed the same
    binary_labels = (forecasted_price > last_known_price).long()
    
    return binary_labels

def online_prediction(model, data_stream, window_size, step_size=1):
    """
    Perform online prediction on streaming financial data
    
    Args:
        model: Trained FinancialModel
        data_stream: Tensor of shape (time_steps, features)
        window_size: Size of sliding window
        step_size: How many steps to move forward for each prediction
        
    Returns:
        Predictions for each window
    """
    model.eval()
    predictions = []
    hidden_state = None
    
    for i in range(0, len(data_stream) - window_size, step_size):
        window = data_stream[i:i+window_size]
        window = window.transpose(0, 1).unsqueeze(0)  # (1, features, window_size)
        
        with torch.no_grad():
            if hidden_state is not None:
                output, _, hidden_state = model(window, hidden_state=hidden_state, return_hidden_state=True)
            else:
                output, _, hidden_state = model(window, return_hidden_state=True)
            
        predictions.append(output.squeeze(0))
    
    return torch.stack(predictions)