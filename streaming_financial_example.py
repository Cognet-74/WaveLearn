import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from financial_cornn_forecast import FinancialForecastModel
from streaming_trainer import train_streaming_financial_model
import wandb

"""
Example script demonstrating how to use the streaming trainer with the wave-based model
for online financial time series forecasting.
"""

def generate_synthetic_data(n_samples=2000, n_features=5, freq=0.1, noise=0.1, regime_shifts=True):
    """
    Generate synthetic financial data with trends, seasonality, noise, and regime shifts.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        freq: Base frequency for seasonality
        noise: Noise level
        regime_shifts: Whether to include market regime shifts
        
    Returns:
        data: Normalized financial time series data
        data_mean: Mean of the data (for denormalization)
        data_std: Standard deviation of the data (for denormalization)
    """
    t = np.arange(n_samples)
    
    # Base price with trend and seasonality
    price = 100 + 0.05 * t + 10 * np.sin(freq * t) + noise * np.random.randn(n_samples)
    
    # Add regime shifts if enabled
    if regime_shifts:
        # Add sudden trend changes at random points
        shift_points = np.random.choice(np.arange(100, n_samples-100), 3, replace=False)
        for shift in shift_points:
            # Change the trend direction
            new_trend = np.random.uniform(-0.2, 0.2)
            price[shift:] += new_trend * np.arange(n_samples - shift)
            
            # Add volatility clusters
            volatility_multiplier = np.random.uniform(1.5, 3.0)
            cluster_length = np.random.randint(30, 100)
            price[shift:shift+cluster_length] += volatility_multiplier * noise * np.random.randn(cluster_length)
    
    # Volume (correlated with price changes)
    volume = 1000 + 500 * np.abs(np.diff(price, prepend=price[0])) + 100 * np.random.randn(n_samples)
    
    # Other features (e.g., technical indicators)
    ma_short = np.convolve(price, np.ones(5)/5, mode='same')  # Short moving average
    ma_long = np.convolve(price, np.ones(20)/20, mode='same')  # Long moving average
    volatility = np.abs(price - ma_short)  # Simple volatility measure
    
    # Combine features
    data = np.column_stack([price, volume, ma_short, ma_long, volatility])
    
    # Normalize data
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    normalized_data = (data - data_mean) / data_std
    
    return normalized_data, data_mean, data_std

def plot_data_with_predictions(data, predictions, window_size, step_size, data_mean, data_std, save_path=None):
    """
    Plot the original data with online predictions.
    
    Args:
        data: Original financial data
        predictions: Model predictions
        window_size: Size of the sliding window used for training
        step_size: Step size used for training
        data_mean: Mean of the data (for denormalization)
        data_std: Standard deviation of the data (for denormalization)
        save_path: Path to save the plot
    """
    # Denormalize data and predictions
    price_mean, price_std = data_mean[0], data_std[0]
    original_price = data[:, 0] * price_std + price_mean
    
    # Create time indices for predictions
    pred_indices = np.arange(window_size, window_size + len(predictions) * step_size, step_size)
    pred_indices = pred_indices[:len(predictions)]  # Ensure same length
    
    # Denormalize predictions
    denorm_predictions = predictions * price_std + price_mean
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    # Plot original price
    plt.plot(original_price, label='Actual Price', color='blue', alpha=0.7)
    
    # Plot predictions
    plt.scatter(pred_indices, denorm_predictions, label='Online Predictions', 
                color='red', alpha=0.5, s=20)
    
    # Add vertical lines for regime shifts (if known)
    regime_shifts = [500, 1000, 1500]
    for shift in regime_shifts:
        if shift < len(original_price):
            plt.axvline(x=shift, color='green', linestyle='--', alpha=0.5)
    
    plt.title('Online Financial Forecasting with Streaming Trainer')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    # Initialize wandb
    wandb.init(project="financial-streaming-forecast", name="streaming-cornn")
    
    # Parameters
    seq_length = 50  # Length of input sequence
    forecast_horizon = 1  # Single-step forecasting for simplicity
    c_in = 5  # Number of input features
    c_mid = 32  # Number of channels in middle layers
    c_out = 16  # Number of output channels in hidden state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic data with regime shifts
    data, data_mean, data_std = generate_synthetic_data(
        n_samples=2000, 
        n_features=c_in, 
        regime_shifts=True
    )
    print(f"Generated synthetic data with shape: {data.shape}")
    
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Create model with FFT readout (good for detecting cycles)
    model = FinancialForecastModel(
        seq_length=seq_length,
        c_in=c_in,
        c_mid=c_mid,
        c_out=c_out,
        forecast_horizon=forecast_horizon,
        min_iters=10,
        max_iters=100,
        dt=0.5,
        T=50,
        readout="fft"  # FFT readout for detecting cycles
    ).to(device)
    
    # Train model with streaming trainer
    trained_model, trainer = train_streaming_financial_model(
        model=model,
        data_stream=data,
        device=device,
        window_size=seq_length,
        step_size=1,  # Move forward one step at a time
        validation_size=100,
        validation_freq=50,
        hidden_reset_prob=0.05,  # 5% chance to reset hidden state
        learning_rate=0.001,
        lr_decay_steps=500,
        lr_decay_rate=0.99,
        max_steps=1500,  # Limit training steps
        log_freq=50,
        cp_path="best_streaming_model.pt",
        use_wandb=True
    )
    
    # Collect predictions from the trained model
    predictions = []
    hidden_state = None
    
    # Use the model to make predictions on the data
    trained_model.eval()
    with torch.no_grad():
        for i in range(0, len(data) - seq_length, 1):
            # Get window
            window = data[i:i+seq_length]
            window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
            window_tensor = window_tensor.transpose(0, 1).unsqueeze(0)  # (1, features, window_size)
            
            # Make prediction
            if hidden_state is not None:
                # Detach hidden state from previous computation graph
                detached_hidden = (hidden_state[0].detach(), hidden_state[1].detach())
                output, _, hidden_state = trained_model(
                    window_tensor,
                    hidden_state=detached_hidden,
                    return_hidden_state=True
                )
            else:
                output, _, hidden_state = trained_model(
                    window_tensor,
                    return_hidden_state=True
                )
            
            # Store prediction
            predictions.append(output.squeeze().cpu().item())
            
            # Reset hidden state with same probability as during training
            if np.random.rand() < 0.05:
                hidden_state = None
    
    # Plot results
    plot_data_with_predictions(
        data=data,
        predictions=np.array(predictions),
        window_size=seq_length,
        step_size=1,
        data_mean=data_mean,
        data_std=data_std,
        save_path="streaming_forecast_results.png"
    )
    
    # Close wandb
    wandb.finish()
    
    print("\nResults saved to streaming_forecast_results.png")

if __name__ == "__main__":
    main()