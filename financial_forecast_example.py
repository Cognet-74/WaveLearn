import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from financial_cornn_forecast import FinancialForecastModel, create_forecast_dataset, online_forecast

"""
Example script demonstrating how to use the enhanced 1D CoRNN model for multi-step financial time series forecasting.
"""

def generate_synthetic_data(n_samples=1000, n_features=5, freq=0.1, noise=0.1):
    """Generate synthetic financial data with trends, seasonality and noise"""
    t = np.arange(n_samples)
    
    # Base price with trend and seasonality
    price = 100 + 0.05 * t + 10 * np.sin(freq * t) + noise * np.random.randn(n_samples)
    
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
    
    return torch.tensor(normalized_data, dtype=torch.float32), data_mean, data_std

def train_forecast_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    """Train the financial CoRNN forecasting model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    n_batches = len(X_train) // batch_size
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions, _, _ = model(X_batch)
            
            # For multi-step forecasting, compare with future values
            loss = criterion(predictions, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / n_batches
        train_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return train_losses

def evaluate_forecast_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    model.eval()
    with torch.no_grad():
        predictions, _, _ = model(X_test)
        mse = nn.MSELoss()(predictions, y_test)
        
        # Calculate directional accuracy for the first forecasted step
        actual_direction = (y_test[:, 0] > X_test[:, 0, -1]).float()
        pred_direction = (predictions[:, 0] > 0).float()
        directional_accuracy = (pred_direction == actual_direction).float().mean()
        
    return mse.item(), directional_accuracy.item(), predictions.detach()

def main():
    # Parameters
    seq_length = 50  # Length of input sequence
    forecast_horizon = 5  # How many steps ahead to predict
    c_in = 5  # Number of input features
    c_mid = 16  # Number of channels in middle layers
    c_out = 8  # Number of output channels in hidden state
    
    # Generate synthetic data
    data, data_mean, data_std = generate_synthetic_data(n_samples=1000, n_features=c_in)
    print(f"Generated synthetic data with shape: {data.shape}")
    
    # Create dataset for multi-step forecasting
    X_train, y_train, X_test, y_test = create_forecast_dataset(
        data, seq_length, forecast_horizon, train_ratio=0.8
    )
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Create models with different readout strategies
    readout_types = ["last", "mean", "fft", "linear"]
    models = {}
    results = {}
    
    for readout in readout_types:
        print(f"\n--- Training model with {readout} readout ---")
        
        # Create model
        models[readout] = FinancialForecastModel(
            seq_length=seq_length,
            c_in=c_in,
            c_mid=c_mid,
            c_out=c_out,
            forecast_horizon=forecast_horizon,
            min_iters=10,
            max_iters=100,
            dt=0.5,
            T=50,
            readout=readout
        )
        
        # Train model
        train_losses = train_forecast_model(
            models[readout], X_train, y_train, epochs=50, batch_size=32
        )
        
        # Evaluate model
        mse, dir_acc, test_preds = evaluate_forecast_model(models[readout], X_test, y_test)
        print(f"Test MSE: {mse:.6f}")
        print(f"Directional Accuracy: {dir_acc:.2%}")
        
        results[readout] = {
            'train_losses': train_losses,
            'mse': mse,
            'dir_acc': dir_acc,
            'test_preds': test_preds
        }
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    for readout in readout_types:
        plt.plot(results[readout]['train_losses'], label=f'{readout} readout')
    plt.title('Training Loss by Readout Type')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Plot test predictions for each readout type
    sample_idx = 0  # Sample index to visualize
    
    plt.subplot(2, 2, 2)
    # Get actual future values
    actual_future = y_test[sample_idx].numpy()
    # Get last known value from input sequence
    last_known = X_test[sample_idx, 0, -1].item()
    
    # Plot actual values
    x_axis = np.arange(forecast_horizon + 1)
    plt.plot(x_axis, np.concatenate([[last_known], actual_future]), 'k-', label='Actual')
    
    # Plot predictions for each readout
    for readout in readout_types:
        pred_future = results[readout]['test_preds'][sample_idx].numpy()
        plt.plot(x_axis[1:], pred_future, '--', label=f'{readout} readout')
    
    plt.title('Multi-step Forecast Comparison')
    plt.xlabel('Time Steps Ahead')
    plt.ylabel('Normalized Price')
    plt.legend()
    
    # Plot MSE comparison
    plt.subplot(2, 2, 3)
    readout_names = list(results.keys())
    mse_values = [results[r]['mse'] for r in readout_names]
    plt.bar(readout_names, mse_values)
    plt.title('Test MSE by Readout Type')
    plt.ylabel('MSE')
    
    # Plot directional accuracy comparison
    plt.subplot(2, 2, 4)
    dir_acc_values = [results[r]['dir_acc'] for r in readout_names]
    plt.bar(readout_names, dir_acc_values)
    plt.title('Directional Accuracy by Readout Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('financial_forecast_results.png')
    plt.show()
    
    print("\nResults saved to financial_forecast_results.png")
    
    # Demonstrate online forecasting with the best model
    best_readout = readout_names[np.argmin(mse_values)]
    print(f"\nDemonstrating online forecasting with {best_readout} readout model...")
    
    # Use a portion of the data for online forecasting
    online_data = data[800:]
    online_predictions = online_forecast(
        models[best_readout], 
        online_data, 
        window_size=seq_length,
        forecast_horizon=forecast_horizon
    )
    
    # Plot online forecasting results
    plt.figure(figsize=(12, 6))
    
    # Get actual values for comparison
    actual_values = online_data[seq_length:seq_length+len(online_predictions), 0].numpy()
    
    # Plot actual vs predicted first step
    plt.plot(actual_values, label='Actual')
    plt.plot(online_predictions[:, 0].numpy(), label='1-step Forecast')
    
    # Plot multi-step forecasts at selected points
    forecast_points = [10, 30, 50, 70]
    for point in forecast_points:
        if point < len(online_predictions):
            x_forecast = np.arange(point, point + forecast_horizon)
            y_forecast = online_predictions[point].numpy()
            plt.plot(x_forecast, y_forecast, 'r--', alpha=0.5)
    
    plt.title(f'Online Multi-step Forecasting with {best_readout.capitalize()} Readout')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Price')
    plt.legend()
    
    plt.savefig('online_forecast_results.png')
    plt.show()
    
    print("Online forecasting results saved to online_forecast_results.png")

if __name__ == "__main__":
    main()