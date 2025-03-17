import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from financial_cornn import FinancialModel, create_financial_dataset, create_binary_labels, online_prediction

"""
Example script demonstrating how to use the 1D CoRNN model for financial time series forecasting.
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

def train_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    """Train the financial CoRNN model"""
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
            
            # For regression, we predict the next value of the first feature (price)
            loss = criterion(predictions, y_batch[:, 0, 0].unsqueeze(1))
            
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

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    model.eval()
    with torch.no_grad():
        predictions, _, _ = model(X_test)
        mse = nn.MSELoss()(predictions, y_test[:, 0, 0].unsqueeze(1))
        
        # Calculate directional accuracy (for binary prediction)
        actual_direction = (y_test[:, 0, 0] > X_test[:, 0, -1]).float()
        pred_direction = (predictions.squeeze() > 0).float()
        directional_accuracy = (pred_direction == actual_direction).float().mean()
        
    return mse.item(), directional_accuracy.item()

def demonstrate_online_learning(model, data, seq_length=50, n_updates=5):
    """Demonstrate online learning capabilities"""
    # Split data into initial training and streaming parts
    initial_data = data[:int(0.6 * len(data))]
    streaming_data = data[int(0.6 * len(data)):]
    
    # Prepare initial training data
    X_train, y_train, _, _ = create_financial_dataset(
        initial_data, seq_length, forecast_horizon=1, train_ratio=1.0
    )
    
    # Train model on initial data
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(100):  # Initial training
        predictions, _, _ = model(X_train)
        loss = criterion(predictions, y_train[:, 0, 0].unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Online prediction and learning
    hidden_state = None
    predictions = []
    true_values = []
    
    for i in range(len(streaming_data) - seq_length - 1):
        # Get current window
        window = streaming_data[i:i+seq_length]
        next_value = streaming_data[i+seq_length, 0]  # Next price
        
        # Prepare input
        X = window.unsqueeze(0).transpose(1, 2)  # (1, features, seq_length)
        
        # Predict with current state
        model.eval()
        with torch.no_grad():
            prediction, _, hidden_state = model(X, hidden_state=hidden_state)
        
        predictions.append(prediction.item())
        true_values.append(next_value.item())
        
        # Online update (if needed)
        if i % (len(streaming_data) // n_updates) == 0 and i > 0:
            model.train()
            # Create a small batch from recent data
            update_start = max(0, i - 32)
            X_update = torch.stack([
                streaming_data[j:j+seq_length].transpose(0, 1)
                for j in range(update_start, i)
            ])
            y_update = torch.stack([
                streaming_data[j+seq_length:j+seq_length+1, 0]
                for j in range(update_start, i)
            ])
            
            # Update model
            for _ in range(5):  # Few update steps
                pred, _, _ = model(X_update)
                loss = criterion(pred, y_update)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    return np.array(predictions), np.array(true_values)

def main():
    # Parameters
    seq_length = 50  # Length of input sequence
    forecast_horizon = 1  # How many steps ahead to predict
    c_in = 5  # Number of input features
    c_mid = 16  # Number of channels in middle layers
    c_out = 8  # Number of output channels in hidden state
    output_size = 1  # 1 for regression (price prediction)
    
    # Generate synthetic data
    data, data_mean, data_std = generate_synthetic_data(n_samples=1000, n_features=c_in)
    print(f"Generated synthetic data with shape: {data.shape}")
    
    # Create dataset
    X_train, y_train, X_test, y_test = create_financial_dataset(
        data, seq_length, forecast_horizon, train_ratio=0.8
    )
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Create model
    model = FinancialModel(
        seq_length=seq_length,
        c_in=c_in,
        c_mid=c_mid,
        c_out=c_out,
        output_size=output_size,
        min_iters=10,
        max_iters=100,
        dt=0.5,
        T=50,
        readout="linear"
    )
    print(f"Created 1D CoRNN model for financial forecasting")
    
    # Train model
    print("Training model...")
    train_losses = train_model(model, X_train, y_train, epochs=50, batch_size=32)
    
    # Evaluate model
    mse, dir_acc = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {mse:.6f}")
    print(f"Directional Accuracy: {dir_acc:.2%}")
    
    # Demonstrate online learning
    print("Demonstrating online learning...")
    online_preds, online_true = demonstrate_online_learning(model, data)
    online_mse = np.mean((online_preds - online_true) ** 2)
    print(f"Online prediction MSE: {online_mse:.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # Plot test predictions
    plt.subplot(2, 2, 2)
    with torch.no_grad():
        test_preds, _, _ = model(X_test[:100])
    plt.plot(y_test[:100, 0, 0].numpy(), label='Actual')
    plt.plot(test_preds.squeeze().numpy(), label='Predicted')
    plt.title('Test Predictions')
    plt.legend()
    
    # Plot online learning results
    plt.subplot(2, 2, 3)
    plt.plot(online_true, label='Actual')
    plt.plot(online_preds, label='Predicted')
    plt.title('Online Learning Predictions')
    plt.legend()
    
    # Plot original price data
    plt.subplot(2, 2, 4)
    original_price = data[:, 0] * data_std[0] + data_mean[0]
    plt.plot(original_price.numpy())
    plt.title('Original Price Data')
    
    plt.tight_layout()
    plt.savefig('financial_cornn_results.png')
    plt.show()
    
    print("Results saved to financial_cornn_results.png")

if __name__ == "__main__":
    main()