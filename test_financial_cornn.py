import torch
import numpy as np
from financial_cornn import FinancialModel, create_financial_dataset

"""
Simple test script to verify the 1D CoRNN implementation for financial time series.
"""

def test_model_forward_pass():
    """Test basic forward pass of the model"""
    print("Testing model forward pass...")
    
    # Model parameters
    seq_length = 20
    c_in = 5
    c_mid = 8
    c_out = 4
    output_size = 1
    
    # Create model
    model = FinancialModel(
        seq_length=seq_length,
        c_in=c_in,
        c_mid=c_mid,
        c_out=c_out,
        output_size=output_size,
        min_iters=10,
        max_iters=50,
        dt=0.5,
        T=20,
        readout="linear"
    )
    
    # Create random input
    batch_size = 2
    x = torch.randn(batch_size, c_in, seq_length)
    
    # Forward pass
    predictions, hidden_states = model(x)
    
    # Check output shapes
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    assert predictions.shape == (batch_size, output_size), "Incorrect predictions shape"
    assert hidden_states.shape[0] == batch_size, "Incorrect batch size in hidden states"
    
    print("Basic forward pass test passed!")
    return True

def test_persistent_state():
    """Test persistent state functionality for online learning"""
    print("\nTesting persistent state functionality...")
    
    # Model parameters
    seq_length = 20
    c_in = 5
    c_mid = 8
    c_out = 4
    output_size = 1
    
    # Create model with persistent state
    model = FinancialModel(
        seq_length=seq_length,
        c_in=c_in,
        c_mid=c_mid,
        c_out=c_out,
        output_size=output_size,
        min_iters=10,
        max_iters=50,
        dt=0.5,
        T=20,
        readout="linear"
    )
    
    # Create random input sequences
    batch_size = 1
    x1 = torch.randn(batch_size, c_in, seq_length)
    x2 = torch.randn(batch_size, c_in, seq_length)
    
    # First forward pass to get initial state
    predictions1, _, hidden_state = model.forecaster(x1, hidden_state=None)
    
    # Second forward pass with persistent state
    predictions2, _, new_hidden_state = model.forecaster(x2, hidden_state=hidden_state)
    
    # Verify that hidden states are different
    hy1, hz1 = hidden_state
    hy2, hz2 = new_hidden_state
    
    hy_diff = torch.norm(hy1 - hy2).item()
    hz_diff = torch.norm(hz1 - hz2).item()
    
    print(f"Hidden state difference (hy): {hy_diff:.6f}")
    print(f"Hidden state difference (hz): {hz_diff:.6f}")
    
    assert hy_diff > 0, "Hidden states hy should be different"
    assert hz_diff > 0, "Hidden states hz should be different"
    
    print("Persistent state test passed!")
    return True

def test_with_synthetic_data():
    """Test model with synthetic financial data"""
    print("\nTesting with synthetic financial data...")
    
    # Generate synthetic data
    n_samples = 200
    n_features = 5
    
    # Create time series with trend and seasonality
    t = np.arange(n_samples)
    price = 100 + 0.05 * t + 10 * np.sin(0.1 * t) + np.random.randn(n_samples)
    volume = 1000 + 100 * np.random.randn(n_samples)
    features = np.column_stack([
        price, 
        volume, 
        np.convolve(price, np.ones(5)/5, mode='same'),  # MA5
        np.convolve(price, np.ones(10)/10, mode='same'),  # MA10
        np.abs(np.diff(price, prepend=price[0]))  # Volatility
    ])
    
    # Normalize
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    data = torch.tensor(features, dtype=torch.float32)
    
    # Create dataset
    seq_length = 20
    X_train, y_train, X_test, y_test = create_financial_dataset(
        data, seq_length, forecast_horizon=1, train_ratio=0.8
    )
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Create and train model
    model = FinancialModel(
        seq_length=seq_length,
        c_in=n_features,
        c_mid=8,
        c_out=4,
        output_size=1,
        min_iters=10,
        max_iters=50,
        dt=0.5,
        T=20,
        readout="linear"
    )
    
    # Train for a few iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("Training for 10 iterations...")
    for i in range(10):
        # Get a small batch
        batch_indices = torch.randint(0, len(X_train), (8,))
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices, 0, 0].unsqueeze(1)  # First feature (price)
        
        # Forward pass
        predictions, _, _ = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 2 == 0:
            print(f"Iteration {i+1}, Loss: {loss.item():.6f}")
    
    # Test prediction
    with torch.no_grad():
        test_predictions, _, _ = model(X_test[:5])
    
    print("\nTest predictions vs actual:")
    for i in range(5):
        print(f"Predicted: {test_predictions[i].item():.4f}, Actual: {y_test[i, 0, 0].item():.4f}")
    
    print("Synthetic data test completed!")
    return True

def run_all_tests():
    """Run all tests"""
    tests_passed = 0
    total_tests = 3
    
    if test_model_forward_pass():
        tests_passed += 1
    
    if test_persistent_state():
        tests_passed += 1
    
    if test_with_synthetic_data():
        tests_passed += 1
    
    print(f"\nTests completed: {tests_passed}/{total_tests} passed")

if __name__ == "__main__":
    print("=== Testing 1D CoRNN for Financial Time Series ===\n")
    run_all_tests()