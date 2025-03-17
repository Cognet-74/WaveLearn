# Financial Time Series Forecasting with CoRNN

This extension of the CoRNN (Continuous-time Recurrent Neural Network) model adapts the readout layer for financial time series forecasting. The implementation supports both single-step and multi-step forecasting with various readout strategies.

## Key Features

- **Multi-step Forecasting**: Predict multiple future time steps in a single forward pass
- **Multiple Readout Strategies**: Choose from different readout mechanisms optimized for time series
- **Online Learning**: Support for persistent state and online adaptation
- **Flexible Architecture**: Configurable for different financial data types and prediction horizons

## Readout Mechanisms

The model supports several readout strategies, each with different strengths:

1. **Last State Readout** (`last`): Uses only the final hidden state for prediction
   - Good for reactive predictions that prioritize recent information
   - Simple and computationally efficient
   - May be sensitive to noise in the most recent observations

2. **Mean State Readout** (`mean`): Averages hidden states across time
   - Produces smoother forecasts by considering the entire sequence
   - More robust to noise in individual time steps
   - May miss important recent changes in the time series

3. **FFT Readout** (`fft`): Uses frequency domain features from the hidden states
   - Excellent for detecting cyclical patterns and seasonality
   - Captures dominant frequency components in the time series
   - Particularly useful for data with strong periodic components

4. **Linear Readout** (`linear`): Projects through time dimension first
   - Flexible approach that can learn complex temporal dependencies
   - Combines aspects of both time and frequency domain analysis
   - More parameters, potentially better for complex patterns

## Usage Example

```python
# Create a model for 5-step forecasting using FFT readout
model = FinancialForecastModel(
    seq_length=50,        # Length of input sequence
    c_in=5,               # Number of input features
    c_mid=16,             # Hidden channels
    c_out=8,              # Output channels
    forecast_horizon=5,   # Predict 5 steps ahead
    readout="fft"         # Use FFT readout for cyclical patterns
)

# Forward pass with a batch of financial data
# x shape: (batch_size, features, time_steps)
predictions, hidden_states = model(x)
# predictions shape: (batch_size, forecast_horizon)
```

## Implementation Details

The readout layer has been modified to:

1. Take the hidden state sequence of shape `(batch, seq, hidden_dim)`
2. Process it according to the selected readout strategy
3. Output predictions of shape `(batch, forecast_horizon)`

### One-Step Prediction

For one-step prediction, set `forecast_horizon=1`:

```python
model = FinancialForecastModel(
    # ... other parameters
    forecast_horizon=1,
    readout="last"
)
```

### Multi-Step Forecasting

For multi-step forecasting, specify the desired horizon:

```python
model = FinancialForecastModel(
    # ... other parameters
    forecast_horizon=10,  # Predict 10 steps ahead
    readout="mean"
)
```

## Online Learning

The model supports online learning with persistent state:

```python
# Initial prediction without state
predictions, _, hidden_state = model(x_batch, return_hidden_state=True)

# Later predictions using the updated state
new_predictions, _, updated_state = model(
    new_data, 
    hidden_state=hidden_state,
    return_hidden_state=True
)
```

## Training for Multi-step Forecasting

When training for multi-step forecasting, ensure your target data matches the forecast horizon:

```python
# Create dataset with appropriate forecast horizon
X_train, y_train, X_test, y_test = create_forecast_dataset(
    data, 
    seq_length=50, 
    forecast_horizon=5
)

# Train with MSE loss comparing all predicted steps
loss = criterion(predictions, y_batch)  # y_batch shape: (batch, forecast_horizon)
```

## Performance Considerations

- **Last State Readout**: Fastest, lowest memory usage
- **Mean State Readout**: Slightly more computation but still efficient
- **FFT Readout**: More computationally intensive, better for cyclical data
- **Linear Readout**: Most parameters, potentially best for complex patterns

## Files

- `financial_cornn_forecast.py`: Main implementation of the forecasting model
- `financial_forecast_example.py`: Example usage and comparison of readout strategies
- `test_financial_forecast.py`: Unit tests for the forecasting functionality