# Financial Time Series Modeling with 1D CoRNN

This implementation adapts the 2D convolutional recurrent neural network (CoRNN) to a 1D version specifically designed for financial time series modeling. The wave-based dynamics of CoRNN are preserved while adapting the architecture to handle sequential, non-stationary financial data.

## Key Modifications

### 1. Conversion from 2D to 1D Convolutions

All 2D convolutional layers have been replaced with 1D equivalents:

```python
# Original 2D convolution
nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1)

# Converted to 1D convolution for time series
nn.Conv1d(c_in, c_mid, kernel_size=3, padding=1, stride=1)
```

### 2. Laplacian Kernel Adaptation

The 2D Laplacian kernel used for wave propagation has been replaced with a 1D version:

```python
# Original 2D Laplacian
laplacian_kernel = torch.tensor([
    [0.,  1.,  0.],
    [1., -4.0,  1.],
    [0.,  1.,  0.]
]).reshape(1, 1, 3, 3)

# Converted to 1D Laplacian
laplacian_kernel = torch.tensor([1., -2., 1.]).reshape(1, 1, 3)
```

This 1D Laplacian ensures proper wave behavior along the time dimension.

### 3. Input/Output Dimension Changes

- Original 2D CoRNN: Inputs of shape `(batch, channels, height, width)`
- Financial 1D CoRNN: Inputs of shape `(batch, features, time_steps)`

The output has been modified to produce either:
- Regression output: Next price prediction `(batch, 1)`
- Classification output: Market direction (up/down) `(batch, 2)`

### 4. Online Learning Capabilities

The model has been enhanced with persistent state functionality to support online learning:

```python
# Forward pass with optional persistent hidden state
def forward(self, x, hidden_state=None):
    # ...
    if self.persistent_state:
        return logits, y_seq, (hy, hz)  # Return updated hidden state
    else:
        return logits, y_seq
```

This allows the model to adapt to non-stationary financial data by updating its internal state as new data arrives.

### 5. Readout Adaptation

The readout mechanisms have been modified to handle 1D time series data and produce financial forecasts:

- Linear readout: Uses a linear projection of the hidden state sequence
- Last readout: Uses only the final hidden state
- Mean/Max time readout: Aggregates information across the time dimension

## Usage Example

```python
# Create model
model = FinancialModel(
    seq_length=50,      # Length of input sequence
    c_in=5,             # Number of input features (OHLCV)
    c_mid=16,           # Hidden channels
    c_out=8,            # Output channels
    output_size=1,      # 1 for regression, 2 for classification
    min_iters=10,
    max_iters=100,
    dt=0.5,             # ODE time step
    T=50,               # RNN unroll steps
    readout="linear"    # Readout method
)

# Forward pass (standard)
predictions, hidden_states = model(financial_data)

# Forward pass with persistent state (online learning)
predictions, hidden_states, new_state = model(financial_data, hidden_state=previous_state)
```

## Advantages for Financial Time Series

1. **Wave Dynamics**: The CoRNN's wave-based dynamics are preserved, allowing the model to capture long-range dependencies in financial data.

2. **Adaptive Learning**: The persistent state mechanism enables online learning, crucial for adapting to changing market conditions.

3. **Feature Integration**: The 1D convolutional structure efficiently integrates multiple financial indicators (price, volume, technical indicators).

4. **Temporal Patterns**: The wave propagation through the 1D Laplacian helps identify cyclical patterns and trends in financial data.

## Files

- `financial_cornn.py`: Implementation of the 1D CoRNN model for financial time series
- `financial_example.py`: Example script demonstrating usage with synthetic financial data

## Running the Example

```bash
python financial_example.py
```

This will:
1. Generate synthetic financial data
2. Train the 1D CoRNN model
3. Evaluate performance on test data
4. Demonstrate online learning capabilities
5. Plot and save results to `financial_cornn_results.png`