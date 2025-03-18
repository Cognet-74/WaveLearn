# WaveLearn: Financial Time Series Forecasting with Wave-Based Dynamics

WaveLearn is a comprehensive solution for financial time series forecasting that leverages wave-based dynamics through Convolutional Recurrent Neural Networks (CoRNN). This repository contains implementations of 1D CoRNN models specifically adapted for financial data, with support for both traditional batch training and online/streaming learning.

## Core Components

### 1. Financial CoRNN Model (`financial_cornn.py`)

A 1D adaptation of the Convolutional Recurrent Neural Network (CoRNN) for financial time series:

- **Wave-Based Dynamics**: Utilizes a 1D Laplacian kernel to model wave propagation through time
- **Multiple Readout Strategies**: Supports various ways to extract predictions from hidden states
- **Persistent State**: Maintains hidden state between predictions for online learning
- **Flexible Architecture**: Configurable for different financial data types and prediction tasks

```python
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
```

### 2. Financial Forecast Model (`financial_cornn_forecast.py`)

An enhanced version of the CoRNN model specifically designed for multi-step forecasting:

- **Multi-step Forecasting**: Predict multiple future time steps in a single forward pass
- **Advanced Readout Mechanisms**: Specialized readout strategies for time series forecasting
- **FFT Readout**: Frequency domain analysis for detecting cyclical patterns
- **Online Adaptation**: Support for continuous learning as new data arrives

```python
forecast_model = FinancialForecastModel(
    seq_length=50,        # Length of input sequence
    c_in=5,               # Number of input features
    c_mid=16,             # Hidden channels
    c_out=8,              # Output channels
    forecast_horizon=5,   # Predict 5 steps ahead
    readout="fft"         # Use FFT readout for cyclical patterns
)
```

### 3. Streaming Trainer (`streaming_trainer.py`)

A specialized trainer for online, streaming-based learning with financial time series:

- **Continuous Learning**: Updates the model as new data arrives
- **Sliding Window**: Processes data in a streaming fashion
- **Persistent State**: Maintains and updates hidden state between batches
- **Adaptive Learning**: Dynamically adjusts learning rate and resets hidden state
- **Rolling Validation**: Continuously evaluates model on recent data
- **Directional Accuracy**: Tracks ability to predict price movement direction

```python
trainer = StreamingTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    window_size=100,      # Size of sliding window
    step_size=1,          # Steps to advance window
    validation_size=50,   # Samples for rolling validation
    hidden_reset_prob=0.05, # Probability to reset hidden state
    lr_decay_steps=500,   # Steps between LR adjustments
    lr_decay_rate=0.99    # LR decay factor
)
```

## Example Scripts

### 1. Basic Financial Example (`financial_example.py`)

Demonstrates the core 1D CoRNN model for financial time series:

- Generates synthetic financial data with trends, seasonality, and noise
- Trains the model for next-step price prediction
- Evaluates performance on test data
- Demonstrates online learning capabilities
- Visualizes results and model performance

### 2. Multi-step Forecasting Example (`financial_forecast_example.py`)

Showcases the enhanced forecasting model with multi-step prediction:

- Compares different readout strategies (last, mean, FFT, linear)
- Trains models for multi-step forecasting
- Evaluates performance metrics including MSE and directional accuracy
- Visualizes forecasts at different time horizons
- Demonstrates online multi-step forecasting

### 3. Streaming Financial Example (`streaming_financial_example.py`)

Illustrates the streaming trainer for continuous online learning:

- Generates synthetic data with regime shifts
- Trains the model in an online fashion using a continuous data stream
- Maintains persistent hidden state between predictions
- Periodically resets hidden state to adapt to regime changes
- Visualizes online predictions against actual values

## Test Suite

The repository includes comprehensive tests to ensure the correctness of implementations:

- **Test Financial Forecast** (`test_financial_forecast.py`): Tests the multi-step forecasting model
- **Test Streaming Trainer** (`test_streaming_trainer.py`): Validates the streaming trainer functionality

## Key Features and Advantages

### Wave-Based Dynamics

The CoRNN architecture uses wave equations to model information propagation through the network:

- **Long-range Dependencies**: Wave dynamics help capture dependencies across longer time horizons
- **Temporal Patterns**: Wave propagation through the 1D Laplacian helps identify cyclical patterns
- **Stable Gradients**: Wave-based models can have more stable gradient flow during training

### Online Learning Capabilities

The models support continuous learning from streaming data:

- **Persistent State**: Hidden state can be maintained between predictions
- **Adaptive Learning**: Learning rate schedules and hidden state resets help adapt to changing markets
- **Rolling Validation**: Continuous evaluation on recent data ensures model remains effective

### Multiple Readout Strategies

Different readout mechanisms are optimized for various financial forecasting scenarios:

- **Last State Readout**: Prioritizes recent information for reactive predictions
- **Mean State Readout**: Produces smoother forecasts by considering the entire sequence
- **FFT Readout**: Excellent for detecting cyclical patterns and seasonality
- **Linear Readout**: Flexible approach that can learn complex temporal dependencies

## Usage Examples

### Training a Basic Model

```python
from financial_cornn import FinancialModel, create_financial_dataset

# Create model
model = FinancialModel(
    seq_length=50, c_in=5, c_mid=16, c_out=8, 
    output_size=1, min_iters=10, max_iters=100,
    dt=0.5, T=50, readout="linear"
)

# Create dataset
X_train, y_train, X_test, y_test = create_financial_dataset(
    data, seq_length=50, forecast_horizon=1, train_ratio=0.8
)

# Train model
train_model(model, X_train, y_train, epochs=50, batch_size=32)
```

### Multi-step Forecasting

```python
from financial_cornn_forecast import FinancialForecastModel, create_forecast_dataset

# Create model for 5-step forecasting
model = FinancialForecastModel(
    seq_length=50, c_in=5, c_mid=16, c_out=8,
    forecast_horizon=5, readout="fft"
)

# Create dataset for multi-step forecasting
X_train, y_train, X_test, y_test = create_forecast_dataset(
    data, seq_length=50, forecast_horizon=5, train_ratio=0.8
)

# Train and evaluate
train_forecast_model(model, X_train, y_train, epochs=50)
mse, dir_acc, predictions = evaluate_forecast_model(model, X_test, y_test)
```

### Online Learning with Streaming Trainer

```python
from financial_cornn_forecast import FinancialForecastModel
from streaming_trainer import train_streaming_financial_model

# Create model
model = FinancialForecastModel(
    seq_length=50, c_in=5, c_mid=32, c_out=16,
    forecast_horizon=1, readout="fft"
)

# Train with streaming trainer
trained_model, trainer = train_streaming_financial_model(
    model=model,
    data_stream=data,
    window_size=50,
    step_size=1,
    validation_size=100,
    hidden_reset_prob=0.05,
    max_steps=1500
)
```

## Performance Visualization

The implementation includes tools for visualizing model performance:

- Training and validation loss curves
- Directional accuracy tracking
- Comparison of different readout strategies
- Visualization of online predictions vs. actual values
- Multi-step forecast visualization

## Getting Started

1. Clone the repository
2. Install dependencies (PyTorch, NumPy, Matplotlib)
3. Run one of the example scripts to see the models in action:
   - `python financial_example.py` - Basic financial forecasting
   - `python financial_forecast_example.py` - Multi-step forecasting
   - `python streaming_financial_example.py` - Online learning with streaming data

## Advanced Usage

### Custom Readout Networks

You can define custom readout networks for specialized forecasting tasks:

```python
custom_readout = nn.Sequential(
    nn.Linear(c_out, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, forecast_horizon)
)

model = FinancialForecastModel(
    seq_length=50, c_in=5, c_mid=16, c_out=8,
    forecast_horizon=5, fc_readout=custom_readout
)
```

### Working with Real Financial Data

While the examples use synthetic data, the models can be applied to real financial data:

1. Preprocess your financial data (normalization, feature engineering)
2. Create sliding window datasets using the provided utility functions
3. Train the models using either batch or streaming approaches
4. Evaluate performance using financial metrics like directional accuracy

## Future Directions

- Integration with additional financial indicators and alternative data
- Ensemble methods combining multiple readout strategies
- Attention mechanisms for improved feature selection
- Reinforcement learning extensions for trading strategy development