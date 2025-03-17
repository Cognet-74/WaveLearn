import torch
import unittest
from financial_cornn_forecast import FinancialForecastModel, create_forecast_dataset

class TestFinancialForecast(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.seq_length = 20
        self.forecast_horizon = 5
        self.batch_size = 4
        self.features = 3
        
        # Create random input data
        self.x = torch.randn(self.batch_size, self.features, self.seq_length)
        
        # Model parameters
        self.c_in = self.features
        self.c_mid = 8
        self.c_out = 4
        
    def test_model_output_shape(self):
        """Test that the model outputs the correct shape for different forecast horizons"""
        for forecast_horizon in [1, 3, 5, 10]:
            for readout in ["last", "mean", "fft", "linear"]:
                model = FinancialForecastModel(
                    seq_length=self.seq_length,
                    c_in=self.c_in,
                    c_mid=self.c_mid,
                    c_out=self.c_out,
                    forecast_horizon=forecast_horizon,
                    min_iters=10,
                    max_iters=20,
                    dt=0.5,
                    T=10,
                    readout=readout
                )
                
                # Forward pass
                output, hidden_seq = model(self.x)
                
                # Check output shape
                self.assertEqual(output.shape, (self.batch_size, forecast_horizon),
                                f"Output shape incorrect for {readout} readout with horizon {forecast_horizon}")
    
    def test_persistent_state(self):
        """Test that the model correctly uses and updates persistent state"""
        model = FinancialForecastModel(
            seq_length=self.seq_length,
            c_in=self.c_in,
            c_mid=self.c_mid,
            c_out=self.c_out,
            forecast_horizon=self.forecast_horizon,
            readout="last"
        )
        
        # First forward pass without hidden state
        output1, _, hidden_state = model(self.x, return_hidden_state=True)
        
        # Second forward pass with provided hidden state
        output2, _, updated_hidden_state = model(self.x, hidden_state=hidden_state, return_hidden_state=True)
        
        # Check that hidden states are different
        hy1, hz1 = hidden_state
        hy2, hz2 = updated_hidden_state
        
        # The hidden states should be updated
        self.assertFalse(torch.allclose(hy1, hy2), "Hidden state 'hy' was not updated")
        self.assertFalse(torch.allclose(hz1, hz2), "Hidden state 'hz' was not updated")
    
    def test_dataset_creation(self):
        """Test the create_forecast_dataset function"""
        # Create synthetic time series data
        time_steps = 100
        features = 3
        data = torch.randn(time_steps, features)
        
        # Create dataset
        seq_length = 20
        forecast_horizon = 5
        X_train, y_train, X_test, y_test = create_forecast_dataset(
            data, seq_length, forecast_horizon, train_ratio=0.8
        )
        
        # Check shapes
        expected_samples = time_steps - seq_length - forecast_horizon + 1
        expected_train_samples = int(expected_samples * 0.8)
        expected_test_samples = expected_samples - expected_train_samples
        
        self.assertEqual(X_train.shape, (expected_train_samples, features, seq_length),
                        "X_train shape is incorrect")
        self.assertEqual(y_train.shape, (expected_train_samples, forecast_horizon),
                        "y_train shape is incorrect")
        self.assertEqual(X_test.shape, (expected_test_samples, features, seq_length),
                        "X_test shape is incorrect")
        self.assertEqual(y_test.shape, (expected_test_samples, forecast_horizon),
                        "y_test shape is incorrect")
    
    def test_different_readouts(self):
        """Test that all readout types produce valid outputs"""
        readout_types = ["last", "mean", "fft", "linear"]
        
        for readout in readout_types:
            model = FinancialForecastModel(
                seq_length=self.seq_length,
                c_in=self.c_in,
                c_mid=self.c_mid,
                c_out=self.c_out,
                forecast_horizon=self.forecast_horizon,
                readout=readout
            )
            
            # Forward pass
            output, _ = model(self.x)
            
            # Check output is valid (no NaNs or infinities)
            self.assertFalse(torch.isnan(output).any(), f"{readout} readout produced NaN values")
            self.assertFalse(torch.isinf(output).any(), f"{readout} readout produced infinite values")

if __name__ == "__main__":
    unittest.main()