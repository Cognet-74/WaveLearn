import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

from utils import save_weights, set_random_seed

class StreamingTrainer:
    """
    Trainer for online, streaming-based learning with financial time series data.
    Continuously updates the model with new data as it arrives, maintaining state
    between updates and adapting to non-stationary market environments.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        window_size=100,
        step_size=1,
        validation_size=50,
        validation_freq=100,
        hidden_reset_prob=0.05,
        lr_decay_steps=500,
        lr_decay_rate=0.99,
        log_freq=100,
        cp_path=None,
        use_wandb=True
    ):
        """
        Initialize the streaming trainer.
        
        Args:
            model: The wave-based recurrent model
            optimizer: PyTorch optimizer (e.g., Adam)
            loss_fn: Loss function (e.g., MSE for regression)
            device: Device to run training on (CPU/GPU)
            window_size: Size of the sliding window for training
            step_size: Number of steps to advance the window each iteration
            validation_size: Number of samples to use for rolling validation
            validation_freq: How often to perform validation (in steps)
            hidden_reset_prob: Probability of resetting hidden state (0-1)
            lr_decay_steps: Steps between learning rate adjustments
            lr_decay_rate: Factor to decay learning rate by
            log_freq: How often to log metrics (in steps)
            cp_path: Path to save checkpoints
            use_wandb: Whether to log metrics to wandb
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.window_size = window_size
        self.step_size = step_size
        self.validation_size = validation_size
        self.validation_freq = validation_freq
        self.hidden_reset_prob = hidden_reset_prob
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.log_freq = log_freq
        self.cp_path = cp_path
        self.use_wandb = use_wandb
        
        # Initialize metrics tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_directional_accuracy': []
        }
        
        # Initialize validation buffer
        self.validation_buffer = deque(maxlen=validation_size)
        
    def get_next_window(self, data, current_idx):
        """
        Fetch the next sliding window from a financial time series.
        
        Args:
            data: Financial time series data (time_steps, features)
            current_idx: Current position in the data stream
            
        Returns:
            input_seq: Input sequence tensor
            target: Target value tensor
            next_idx: Next position in the data stream
        """
        end_idx = min(current_idx + self.window_size, len(data) - 1)
        
        if end_idx - current_idx < self.window_size:
            # Not enough data left
            return None, None, end_idx
        
        # Get input sequence and target
        input_seq = data[current_idx:end_idx]
        target = data[end_idx, 0]  # Predict the first feature (price)
        
        # Convert to tensors and add batch dimension
        input_seq = torch.tensor(input_seq, dtype=torch.float32).to(self.device)
        input_seq = input_seq.transpose(0, 1).unsqueeze(0)  # (1, features, window_size)
        target = torch.tensor(target, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Move forward by step_size
        next_idx = current_idx + self.step_size
        
        return input_seq, target, next_idx
    
    def should_reset_hidden_state(self):
        """
        Dynamically decide when to reset hidden states (e.g., market regime shift).
        Uses random sampling with a small probability to occasionally reset.
        
        Returns:
            bool: True if hidden state should be reset, False otherwise
        """
        return np.random.rand() < self.hidden_reset_prob
    
    def adjust_learning_rate(self):
        """
        Dynamically adjust learning rate based on training progress.
        Implements a step decay schedule.
        """
        new_lr = self.optimizer.param_groups[0]['lr'] * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        if self.use_wandb:
            wandb.log({"learning_rate": new_lr}, step=self.step)
    
    def train_streaming(self, data_stream, max_steps=None, validation_stream=None):
        """
        Trains the model in an online fashion using a continuous data stream.
        
        Args:
            data_stream: Financial time series data (time_steps, features)
            max_steps: Maximum number of steps to train for (None for unlimited)
            validation_stream: Optional separate validation data stream
            
        Returns:
            model: The trained model
        """
        self.model.train()
        hidden_state = None
        current_idx = 0
        
        # Progress tracking
        pbar = tqdm(total=max_steps if max_steps else len(data_stream), 
                   desc="Online Training")
        
        while True:
            # Get next window
            input_seq, target, next_idx = self.get_next_window(data_stream, current_idx)
            
            # Check if we've reached the end of the data stream
            if input_seq is None or (max_steps and self.step >= max_steps):
                break
                
            # Forward pass with persistent hidden state
            if hidden_state is not None:
                # Detach hidden state from previous computation graph
                detached_hidden = (hidden_state[0].detach(), hidden_state[1].detach())
                output, _, hidden_state = self.model(input_seq, hidden_state=detached_hidden, return_hidden_state=True)
            else:
                output, _, hidden_state = self.model(input_seq, return_hidden_state=True)
            
            # Compute loss - ensure dimensions match
            loss = self.loss_fn(output.squeeze(), target.squeeze())
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()  # No need for retain_graph since we detached the hidden state
            self.optimizer.step()
            
            # Add to validation buffer for rolling validation
            self.validation_buffer.append((input_seq.detach(), target.detach()))
            
            # Reset hidden state periodically to avoid drift
            if self.should_reset_hidden_state():
                hidden_state = None
                if self.use_wandb:
                    wandb.log({"hidden_state_reset": 1}, step=self.step)
            
            # Adjust learning rate periodically
            if self.step > 0 and self.step % self.lr_decay_steps == 0:
                self.adjust_learning_rate()
            
            # Perform rolling validation
            if self.step % self.validation_freq == 0:
                val_loss, val_dir_acc = self.evaluate_model(validation_stream)
                
                # Log validation metrics
                self.metrics_history['val_loss'].append(val_loss)
                self.metrics_history['val_directional_accuracy'].append(val_dir_acc)
                
                if self.use_wandb:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_directional_accuracy": val_dir_acc
                    }, step=self.step)
                
                # Save best model
                if val_loss < self.best_val_loss and self.cp_path:
                    self.best_val_loss = val_loss
                    save_weights(self.model.state_dict(), self.cp_path)
                    
                # Reset model to training mode
                self.model.train()
            
            # Log training metrics
            if self.step % self.log_freq == 0:
                self.metrics_history['train_loss'].append(loss.item())
                
                if self.use_wandb:
                    wandb.log({"train_loss": loss.item()}, step=self.step)
                
                pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})
            
            # Update progress
            current_idx = next_idx
            self.step += 1
            pbar.update(1)
        
        pbar.close()
        
        # Load best model if checkpoint path was provided
        if self.cp_path:
            self.model.load_state_dict(torch.load(self.cp_path))
            
        return self.model
    
    def evaluate_model(self, validation_stream=None):
        """
        Continuously evaluate model on a rolling validation set.
        
        Args:
            validation_stream: Optional separate validation data stream
            
        Returns:
            total_loss: Average loss on validation data
            directional_accuracy: Accuracy of predicting price direction
        """
        self.model.eval()
        total_loss = 0
        correct_directions = 0
        total_samples = 0
        
        with torch.no_grad():
            # If external validation stream is provided, use it
            if validation_stream is not None:
                val_idx = 0
                while True:
                    input_seq, target, next_idx = self.get_next_window(validation_stream, val_idx)
                    if input_seq is None:
                        break
                        
                    # Get prediction (without tracking gradients)
                    output, _ = self.model(input_seq)
                    
                    # Compute loss - ensure dimensions match
                    loss = self.loss_fn(output.squeeze(), target.squeeze())
                    total_loss += loss.item()
                    
                    # Compute directional accuracy
                    last_price = input_seq[0, 0, -1].item()  # Last price in input sequence
                    pred_direction = (output.squeeze().item() > last_price)
                    actual_direction = (target.item() > last_price)
                    correct_directions += int(pred_direction == actual_direction)
                    
                    total_samples += 1
                    val_idx = next_idx
            
            # Otherwise use the validation buffer
            else:
                for input_seq, target in self.validation_buffer:
                    # Get prediction
                    output, _ = self.model(input_seq)
                    
                    # Compute loss - ensure dimensions match
                    loss = self.loss_fn(output.squeeze(), target.squeeze())
                    total_loss += loss.item()
                    
                    # Compute directional accuracy
                    last_price = input_seq[0, 0, -1].item()  # Last price in input sequence
                    pred_direction = (output.squeeze().item() > last_price)
                    actual_direction = (target.item() > last_price)
                    correct_directions += int(pred_direction == actual_direction)
                    
                    total_samples += 1
        
        # Avoid division by zero
        if total_samples == 0:
            return 0.0, 0.0
            
        avg_loss = total_loss / total_samples
        dir_accuracy = correct_directions / total_samples
        
        return avg_loss, dir_accuracy
    
    def plot_metrics(self, save_path=None):
        """
        Plot training and validation metrics.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics_history['train_loss'], label='Training Loss')
        plt.plot(np.arange(0, self.step, self.validation_freq), 
                 self.metrics_history['val_loss'], label='Validation Loss')
        plt.title('Loss During Online Training')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot directional accuracy
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, self.step, self.validation_freq), 
                 self.metrics_history['val_directional_accuracy'], label='Directional Accuracy')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
        plt.title('Directional Accuracy During Online Training')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()


def train_streaming_financial_model(
    model, 
    data_stream, 
    device, 
    window_size=100,
    step_size=1,
    validation_size=50,
    validation_freq=100,
    hidden_reset_prob=0.05,
    learning_rate=0.001,
    lr_decay_steps=500,
    lr_decay_rate=0.99,
    max_steps=None,
    log_freq=100,
    cp_path=None,
    use_wandb=True
):
    """
    Convenience function to train a financial model with streaming data.
    
    Args:
        model: The wave-based recurrent model
        data_stream: Financial time series data (time_steps, features)
        device: Device to run training on (CPU/GPU)
        window_size: Size of the sliding window for training
        step_size: Number of steps to advance the window each iteration
        validation_size: Number of samples to use for rolling validation
        validation_freq: How often to perform validation (in steps)
        hidden_reset_prob: Probability of resetting hidden state (0-1)
        learning_rate: Initial learning rate
        lr_decay_steps: Steps between learning rate adjustments
        lr_decay_rate: Factor to decay learning rate by
        max_steps: Maximum number of steps to train for (None for unlimited)
        log_freq: How often to log metrics (in steps)
        cp_path: Path to save checkpoints
        use_wandb: Whether to log metrics to wandb
        
    Returns:
        model: The trained model
        trainer: The trainer object with metrics history
    """
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = StreamingTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        window_size=window_size,
        step_size=step_size,
        validation_size=validation_size,
        validation_freq=validation_freq,
        hidden_reset_prob=hidden_reset_prob,
        lr_decay_steps=lr_decay_steps,
        lr_decay_rate=lr_decay_rate,
        log_freq=log_freq,
        cp_path=cp_path,
        use_wandb=use_wandb
    )
    
    # Split data into training and validation
    val_split = int(0.8 * len(data_stream))
    train_data = data_stream[:val_split]
    val_data = data_stream[val_split:]
    
    # Train model
    model = trainer.train_streaming(
        data_stream=train_data,
        max_steps=max_steps,
        validation_stream=val_data
    )
    
    # Plot metrics
    trainer.plot_metrics(save_path="streaming_training_metrics.png")
    
    return model, trainer