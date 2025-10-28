"""
Pitch and Onset Detection CNN Model

CNN-LSTM architecture for detecting pitch and note onsets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PitchOnsetCNN(nn.Module):
    """
    CNN-LSTM model for pitch and onset detection
    
    Architecture:
    - Multiple CNN layers for feature extraction
    - Bidirectional LSTM for temporal modeling
    - Separate heads for pitch and onset prediction
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 cnn_filters: list = [32, 64, 128],
                 cnn_kernel_sizes: list = [3, 3, 3],
                 cnn_pool_sizes: list = [2, 2, 2],
                 lstm_units: int = 256,
                 lstm_layers: int = 2,
                 dropout: float = 0.3,
                 pitch_classes: int = 88,
                 bidirectional: bool = True):
        """
        Initialize PitchOnsetCNN model
        
        Args:
            input_dim: Input feature dimension (e.g., 128 for mel spectrogram)
            cnn_filters: Number of filters for each CNN layer
            cnn_kernel_sizes: Kernel sizes for each CNN layer
            cnn_pool_sizes: Pool sizes for each CNN layer
            lstm_units: Number of LSTM units
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            pitch_classes: Number of pitch classes (88 for piano range)
            bidirectional: Use bidirectional LSTM
        """
        super(PitchOnsetCNN, self).__init__()
        
        self.input_dim = input_dim
        self.pitch_classes = pitch_classes
        self.bidirectional = bidirectional
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        
        for filters, kernel_size, pool_size in zip(cnn_filters, cnn_kernel_sizes, cnn_pool_sizes):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size),
                    nn.Dropout2d(dropout)
                )
            )
            in_channels = filters
        
        # Calculate dimension after CNN layers
        self.cnn_output_dim = self._calculate_cnn_output_dim(input_dim, cnn_pool_sizes)
        
        # LSTM layers
        lstm_input_dim = cnn_filters[-1] * self.cnn_output_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_units * 2 if bidirectional else lstm_units
        
        # Pitch prediction head
        self.pitch_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, pitch_classes),
            nn.Sigmoid()
        )
        
        # Onset prediction head
        self.onset_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized PitchOnsetCNN with {self.count_parameters()} parameters")
    
    def _calculate_cnn_output_dim(self, input_dim: int, pool_sizes: list) -> int:
        """Calculate output dimension after CNN pooling"""
        dim = input_dim
        for pool_size in pool_sizes:
            dim = dim // pool_size
        return dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Tuple of (pitch_output, onset_output)
        """
        batch_size, time_steps, features = x.shape
        
        # Add channel dimension: (batch, 1, time, features)
        x = x.unsqueeze(1)
        
        # CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, -1, x.size(2) * x.size(3))
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Prediction heads
        pitch_out = self.pitch_head(lstm_out)
        onset_out = self.onset_head(lstm_out)
        
        return pitch_out, onset_out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict(self, x: torch.Tensor, 
                pitch_threshold: float = 0.5,
                onset_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with thresholding
        
        Args:
            x: Input tensor
            pitch_threshold: Threshold for pitch detection
            onset_threshold: Threshold for onset detection
            
        Returns:
            Tuple of (pitch_predictions, onset_predictions)
        """
        self.eval()
        with torch.no_grad():
            pitch_out, onset_out = self.forward(x)
            
            # Apply thresholds
            pitch_pred = (pitch_out > pitch_threshold).float()
            onset_pred = (onset_out > onset_threshold).float()
            
        return pitch_pred, onset_pred


class SimpleCNN(nn.Module):
    """Simplified CNN model for faster training/testing"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 88):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened dimension
        self.flat_dim = 64 * (input_dim // 4)
        
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), x.size(1), -1)
        x = x.mean(dim=1)  # Average across time
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


def create_model(config: dict) -> nn.Module:
    """
    Create model from configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = config.get('architecture', 'cnn_lstm')
    
    if model_type == 'cnn_lstm':
        model = PitchOnsetCNN(
            input_dim=config['input_dim'],
            cnn_filters=config['cnn']['filters'],
            cnn_kernel_sizes=config['cnn']['kernel_size'],
            cnn_pool_sizes=config['cnn']['pool_size'],
            lstm_units=config['lstm']['units'],
            lstm_layers=config['lstm']['layers'],
            dropout=config['cnn']['dropout'],
            pitch_classes=config['output']['pitch_classes'],
            bidirectional=config['lstm']['bidirectional']
        )
    elif model_type == 'simple_cnn':
        model = SimpleCNN(
            input_dim=config.get('input_dim', 128),
            output_dim=config['output']['pitch_classes']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model")
    return model
