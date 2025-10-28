"""
Model Trainer Module

Handles training and evaluation of transcription models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Callable
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TranscriptionDataset(Dataset):
    """Dataset for transcription training"""
    
    def __init__(self, features: np.ndarray, 
                 pitch_labels: np.ndarray,
                 onset_labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            features: Audio features (N, time, features)
            pitch_labels: Pitch labels (N, time, pitch_classes)
            onset_labels: Onset labels (N, time, 1)
        """
        self.features = torch.FloatTensor(features)
        self.pitch_labels = torch.FloatTensor(pitch_labels)
        self.onset_labels = torch.FloatTensor(onset_labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'pitch': self.pitch_labels[idx],
            'onset': self.onset_labels[idx]
        }


class ModelTrainer:
    """Train and evaluate transcription models"""
    
    def __init__(self, model: nn.Module, config: dict, device: str = 'cpu'):
        """
        Initialize ModelTrainer
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss functions
        self.pitch_criterion = nn.BCELoss()
        self.onset_criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pitch_loss': [],
            'train_onset_loss': [],
            'val_pitch_loss': [],
            'val_onset_loss': []
        }
        
        logger.info(f"Initialized ModelTrainer on {device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_pitch_loss = 0
        total_onset_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            features = batch['features'].to(self.device)
            pitch_labels = batch['pitch'].to(self.device)
            onset_labels = batch['onset'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pitch_out, onset_out = self.model(features)
            
            # Calculate losses
            pitch_loss = self.pitch_criterion(pitch_out, pitch_labels)
            onset_loss = self.onset_criterion(onset_out, onset_labels)
            loss = pitch_loss + onset_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_pitch_loss += pitch_loss.item()
            total_onset_loss += onset_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'pitch': pitch_loss.item(),
                'onset': onset_loss.item()
            })
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'pitch_loss': total_pitch_loss / n_batches,
            'onset_loss': total_onset_loss / n_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_pitch_loss = 0
        total_onset_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                features = batch['features'].to(self.device)
                pitch_labels = batch['pitch'].to(self.device)
                onset_labels = batch['onset'].to(self.device)
                
                # Forward pass
                pitch_out, onset_out = self.model(features)
                
                # Calculate losses
                pitch_loss = self.pitch_criterion(pitch_out, pitch_labels)
                onset_loss = self.onset_criterion(onset_out, onset_labels)
                loss = pitch_loss + onset_loss
                
                # Update metrics
                total_loss += loss.item()
                total_pitch_loss += pitch_loss.item()
                total_onset_loss += onset_loss.item()
        
        n_batches = len(val_loader)
        return {
            'loss': total_loss / n_batches,
            'pitch_loss': total_pitch_loss / n_batches,
            'onset_loss': total_onset_loss / n_batches
        }
    
    def train(self, train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             epochs: int = 100,
             checkpoint_dir: Optional[str] = None,
             early_stopping_patience: Optional[int] = None):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_pitch_loss'].append(train_metrics['pitch_loss'])
            self.history['train_onset_loss'].append(train_metrics['onset_loss'])
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Pitch: {train_metrics['pitch_loss']:.4f}, "
                       f"Onset: {train_metrics['onset_loss']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_pitch_loss'].append(val_metrics['pitch_loss'])
                self.history['val_onset_loss'].append(val_metrics['onset_loss'])
                
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Pitch: {val_metrics['pitch_loss']:.4f}, "
                           f"Onset: {val_metrics['onset_loss']:.4f}")
                
                # Save best model
                if checkpoint_dir and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.save_checkpoint(
                        Path(checkpoint_dir) / 'best_model.pth',
                        epoch, val_metrics
                    )
                    logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save periodic checkpoint
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    Path(checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pth',
                    epoch, train_metrics
                )
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        metrics = self.validate(test_loader)
        
        # Calculate additional metrics
        self.model.eval()
        all_pitch_preds = []
        all_pitch_labels = []
        all_onset_preds = []
        all_onset_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluation'):
                features = batch['features'].to(self.device)
                pitch_labels = batch['pitch']
                onset_labels = batch['onset']
                
                pitch_out, onset_out = self.model(features)
                
                all_pitch_preds.append(pitch_out.cpu())
                all_pitch_labels.append(pitch_labels)
                all_onset_preds.append(onset_out.cpu())
                all_onset_labels.append(onset_labels)
        
        # Concatenate all predictions
        pitch_preds = torch.cat(all_pitch_preds, dim=0)
        pitch_labels = torch.cat(all_pitch_labels, dim=0)
        onset_preds = torch.cat(all_onset_preds, dim=0)
        onset_labels = torch.cat(all_onset_labels, dim=0)
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        pitch_f1 = f1_score(
            pitch_labels.numpy().flatten() > 0.5,
            pitch_preds.numpy().flatten() > 0.5,
            average='binary'
        )
        
        onset_f1 = f1_score(
            onset_labels.numpy().flatten() > 0.5,
            onset_preds.numpy().flatten() > 0.5,
            average='binary'
        )
        
        metrics['pitch_f1'] = pitch_f1
        metrics['onset_f1'] = onset_f1
        
        logger.info(f"Evaluation - Loss: {metrics['loss']:.4f}, "
                   f"Pitch F1: {pitch_f1:.4f}, Onset F1: {onset_f1:.4f}")
        
        return metrics
