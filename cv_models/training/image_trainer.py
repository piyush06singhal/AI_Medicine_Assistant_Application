"""
Medical Image Model Trainer
Trains CNN models for medical image classification with transfer learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageTrainer:
    """Trainer for medical image classification models."""
    
    def __init__(self, model: nn.Module, device: str = 'auto', 
                 class_names: Optional[List[str]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training ('auto', 'cpu', 'cuda')
            class_names: List of class names for evaluation
        """
        self.model = model
        self.class_names = class_names or []
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("Using CPU")
        else:
            logger.info(f"Using {device}")
        
        return torch.device(device)
    
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, scheduler: Optional[Any] = None) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model for one epoch.
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy, predictions, true_labels)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 10, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, scheduler_type: str = 'step',
              class_weights: Optional[Dict[int, float]] = None,
              save_best: bool = True, save_dir: str = './models/medical_cnn') -> Dict[str, Any]:
        """
        Train the medical image classification model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_type: Type of learning rate scheduler
            class_weights: Class weights for imbalanced data
            save_best: Whether to save the best model
            save_dir: Directory to save the model
            
        Returns:
            Training results dictionary
        """
        # Setup loss function
        if class_weights:
            class_weights_tensor = torch.FloatTensor([class_weights[i] for i in range(len(class_weights))]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup scheduler
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        else:
            scheduler = None
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Update best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                if save_best:
                    self.save_model(save_path / "best_model.pth")
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Best Val Acc: {self.best_val_accuracy:.2f}%")
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model
        self.save_model(save_path / "final_model.pth")
        
        # Training time
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Plot training curves
        self.plot_training_curves(save_path / "training_curves.png")
        
        # Return results
        results = {
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'training_time': training_time,
            'model_path': str(save_path),
            'class_names': self.class_names
        }
        
        return results
    
    def evaluate(self, test_loader: DataLoader, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            model_path: Path to saved model (optional)
            
        Returns:
            Evaluation results dictionary
        """
        if model_path:
            self.load_model(model_path)
        
        # Evaluate
        test_loss, test_acc, predictions, true_labels = self.validate_epoch(test_loader, nn.CrossEntropyLoss())
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, self.class_names)
        
        results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist()
        }
        
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return results
    
    def save_model(self, path: str):
        """Save the model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_class': self.model.__class__.__name__
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', [])
        logger.info(f"Model loaded from {path}")
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {save_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: str = None):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if save_path:
            logger.info(f"Confusion matrix saved to {save_path}")
    
    def predict_single_image(self, image_tensor: torch.Tensor, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predict disease from a single image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results dictionary
        """
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            
            # Get prediction
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item()
            
            result = {
                'predicted_class': predicted_class,
                'predicted_disease': self.class_names[predicted_class] if predicted_class < len(self.class_names) else 'Unknown',
                'confidence': confidence
            }
            
            if return_probabilities:
                result['probabilities'] = probabilities[0].cpu().numpy().tolist()
                result['class_probabilities'] = {
                    self.class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0]) 
                    if i < len(self.class_names)
                }
            
            return result
