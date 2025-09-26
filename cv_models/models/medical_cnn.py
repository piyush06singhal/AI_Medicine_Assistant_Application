"""
Medical CNN Models for Disease Prediction
Implements various CNN architectures and transfer learning for medical image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalCNN(nn.Module):
    """Custom CNN model for medical image classification."""
    
    def __init__(self, num_classes: int, input_channels: int = 3, dropout_rate: float = 0.5):
        """
        Initialize the custom CNN model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            dropout_rate: Dropout rate for regularization
        """
        super(MedicalCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers with ReLU and batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class MedicalResNet(nn.Module):
    """ResNet-based model for medical image classification using transfer learning."""
    
    def __init__(self, num_classes: int, model_name: str = 'resnet50', 
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the ResNet-based model.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of the ResNet model ('resnet18', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone for fine-tuning
        """
        super(MedicalResNet, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained ResNet model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

class MedicalEfficientNet(nn.Module):
    """EfficientNet-based model for medical image classification using transfer learning."""
    
    def __init__(self, num_classes: int, model_name: str = 'efficientnet_b0',
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the EfficientNet-based model.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of the EfficientNet model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone for fine-tuning
        """
        super(MedicalEfficientNet, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        try:
            import timm
            # Load pretrained EfficientNet model using timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            
            # Get feature dimension
            feature_dim = self.backbone.classifier.in_features
            
            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Replace the classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        except ImportError:
            logger.warning("timm not available, falling back to torchvision EfficientNet")
            # Fallback to torchvision EfficientNet
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = 1280
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                feature_dim = 1280
            else:
                raise ValueError(f"Unsupported EfficientNet model: {model_name}")
            
            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Replace the classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

class MedicalDenseNet(nn.Module):
    """DenseNet-based model for medical image classification using transfer learning."""
    
    def __init__(self, num_classes: int, model_name: str = 'densenet121',
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the DenseNet-based model.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of the DenseNet model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone for fine-tuning
        """
        super(MedicalDenseNet, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained DenseNet model
        if model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = 1024
        elif model_name == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            feature_dim = 2208
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            feature_dim = 1664
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            feature_dim = 1920
        else:
            raise ValueError(f"Unsupported DenseNet model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

class MedicalViT(nn.Module):
    """Vision Transformer-based model for medical image classification."""
    
    def __init__(self, num_classes: int, model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the Vision Transformer model.
        
        Args:
            num_classes: Number of output classes
            model_name: Name of the ViT model
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone for fine-tuning
        """
        super(MedicalViT, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        try:
            import timm
            # Load pretrained ViT model using timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            
            # Get feature dimension
            feature_dim = self.backbone.head.in_features
            
            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Replace the head
            self.backbone.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        except ImportError:
            logger.warning("timm not available, ViT model not supported")
            raise ImportError("timm is required for Vision Transformer models")
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)

def create_medical_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function to create medical image classification models.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        **kwargs: Additional arguments for model creation
        
    Returns:
        PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return MedicalCNN(num_classes=num_classes, **kwargs)
    elif model_type.startswith('resnet'):
        model_name = model_type.replace('resnet', 'resnet')
        return MedicalResNet(num_classes=num_classes, model_name=model_name, **kwargs)
    elif model_type.startswith('efficientnet'):
        return MedicalEfficientNet(num_classes=num_classes, model_name=model_type, **kwargs)
    elif model_type.startswith('densenet'):
        return MedicalDenseNet(num_classes=num_classes, model_name=model_type, **kwargs)
    elif model_type.startswith('vit'):
        return MedicalViT(num_classes=num_classes, model_name=model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of available model types and their variants.
    
    Returns:
        Dictionary mapping model types to their variants
    """
    return {
        'cnn': ['cnn'],
        'resnet': ['resnet18', 'resnet50', 'resnet101', 'resnet152'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
        'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'vit': ['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224']
    }
