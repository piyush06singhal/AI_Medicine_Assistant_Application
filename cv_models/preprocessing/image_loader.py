"""
Medical Image Data Loader and Preprocessor
Handles loading, preprocessing, and augmentation of medical image datasets.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = self._load_image(image_path)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {str(e)}")
            # Return a black image as fallback
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        # Try different loading methods
        try:
            # Method 1: OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        except:
            pass
        
        try:
            # Method 2: PIL
            image = Image.open(image_path)
            image = np.array(image.convert('RGB'))
            return image
        except:
            pass
        
        # Method 3: DICOM (for medical images)
        try:
            import pydicom
            ds = pydicom.dcmread(image_path)
            image = ds.pixel_array
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            return image
        except:
            pass
        
        raise ValueError(f"Could not load image: {image_path}")

class MedicalImageLoader:
    """Handles loading and preprocessing of medical image datasets."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image loader.
        
        Args:
            data_dir: Directory containing the image dataset
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.label_encoder = LabelEncoder()
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
    def load_from_folder_structure(self, class_folders: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load images from folder structure where each folder represents a class.
        
        Args:
            class_folders: List of class folder names. If None, auto-detect from directory structure.
            
        Returns:
            DataFrame with image paths and labels
        """
        if class_folders is None:
            # Auto-detect class folders
            class_folders = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        self.class_names = sorted(class_folders)
        logger.info(f"Found classes: {self.class_names}")
        
        data = []
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all image files in the class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm', '.nii', '.nii.gz'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(f"*{ext}")))
                image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            logger.info(f"Found {len(image_files)} images in class '{class_name}'")
            
            for image_path in image_files:
                data.append({
                    'image_path': str(image_path),
                    'class_name': class_name,
                    'class_id': class_idx
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} images from {len(self.class_names)} classes")
        
        return df
    
    def load_from_csv(self, csv_path: str, image_path_col: str = 'image_path', 
                     label_col: str = 'label') -> pd.DataFrame:
        """
        Load images from CSV file with image paths and labels.
        
        Args:
            csv_path: Path to CSV file
            image_path_col: Name of column containing image paths
            label_col: Name of column containing labels
            
        Returns:
            DataFrame with image paths and labels
        """
        df = pd.read_csv(csv_path)
        
        # Convert relative paths to absolute paths
        df[image_path_col] = df[image_path_col].apply(
            lambda x: str(self.data_dir / x) if not Path(x).is_absolute() else x
        )
        
        # Encode labels
        df['encoded_label'] = self.label_encoder.fit_transform(df[label_col])
        self.class_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"Loaded {len(df)} images from CSV")
        logger.info(f"Classes: {self.class_names}")
        
        return df
    
    def preprocess_images(self, df: pd.DataFrame, 
                         image_path_col: str = 'image_path',
                         label_col: str = 'encoded_label') -> Tuple[List[str], List[int]]:
        """
        Preprocess images and prepare for training.
        
        Args:
            df: DataFrame with image paths and labels
            image_path_col: Name of column containing image paths
            label_col: Name of column containing labels
            
        Returns:
            Tuple of (image_paths, labels)
        """
        # Filter out non-existent images
        valid_images = []
        for idx, row in df.iterrows():
            image_path = row[image_path_col]
            if Path(image_path).exists():
                valid_images.append(row)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        if not valid_images:
            raise ValueError("No valid images found")
        
        valid_df = pd.DataFrame(valid_images)
        
        self.image_paths = valid_df[image_path_col].tolist()
        self.labels = valid_df[label_col].tolist()
        
        logger.info(f"Preprocessed {len(self.image_paths)} valid images")
        
        return self.image_paths, self.labels
    
    def split_data(self, image_paths: List[str], labels: List[int],
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, test splits
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return splits
    
    def get_transforms(self, phase: str = 'train') -> A.Compose:
        """
        Get data augmentation transforms for training/validation.
        
        Args:
            phase: 'train' or 'val'
            
        Returns:
            Albumentations transform pipeline
        """
        if phase == 'train':
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:  # validation/test
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def create_datasets(self, splits: Dict[str, Tuple[List[str], List[int]]]) -> Dict[str, MedicalImageDataset]:
        """
        Create PyTorch datasets for train, validation, and test sets.
        
        Args:
            splits: Dictionary containing data splits
            
        Returns:
            Dictionary containing datasets
        """
        datasets = {}
        
        for phase in ['train', 'val', 'test']:
            if phase in splits:
                image_paths, labels = splits[phase]
                transform = self.get_transforms(phase)
                
                datasets[phase] = MedicalImageDataset(
                    image_paths=image_paths,
                    labels=labels,
                    transform=transform,
                    image_size=self.image_size
                )
        
        return datasets
    
    def create_dataloaders(self, datasets: Dict[str, MedicalImageDataset],
                          batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test sets.
        
        Args:
            datasets: Dictionary containing datasets
            batch_size: Batch size for training
            num_workers: Number of worker processes
            
        Returns:
            Dictionary containing DataLoaders
        """
        dataloaders = {}
        
        for phase, dataset in datasets.items():
            shuffle = (phase == 'train')
            dataloaders[phase] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return dataloaders
    
    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced', classes=unique_labels, y=labels
        )
        
        return dict(zip(unique_labels, class_weights))
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names
    
    def get_label_encoder(self) -> LabelEncoder:
        """Get the fitted label encoder."""
        return self.label_encoder
