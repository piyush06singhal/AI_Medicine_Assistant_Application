"""
Medical Image Disease Predictor
Provides disease prediction from medical images using trained CNN models.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImagePredictor:
    """Medical image disease predictor using trained CNN models."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the image predictor.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use for inference ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = []
        self.model_info = None
        self.transform = None
        
        # Load model information
        self._load_model_info()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the device for inference."""
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
    
    def _load_model_info(self):
        """Load model information and metadata."""
        try:
            # Load model info from JSON file
            info_path = self.model_path / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                self.class_names = self.model_info.get('class_names', [])
            else:
                # Try to load from checkpoint
                checkpoint_path = self.model_path / "best_model.pth"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    self.class_names = checkpoint.get('class_names', [])
                    self.model_info = {'class_names': self.class_names}
                else:
                    logger.warning("No model info found, using default settings")
                    self.class_names = []
                    self.model_info = {}
            
            logger.info(f"Loaded model info with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            self.class_names = []
            self.model_info = {}
    
    def load_model(self, model_type: str = 'resnet50'):
        """
        Load the trained model.
        
        Args:
            model_type: Type of model to load
        """
        try:
            # Import model creation function
            from cv_models.models.medical_cnn import create_medical_model
            
            # Create model
            self.model = create_medical_model(
                model_type=model_type,
                num_classes=len(self.class_names)
            )
            
            # Load model weights
            checkpoint_path = self.model_path / "best_model.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model from best_model.pth")
            else:
                # Try final model
                final_path = self.model_path / "final_model.pth"
                if final_path.exists():
                    checkpoint = torch.load(final_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Loaded model from final_model.pth")
                else:
                    raise FileNotFoundError("No model checkpoint found")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self._setup_transforms()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        image_size = self.model_info.get('image_size', (224, 224))
        
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = self._load_image(image_path)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image_tensor = transformed['image']
            else:
                # Basic preprocessing if no transforms
                image = cv2.resize(image, (224, 224))
                image = image.astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
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
    
    def predict_disease_from_image(self, image_path: str, 
                                 return_probabilities: bool = True,
                                 top_k: int = 3) -> Dict:
        """
        Predict disease from medical image.
        
        Args:
            image_path: Path to the medical image
            return_probabilities: Whether to return prediction probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor)
                
                # Get prediction
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
                
                # Get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted_class].item()
            
            # Get predicted disease
            predicted_disease = self.class_names[predicted_class] if predicted_class < len(self.class_names) else 'Unknown'
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.class_names)))
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                disease_name = self.class_names[idx.item()] if idx.item() < len(self.class_names) else 'Unknown'
                predictions.append({
                    'disease': disease_name,
                    'confidence': prob.item(),
                    'rank': i + 1
                })
            
            # Get disease information
            related_symptoms, precautions = self._get_disease_info(predicted_disease)
            
            result = {
                'predicted_disease': predicted_disease,
                'confidence': confidence,
                'related_symptoms': related_symptoms,
                'precautions': precautions,
                'top_predictions': predictions if return_probabilities else None,
                'image_path': image_path,
                'class_probabilities': {
                    self.class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0]) 
                    if i < len(self.class_names)
                } if return_probabilities else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting disease from image {image_path}: {str(e)}")
            return {
                'error': str(e),
                'predicted_disease': None,
                'confidence': 0.0,
                'related_symptoms': [],
                'precautions': []
            }
    
    def _get_disease_info(self, disease_name: str) -> Tuple[List[str], List[str]]:
        """
        Get related symptoms and precautions for a disease.
        This is a mock implementation - replace with real medical database.
        
        Args:
            disease_name: Name of the predicted disease
            
        Returns:
            Tuple of (related_symptoms, precautions)
        """
        # Mock medical database - replace with real medical data
        disease_database = {
            'pneumonia': {
                'symptoms': ['cough with phlegm', 'fever', 'chills', 'shortness of breath', 'chest pain'],
                'precautions': ['Get plenty of rest', 'Stay hydrated', 'Take prescribed antibiotics', 'Monitor breathing']
            },
            'covid19': {
                'symptoms': ['fever', 'cough', 'shortness of breath', 'fatigue', 'loss of taste/smell'],
                'precautions': ['Isolate from others', 'Wear mask', 'Monitor symptoms', 'Seek medical care if severe']
            },
            'tuberculosis': {
                'symptoms': ['persistent cough', 'chest pain', 'coughing up blood', 'fatigue', 'weight loss'],
                'precautions': ['Complete antibiotic course', 'Isolate during treatment', 'Monitor for side effects']
            },
            'lung_cancer': {
                'symptoms': ['persistent cough', 'chest pain', 'shortness of breath', 'weight loss', 'fatigue'],
                'precautions': ['Avoid smoking', 'Regular checkups', 'Follow treatment plan', 'Maintain healthy lifestyle']
            },
            'skin_cancer': {
                'symptoms': ['unusual mole', 'skin lesion', 'asymmetrical growth', 'irregular borders', 'color changes'],
                'precautions': ['Avoid sun exposure', 'Use sunscreen', 'Regular skin checks', 'Early detection']
            },
            'melanoma': {
                'symptoms': ['dark mole', 'irregular shape', 'changing appearance', 'asymmetrical borders', 'color variation'],
                'precautions': ['Avoid UV exposure', 'Regular dermatologist visits', 'Self-examination', 'Early treatment']
            },
            'basal_cell_carcinoma': {
                'symptoms': ['pearly bump', 'waxy growth', 'flat lesion', 'brown scar-like area', 'bleeding sore'],
                'precautions': ['Sun protection', 'Regular skin exams', 'Avoid tanning', 'Early treatment']
            },
            'squamous_cell_carcinoma': {
                'symptoms': ['firm red nodule', 'flat lesion with scaly surface', 'new sore', 'wart-like growth'],
                'precautions': ['Sun protection', 'Regular checkups', 'Avoid UV exposure', 'Early detection']
            }
        }
        
        # Get disease info or return generic info
        disease_info = disease_database.get(disease_name.lower(), {
            'symptoms': ['Consult a healthcare professional for specific symptoms'],
            'precautions': ['Seek medical advice', 'Follow doctor recommendations', 'Monitor symptoms']
        })
        
        return disease_info['symptoms'], disease_info['precautions']
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict_disease_from_image(image_path)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path),
            'class_names': self.class_names,
            'model_info': self.model_info,
            'device': str(self.device)
        }

# Convenience function for easy usage
def predict_disease_from_image(image_path: str, model_path: str = "./models/medical_cnn") -> Dict:
    """
    Convenience function to predict disease from medical image.
    
    Args:
        image_path: Path to the medical image
        model_path: Path to the trained model directory
        
    Returns:
        Dictionary containing prediction results
    """
    predictor = MedicalImagePredictor(model_path)
    return predictor.predict_disease_from_image(image_path)
