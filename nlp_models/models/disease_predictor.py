"""
Medical Disease Predictor
Provides disease prediction from symptom text using trained BERT/BioBERT models.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalBERTClassifier(nn.Module):
    """BERT-based classifier for medical disease prediction."""
    
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.3):
        super(MedicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class DiseasePredictor:
    """Medical disease predictor using trained BERT/BioBERT models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the disease predictor.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self.disease_info = None
        self.class_names = None
        
        # Load model information
        self._load_model_info()
        
    def _load_model_info(self):
        """Load model information and metadata."""
        try:
            with open(self.model_path / "model_info.json", 'r') as f:
                self.model_info = json.load(f)
            
            self.class_names = self.model_info['class_names']
            self.disease_info = self.model_info.get('disease_mapping', {})
            
            logger.info(f"Loaded model info for {len(self.class_names)} diseases")
            
        except Exception as e:
            logger.error(f"Error loading model info: {str(e)}")
            raise
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = MedicalBERTClassifier(
                model_name=self.model_info['model_name'],
                num_classes=self.model_info['num_classes']
            )
            
            # Load model weights
            model_file = self.model_path / "pytorch_model.bin"
            if model_file.exists():
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            else:
                # Try loading from HuggingFace format
                self.model = MedicalBERTClassifier.from_pretrained(self.model_path)
            
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text.
        
        Args:
            text: Raw symptom text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def predict_disease_from_text(self, symptoms: str, 
                                 return_probabilities: bool = True,
                                 top_k: int = 3) -> Dict:
        """
        Predict disease from symptom text.
        
        Args:
            symptoms: Symptom text description
            return_probabilities: Whether to return prediction probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(symptoms)
        
        if not cleaned_text:
            return {
                'error': 'No valid symptoms provided',
                'predicted_disease': None,
                'confidence': 0.0,
                'related_symptoms': [],
                'precautions': []
            }
        
        # Tokenize text
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.model_info['max_length'],
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.class_names)))
        
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            disease_name = self.class_names[idx.item()]
            predictions.append({
                'disease': disease_name,
                'confidence': prob.item(),
                'rank': i + 1
            })
        
        # Get predicted disease info
        predicted_disease = self.class_names[predicted_class]
        
        # Get related symptoms and precautions (mock data - replace with real medical data)
        related_symptoms, precautions = self._get_disease_info(predicted_disease)
        
        result = {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'related_symptoms': related_symptoms,
            'precautions': precautions,
            'top_predictions': predictions if return_probabilities else None,
            'input_symptoms': symptoms,
            'processed_symptoms': cleaned_text
        }
        
        return result
    
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
            'diabetes': {
                'symptoms': ['frequent urination', 'excessive thirst', 'fatigue', 'blurred vision', 'slow healing'],
                'precautions': ['Monitor blood sugar regularly', 'Maintain healthy diet', 'Exercise regularly', 'Take medications as prescribed']
            },
            'hypertension': {
                'symptoms': ['headache', 'dizziness', 'chest pain', 'shortness of breath', 'nosebleeds'],
                'precautions': ['Reduce sodium intake', 'Exercise regularly', 'Manage stress', 'Take blood pressure medications']
            },
            'asthma': {
                'symptoms': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'trouble sleeping'],
                'precautions': ['Avoid triggers', 'Use inhaler as prescribed', 'Keep rescue inhaler handy', 'Monitor symptoms']
            },
            'migraine': {
                'symptoms': ['severe headache', 'nausea', 'vomiting', 'sensitivity to light', 'sensitivity to sound'],
                'precautions': ['Identify triggers', 'Maintain regular sleep schedule', 'Stay hydrated', 'Consider preventive medications']
            },
            'pneumonia': {
                'symptoms': ['cough with phlegm', 'fever', 'chills', 'shortness of breath', 'chest pain'],
                'precautions': ['Get plenty of rest', 'Stay hydrated', 'Take prescribed antibiotics', 'Monitor breathing']
            },
            'gastroenteritis': {
                'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'fever'],
                'precautions': ['Stay hydrated', 'Avoid solid foods initially', 'Rest', 'Wash hands frequently']
            }
        }
        
        # Get disease info or return generic info
        disease_info = disease_database.get(disease_name.lower(), {
            'symptoms': ['Consult a healthcare professional for specific symptoms'],
            'precautions': ['Seek medical advice', 'Follow doctor recommendations', 'Monitor symptoms']
        })
        
        return disease_info['symptoms'], disease_info['precautions']
    
    def batch_predict(self, symptoms_list: List[str]) -> List[Dict]:
        """
        Predict diseases for multiple symptom texts.
        
        Args:
            symptoms_list: List of symptom texts
            
        Returns:
            List of prediction results
        """
        results = []
        for symptoms in symptoms_list:
            result = self.predict_disease_from_text(symptoms)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_info['model_name'],
            'num_classes': self.model_info['num_classes'],
            'max_length': self.model_info['max_length'],
            'class_names': self.class_names,
            'disease_mapping': self.disease_info
        }

# Convenience function for easy usage
def predict_disease_from_text(symptoms: str, model_path: str = "./models/medical_bert") -> Dict:
    """
    Convenience function to predict disease from symptom text.
    
    Args:
        symptoms: Symptom text description
        model_path: Path to the trained model directory
        
    Returns:
        Dictionary containing prediction results
    """
    predictor = DiseasePredictor(model_path)
    return predictor.predict_disease_from_text(symptoms)
