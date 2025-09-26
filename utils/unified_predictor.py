"""
Unified Disease Prediction System
Combines NLP and Computer Vision models for comprehensive disease prediction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from datetime import datetime
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import NLP and CV predictors
from nlp_models.models.disease_predictor import DiseasePredictor as NLPPredictor
from cv_models.models.image_predictor import MedicalImagePredictor as CVPredictor

# Import logging and multi-language support
from utils.query_logger import log_user_query
from utils.multilang_support import detect_and_translate_symptoms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDiseasePredictor:
    """Unified predictor that combines NLP and CV models for disease prediction."""
    
    def __init__(self, nlp_model_path: str = "./models/medical_bert", 
                 cv_model_path: str = "./models/medical_cnn"):
        """
        Initialize the unified predictor.
        
        Args:
            nlp_model_path: Path to the trained NLP model
            cv_model_path: Path to the trained CV model
        """
        self.nlp_model_path = Path(nlp_model_path)
        self.cv_model_path = Path(cv_model_path)
        
        self.nlp_predictor = None
        self.cv_predictor = None
        
        # Initialize predictors
        self._initialize_predictors()
        
        # Prediction weights (can be adjusted based on model performance)
        self.nlp_weight = 0.6  # Weight for NLP predictions
        self.cv_weight = 0.4   # Weight for CV predictions
        
    def _initialize_predictors(self):
        """Initialize NLP and CV predictors."""
        try:
            # Initialize NLP predictor
            if self.nlp_model_path.exists():
                self.nlp_predictor = NLPPredictor(str(self.nlp_model_path))
                logger.info("NLP predictor initialized successfully")
            else:
                logger.warning(f"NLP model not found at {self.nlp_model_path}")
                
            # Initialize CV predictor
            if self.cv_model_path.exists():
                self.cv_predictor = CVPredictor(str(self.cv_model_path))
                logger.info("CV predictor initialized successfully")
            else:
                logger.warning(f"CV model not found at {self.cv_model_path}")
                
        except Exception as e:
            logger.error(f"Error initializing predictors: {str(e)}")
    
    def predict_disease(self, symptoms: str, image_path: Optional[str] = None, 
                       return_probabilities: bool = True, 
                       top_k: int = 3, language: str = 'en') -> Dict[str, Any]:
        """
        Predict disease from symptoms text and optional image.
        
        Args:
            symptoms: Text description of symptoms
            image_path: Optional path to medical image
            return_probabilities: Whether to return prediction probabilities
            top_k: Number of top predictions to return
            language: Language of input symptoms ('en' or 'hi')
            
        Returns:
            Dictionary containing unified prediction results
        """
        # Start timing
        start_time = time.time()
        
        # Detect and translate symptoms if needed
        translated_symptoms, detected_language = detect_and_translate_symptoms(symptoms, 'en')
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_symptoms': symptoms,
            'translated_symptoms': translated_symptoms,
            'detected_language': detected_language,
            'input_language': language,
            'input_image': image_path,
            'nlp_prediction': None,
            'cv_prediction': None,
            'unified_prediction': None,
            'confidence': 0.0,
            'related_symptoms': [],
            'precautions': [],
            'top_predictions': [],
            'model_availability': {
                'nlp_available': self.nlp_predictor is not None,
                'cv_available': self.cv_predictor is not None
            }
        }
        
        # Get NLP prediction (use translated symptoms for better accuracy)
        if self.nlp_predictor and translated_symptoms.strip():
            try:
                nlp_result = self.nlp_predictor.predict_disease_from_text(
                    translated_symptoms, 
                    return_probabilities=return_probabilities,
                    top_k=top_k
                )
                results['nlp_prediction'] = nlp_result
                logger.info(f"NLP prediction: {nlp_result.get('predicted_disease', 'Unknown')}")
            except Exception as e:
                logger.error(f"NLP prediction failed: {str(e)}")
                results['nlp_prediction'] = {'error': str(e)}
        
        # Get CV prediction
        if self.cv_predictor and image_path and Path(image_path).exists():
            try:
                cv_result = self.cv_predictor.predict_disease_from_image(
                    image_path,
                    return_probabilities=return_probabilities,
                    top_k=top_k
                )
                results['cv_prediction'] = cv_result
                logger.info(f"CV prediction: {cv_result.get('predicted_disease', 'Unknown')}")
            except Exception as e:
                logger.error(f"CV prediction failed: {str(e)}")
                results['cv_prediction'] = {'error': str(e)}
        
        # Combine predictions
        results['unified_prediction'] = self._combine_predictions(
            results['nlp_prediction'], 
            results['cv_prediction'],
            return_probabilities,
            top_k
        )
        
        # Extract final results
        if results['unified_prediction']:
            results['predicted_disease'] = results['unified_prediction']['predicted_disease']
            results['confidence'] = results['unified_prediction']['confidence']
            results['related_symptoms'] = results['unified_prediction']['related_symptoms']
            results['precautions'] = results['unified_prediction']['precautions']
            results['top_predictions'] = results['unified_prediction'].get('top_predictions', [])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        # Log the query for model improvement
        try:
            query_id = log_user_query(
                symptoms_text=symptoms,
                image_path=image_path,
                predicted_disease=results.get('predicted_disease', 'Unknown'),
                confidence=results.get('confidence', 0.0),
                model_availability=results.get('model_availability', {}),
                prediction_source=results.get('unified_prediction', {}).get('source', 'Unknown'),
                processing_time=processing_time,
                language=detected_language
            )
            results['query_id'] = query_id
            logger.info(f"Query logged with ID: {query_id}")
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            results['query_id'] = None
        
        return results
    
    def _combine_predictions(self, nlp_result: Optional[Dict], cv_result: Optional[Dict],
                           return_probabilities: bool, top_k: int) -> Optional[Dict]:
        """
        Combine NLP and CV predictions into a unified result.
        
        Args:
            nlp_result: NLP prediction result
            cv_result: CV prediction result
            return_probabilities: Whether to return probabilities
            top_k: Number of top predictions
            
        Returns:
            Combined prediction result
        """
        # If only one model is available, return its result
        if nlp_result and not cv_result:
            return self._format_single_prediction(nlp_result, 'NLP')
        elif cv_result and not nlp_result:
            return self._format_single_prediction(cv_result, 'CV')
        elif not nlp_result and not cv_result:
            return {
                'predicted_disease': 'Unknown',
                'confidence': 0.0,
                'related_symptoms': ['Consult a healthcare professional'],
                'precautions': ['Seek medical advice'],
                'source': 'No models available'
            }
        
        # Both models available - combine predictions
        try:
            return self._weighted_combination(nlp_result, cv_result, return_probabilities, top_k)
        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            # Fallback to NLP prediction
            return self._format_single_prediction(nlp_result, 'NLP (fallback)')
    
    def _format_single_prediction(self, result: Dict, source: str) -> Dict:
        """Format single model prediction result."""
        if 'error' in result:
            return {
                'predicted_disease': 'Unknown',
                'confidence': 0.0,
                'related_symptoms': ['Consult a healthcare professional'],
                'precautions': ['Seek medical advice'],
                'source': f'{source} (error)'
            }
        
        return {
            'predicted_disease': result.get('predicted_disease', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'related_symptoms': result.get('related_symptoms', []),
            'precautions': result.get('precautions', []),
            'top_predictions': result.get('top_predictions', []),
            'source': source
        }
    
    def _weighted_combination(self, nlp_result: Dict, cv_result: Dict, 
                           return_probabilities: bool, top_k: int) -> Dict:
        """
        Combine predictions using weighted voting.
        
        Args:
            nlp_result: NLP prediction result
            cv_result: CV prediction result
            return_probabilities: Whether to return probabilities
            top_k: Number of top predictions
            
        Returns:
            Combined prediction result
        """
        # Check for errors
        nlp_error = 'error' in nlp_result
        cv_error = 'error' in cv_result
        
        if nlp_error and cv_error:
            return self._format_single_prediction(nlp_result, 'Both models failed')
        elif nlp_error:
            return self._format_single_prediction(cv_result, 'CV only')
        elif cv_error:
            return self._format_single_prediction(nlp_result, 'NLP only')
        
        # Get predictions and confidences
        nlp_disease = nlp_result.get('predicted_disease', 'Unknown')
        nlp_confidence = nlp_result.get('confidence', 0.0)
        cv_disease = cv_result.get('predicted_disease', 'Unknown')
        cv_confidence = cv_result.get('confidence', 0.0)
        
        # Weighted confidence calculation
        weighted_confidence = (self.nlp_weight * nlp_confidence + 
                             self.cv_weight * cv_confidence)
        
        # Determine final prediction
        if nlp_disease == cv_disease:
            # Both models agree
            final_disease = nlp_disease
            final_confidence = min(1.0, weighted_confidence * 1.2)  # Boost confidence for agreement
        else:
            # Models disagree - use weighted approach
            if nlp_confidence > cv_confidence:
                final_disease = nlp_disease
                final_confidence = weighted_confidence
            else:
                final_disease = cv_disease
                final_confidence = weighted_confidence
        
        # Combine related symptoms and precautions
        nlp_symptoms = nlp_result.get('related_symptoms', [])
        cv_symptoms = cv_result.get('related_symptoms', [])
        combined_symptoms = list(set(nlp_symptoms + cv_symptoms))[:5]  # Limit to 5 unique symptoms
        
        nlp_precautions = nlp_result.get('precautions', [])
        cv_precautions = cv_result.get('precautions', [])
        combined_precautions = list(set(nlp_precautions + cv_precautions))[:4]  # Limit to 4 unique precautions
        
        # Create top predictions if requested
        top_predictions = []
        if return_probabilities:
            # Combine top predictions from both models
            nlp_top = nlp_result.get('top_predictions', [])
            cv_top = cv_result.get('top_predictions', [])
            
            # Create combined predictions
            all_predictions = {}
            for pred in nlp_top:
                disease = pred['disease']
                if disease in all_predictions:
                    all_predictions[disease] = max(all_predictions[disease], pred['confidence'])
                else:
                    all_predictions[disease] = pred['confidence'] * self.nlp_weight
            
            for pred in cv_top:
                disease = pred['disease']
                if disease in all_predictions:
                    all_predictions[disease] += pred['confidence'] * self.cv_weight
                else:
                    all_predictions[disease] = pred['confidence'] * self.cv_weight
            
            # Sort by combined confidence
            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            
            for i, (disease, confidence) in enumerate(sorted_predictions[:top_k]):
                top_predictions.append({
                    'disease': disease,
                    'confidence': confidence,
                    'rank': i + 1
                })
        
        return {
            'predicted_disease': final_disease,
            'confidence': final_confidence,
            'related_symptoms': combined_symptoms,
            'precautions': combined_precautions,
            'top_predictions': top_predictions,
            'source': 'Combined NLP + CV',
            'nlp_contribution': {
                'disease': nlp_disease,
                'confidence': nlp_confidence
            },
            'cv_contribution': {
                'disease': cv_disease,
                'confidence': cv_confidence
            }
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of available models."""
        return {
            'nlp_model': {
                'available': self.nlp_predictor is not None,
                'path': str(self.nlp_model_path),
                'exists': self.nlp_model_path.exists()
            },
            'cv_model': {
                'available': self.cv_predictor is not None,
                'path': str(self.cv_model_path),
                'exists': self.cv_model_path.exists()
            },
            'weights': {
                'nlp_weight': self.nlp_weight,
                'cv_weight': self.cv_weight
            }
        }
    
    def update_weights(self, nlp_weight: float, cv_weight: float):
        """Update prediction weights."""
        total_weight = nlp_weight + cv_weight
        if total_weight > 0:
            self.nlp_weight = nlp_weight / total_weight
            self.cv_weight = cv_weight / total_weight
            logger.info(f"Updated weights - NLP: {self.nlp_weight:.2f}, CV: {self.cv_weight:.2f}")
        else:
            logger.warning("Invalid weights provided")

# Convenience function for easy usage
def predict_disease(symptoms: str, image_path: str = None, 
                   nlp_model_path: str = "./models/medical_bert",
                   cv_model_path: str = "./models/medical_cnn",
                   language: str = 'en') -> Dict[str, Any]:
    """
    Convenience function to predict disease from symptoms and optional image.
    
    Args:
        symptoms: Text description of symptoms
        image_path: Optional path to medical image
        nlp_model_path: Path to NLP model
        cv_model_path: Path to CV model
        language: Language of input symptoms ('en' or 'hi')
        
    Returns:
        Dictionary containing prediction results
    """
    predictor = UnifiedDiseasePredictor(nlp_model_path, cv_model_path)
    return predictor.predict_disease(symptoms, image_path, language=language)
