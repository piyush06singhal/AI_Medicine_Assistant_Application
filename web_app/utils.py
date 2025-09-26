"""
Utility functions for the AI Medical Assistant web application.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import logging
import base64
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_prediction_history(prediction_data: Dict[str, Any], 
                          history_file: str = "prediction_history.json"):
    """
    Save prediction to history file.
    
    Args:
        prediction_data: Dictionary containing prediction data
        history_file: Path to history file
    """
    try:
        history_path = Path(history_file)
        
        # Load existing history
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new prediction
        history.append({
            'timestamp': datetime.now().isoformat(),
            'symptoms': prediction_data.get('input_symptoms', ''),
            'image_uploaded': prediction_data.get('input_image') is not None,
            'predicted_disease': prediction_data.get('predicted_disease', 'Unknown'),
            'confidence': prediction_data.get('confidence', 0.0),
            'model_availability': prediction_data.get('model_availability', {})
        })
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[-100:]
        
        # Save updated history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Prediction saved to history: {len(history)} total predictions")
        
    except Exception as e:
        logger.error(f"Error saving prediction history: {str(e)}")

def load_prediction_history(history_file: str = "prediction_history.json") -> List[Dict[str, Any]]:
    """
    Load prediction history from file.
    
    Args:
        history_file: Path to history file
        
    Returns:
        List of prediction records
    """
    try:
        history_path = Path(history_file)
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            return history
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error loading prediction history: {str(e)}")
        return []

def create_prediction_summary(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of prediction data for display.
    
    Args:
        prediction_data: Dictionary containing prediction data
        
    Returns:
        Summary dictionary
    """
    summary = {
        'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
        'predicted_disease': prediction_data.get('predicted_disease', 'Unknown'),
        'confidence': prediction_data.get('confidence', 0.0),
        'related_symptoms': prediction_data.get('related_symptoms', []),
        'precautions': prediction_data.get('precautions', []),
        'top_predictions': prediction_data.get('top_predictions', []),
        'model_availability': prediction_data.get('model_availability', {}),
        'input_symptoms': prediction_data.get('input_symptoms', ''),
        'input_image': prediction_data.get('input_image', ''),
        'source': prediction_data.get('unified_prediction', {}).get('source', 'Unknown')
    }
    
    return summary

def format_confidence(confidence: float) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Formatted confidence string
    """
    if confidence >= 0.9:
        return f"{confidence:.1%} (Very High)"
    elif confidence >= 0.7:
        return f"{confidence:.1%} (High)"
    elif confidence >= 0.5:
        return f"{confidence:.1%} (Medium)"
    elif confidence >= 0.3:
        return f"{confidence:.1%} (Low)"
    else:
        return f"{confidence:.1%} (Very Low)"

def get_confidence_color(confidence: float) -> str:
    """
    Get color for confidence score.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Color string
    """
    if confidence >= 0.7:
        return "green"
    elif confidence >= 0.5:
        return "orange"
    else:
        return "red"

def create_model_status_display(model_availability: Dict[str, bool]) -> str:
    """
    Create model status display string.
    
    Args:
        model_availability: Dictionary with model availability
        
    Returns:
        Status display string
    """
    nlp_status = "✅" if model_availability.get('nlp_available', False) else "❌"
    cv_status = "✅" if model_availability.get('cv_available', False) else "❌"
    
    return f"NLP: {nlp_status} | CV: {cv_status}"

def validate_symptoms_input(symptoms: str) -> tuple[bool, str]:
    """
    Validate symptoms input.
    
    Args:
        symptoms: Symptoms text input
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not symptoms or not symptoms.strip():
        return False, "Please enter your symptoms."
    
    if len(symptoms.strip()) < 10:
        return False, "Please provide more detailed symptoms (at least 10 characters)."
    
    if len(symptoms) > 1000:
        return False, "Symptoms text is too long. Please keep it under 1000 characters."
    
    return True, ""

def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded image file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return True, ""  # Image is optional
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return False, "Image file is too large. Please upload a file smaller than 10MB."
    
    # Check file type
    allowed_types = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm']
    file_extension = Path(uploaded_file.name).suffix.lower()[1:]
    
    if file_extension not in allowed_types:
        return False, f"Unsupported file type. Please upload: {', '.join(allowed_types)}"
    
    return True, ""

def create_prediction_metrics(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metrics for prediction display.
    
    Args:
        prediction_data: Dictionary containing prediction data
        
    Returns:
        Metrics dictionary
    """
    metrics = {
        'confidence_score': prediction_data.get('confidence', 0.0),
        'confidence_level': format_confidence(prediction_data.get('confidence', 0.0)),
        'confidence_color': get_confidence_color(prediction_data.get('confidence', 0.0)),
        'model_status': create_model_status_display(prediction_data.get('model_availability', {})),
        'prediction_source': prediction_data.get('unified_prediction', {}).get('source', 'Unknown'),
        'related_symptoms_count': len(prediction_data.get('related_symptoms', [])),
        'precautions_count': len(prediction_data.get('precautions', [])),
        'top_predictions_count': len(prediction_data.get('top_predictions', []))
    }
    
    return metrics

def create_download_link(data: Dict[str, Any], filename: str = "prediction_results.json") -> str:
    """
    Create download link for prediction results.
    
    Args:
        data: Prediction data to download
        filename: Name of the download file
        
    Returns:
        HTML download link
    """
    try:
        # Convert data to JSON
        json_data = json.dumps(data, indent=2)
        
        # Create base64 encoded data
        b64_data = base64.b64encode(json_data.encode()).decode()
        
        # Create download link
        href = f'<a href="data:application/json;base64,{b64_data}" download="{filename}">📥 Download Results</a>'
        
        return href
        
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")
        return "Download not available"

def create_prediction_export(prediction_data: Dict[str, Any]) -> str:
    """
    Create exportable prediction report.
    
    Args:
        prediction_data: Dictionary containing prediction data
        
    Returns:
        Formatted report string
    """
    report = f"""
# AI Medical Assistant - Prediction Report

**Generated:** {prediction_data.get('timestamp', 'Unknown')}

## Input Information
- **Symptoms:** {prediction_data.get('input_symptoms', 'Not provided')}
- **Image Uploaded:** {'Yes' if prediction_data.get('input_image') else 'No'}

## Prediction Results
- **Predicted Disease:** {prediction_data.get('predicted_disease', 'Unknown')}
- **Confidence:** {format_confidence(prediction_data.get('confidence', 0.0))}
- **Prediction Source:** {prediction_data.get('unified_prediction', {}).get('source', 'Unknown')}

## Related Symptoms
{chr(10).join([f"- {symptom}" for symptom in prediction_data.get('related_symptoms', [])])}

## Precautions & Recommendations
{chr(10).join([f"- {precaution}" for precaution in prediction_data.get('precautions', [])])}

## Top Predictions
{chr(10).join([f"{i+1}. {pred['disease']} ({pred['confidence']:.1%})" for i, pred in enumerate(prediction_data.get('top_predictions', [])[:5])])}

## Model Status
- **NLP Model:** {'Available' if prediction_data.get('model_availability', {}).get('nlp_available') else 'Not Available'}
- **CV Model:** {'Available' if prediction_data.get('model_availability', {}).get('cv_available') else 'Not Available'}

---
**Disclaimer:** This is AI-based information for educational purposes only. Consult a healthcare professional for medical advice.
"""
    
    return report

def create_analytics_dashboard(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create analytics dashboard data from prediction history.
    
    Args:
        history: List of prediction records
        
    Returns:
        Analytics data dictionary
    """
    if not history:
        return {
            'total_predictions': 0,
            'disease_distribution': {},
            'confidence_stats': {},
            'model_usage': {},
            'recent_predictions': []
        }
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(history)
    
    # Total predictions
    total_predictions = len(df)
    
    # Disease distribution
    disease_distribution = df['predicted_disease'].value_counts().to_dict()
    
    # Confidence statistics
    confidence_stats = {
        'mean': df['confidence'].mean(),
        'median': df['confidence'].median(),
        'std': df['confidence'].std(),
        'min': df['confidence'].min(),
        'max': df['confidence'].max()
    }
    
    # Model usage
    model_usage = {
        'nlp_only': sum(1 for record in history if record.get('model_availability', {}).get('nlp_available') and not record.get('model_availability', {}).get('cv_available')),
        'cv_only': sum(1 for record in history if record.get('model_availability', {}).get('cv_available') and not record.get('model_availability', {}).get('nlp_available')),
        'both_models': sum(1 for record in history if record.get('model_availability', {}).get('nlp_available') and record.get('model_availability', {}).get('cv_available')),
        'no_models': sum(1 for record in history if not record.get('model_availability', {}).get('nlp_available') and not record.get('model_availability', {}).get('cv_available'))
    }
    
    # Recent predictions (last 10)
    recent_predictions = history[-10:] if len(history) >= 10 else history
    
    analytics = {
        'total_predictions': total_predictions,
        'disease_distribution': disease_distribution,
        'confidence_stats': confidence_stats,
        'model_usage': model_usage,
        'recent_predictions': recent_predictions
    }
    
    return analytics
