"""
Multi-language Support for AI Medical Assistant
Supports English and Hindi for symptom input and interface.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from googletrans import Translator
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiLanguageSupport:
    """Multi-language support for English and Hindi."""
    
    def __init__(self):
        """Initialize multi-language support."""
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'hi': 'हिंदी (Hindi)'
        }
        
        # Medical terminology translations
        self.medical_terms = {
            'en': {
                'symptoms': 'Symptoms',
                'upload_image': 'Upload Medical Image',
                'analyze': 'Analyze Symptoms',
                'predicted_disease': 'Predicted Disease',
                'confidence': 'Confidence',
                'related_symptoms': 'Related Symptoms',
                'precautions': 'Precautions & Recommendations',
                'disclaimer': 'This is AI-based information. Consult a doctor before taking any action.',
                'enter_symptoms': 'Enter your symptoms here...',
                'supported_formats': 'Supported formats: PNG, JPG, JPEG, BMP, TIFF, DICOM',
                'be_detailed': 'Be as detailed as possible. Include duration, severity, and any other relevant information.',
                'system_status': 'System Status',
                'model_ready': 'Model: Ready',
                'model_not_available': 'Model: Not Available',
                'quick_stats': 'Quick Stats',
                'predictions_today': 'Predictions Today',
                'model_accuracy': 'Model Accuracy',
                'about': 'About',
                'nlp_models': 'NLP Models: For symptom text analysis',
                'cv_models': 'Computer Vision: For medical image analysis',
                'unified_prediction': 'Unified Prediction: Combined AI insights',
                'analysis_results': 'Analysis Results',
                'model_contributions': 'Model Contributions',
                'top_predictions': 'Top Predictions',
                'technical_details': 'Technical Details',
                'export_results': 'Export Results',
                'download_report': 'Download Report',
                'important_disclaimer': 'Important Medical Disclaimer',
                'ai_information': 'This is AI-based information. Consult a doctor before taking any action.',
                'important_note': 'Important: This AI Medical Assistant is for educational and research purposes only.',
                'not_for_clinical': 'It is not intended for clinical use or medical diagnosis.',
                'consult_healthcare': 'Always consult with qualified healthcare professionals for:',
                'medical_diagnosis': 'Medical diagnosis and treatment',
                'emergency_situations': 'Emergency medical situations',
                'serious_symptoms': 'Serious or persistent symptoms',
                'health_concerns': 'Any health concerns',
                'emergency_call': 'In case of emergency, call your local emergency services immediately.',
                'powered_by': 'Powered by Advanced AI Technology',
                'educational_purposes': 'For educational and research purposes only',
                'not_clinical_use': 'Not for clinical use',
                'copyright': '© 2024 AI Medical Assistant Team'
            },
            'hi': {
                'symptoms': 'लक्षण',
                'upload_image': 'चिकित्सा छवि अपलोड करें',
                'analyze': 'लक्षणों का विश्लेषण करें',
                'predicted_disease': 'भविष्यवाणी की गई बीमारी',
                'confidence': 'आत्मविश्वास',
                'related_symptoms': 'संबंधित लक्षण',
                'precautions': 'सावधानियां और सुझाव',
                'disclaimer': 'यह AI-आधारित जानकारी है। कोई भी कार्रवाई करने से पहले डॉक्टर से सलाह लें।',
                'enter_symptoms': 'अपने लक्षण यहाँ दर्ज करें...',
                'supported_formats': 'समर्थित प्रारूप: PNG, JPG, JPEG, BMP, TIFF, DICOM',
                'be_detailed': 'जितना संभव हो विस्तृत रहें। अवधि, गंभीरता और अन्य प्रासंगिक जानकारी शामिल करें।',
                'system_status': 'सिस्टम स्थिति',
                'model_ready': 'मॉडल: तैयार',
                'model_not_available': 'मॉडल: उपलब्ध नहीं',
                'quick_stats': 'त्वरित आंकड़े',
                'predictions_today': 'आज की भविष्यवाणियां',
                'model_accuracy': 'मॉडल सटीकता',
                'about': 'के बारे में',
                'nlp_models': 'NLP मॉडल: लक्षण पाठ विश्लेषण के लिए',
                'cv_models': 'कंप्यूटर विजन: चिकित्सा छवि विश्लेषण के लिए',
                'unified_prediction': 'एकीकृत भविष्यवाणी: संयुक्त AI अंतर्दृष्टि',
                'analysis_results': 'विश्लेषण परिणाम',
                'model_contributions': 'मॉडल योगदान',
                'top_predictions': 'शीर्ष भविष्यवाणियां',
                'technical_details': 'तकनीकी विवरण',
                'export_results': 'परिणाम निर्यात करें',
                'download_report': 'रिपोर्ट डाउनलोड करें',
                'important_disclaimer': 'महत्वपूर्ण चिकित्सा अस्वीकरण',
                'ai_information': 'यह AI-आधारित जानकारी है। कोई भी कार्रवाई करने से पहले डॉक्टर से सलाह लें।',
                'important_note': 'महत्वपूर्ण: यह AI मेडिकल असिस्टेंट केवल शैक्षिक और अनुसंधान उद्देश्यों के लिए है।',
                'not_for_clinical': 'यह नैदानिक उपयोग या चिकित्सा निदान के लिए अभिप्रेत नहीं है।',
                'consult_healthcare': 'हमेशा योग्य स्वास्थ्य देखभाल पेशेवरों से सलाह लें:',
                'medical_diagnosis': 'चिकित्सा निदान और उपचार',
                'emergency_situations': 'आपातकालीन चिकित्सा स्थितियां',
                'serious_symptoms': 'गंभीर या लगातार लक्षण',
                'health_concerns': 'कोई भी स्वास्थ्य चिंता',
                'emergency_call': 'आपातकाल की स्थिति में, तुरंत अपनी स्थानीय आपातकालीन सेवाओं को कॉल करें।',
                'powered_by': 'उन्नत AI तकनीक द्वारा संचालित',
                'educational_purposes': 'केवल शैक्षिक और अनुसंधान उद्देश्यों के लिए',
                'not_clinical_use': 'नैदानिक उपयोग के लिए नहीं',
                'copyright': '© 2024 AI मेडिकल असिस्टेंट टीम'
            }
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en' or 'hi')
        """
        try:
            # Simple heuristic detection
            hindi_chars = re.findall(r'[\u0900-\u097F]', text)
            english_chars = re.findall(r'[a-zA-Z]', text)
            
            if len(hindi_chars) > len(english_chars):
                return 'hi'
            else:
                return 'en'
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'en'  # Default to English
    
    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            if not text.strip():
                return text
            
            # Detect source language
            source_lang = self.detect_language(text)
            
            # If already in target language, return as is
            if source_lang == target_lang:
                return text
            
            # Translate using Google Translate
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return text  # Return original text if translation fails
    
    def get_ui_text(self, key: str, language: str = 'en') -> str:
        """
        Get UI text for the specified language.
        
        Args:
            key: Text key
            language: Language code
            
        Returns:
            Translated UI text
        """
        return self.medical_terms.get(language, {}).get(key, self.medical_terms['en'].get(key, key))
    
    def translate_symptoms(self, symptoms: str, target_lang: str = 'en') -> str:
        """
        Translate symptoms text to target language.
        
        Args:
            symptoms: Symptoms text
            target_lang: Target language code
            
        Returns:
            Translated symptoms
        """
        return self.translate_text(symptoms, target_lang)
    
    def get_language_selector(self) -> str:
        """Get language selector for Streamlit."""
        return st.selectbox(
            "🌐 Select Language / भाषा चुनें",
            options=list(self.supported_languages.keys()),
            format_func=lambda x: self.supported_languages[x],
            key="language_selector"
        )
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return self.supported_languages
    
    def is_hindi_text(self, text: str) -> bool:
        """Check if text contains Hindi characters."""
        return bool(re.search(r'[\u0900-\u097F]', text))
    
    def is_english_text(self, text: str) -> bool:
        """Check if text contains English characters."""
        return bool(re.search(r'[a-zA-Z]', text))
    
    def get_mixed_language_text(self, text: str) -> Tuple[str, str]:
        """
        Split text into Hindi and English parts.
        
        Args:
            text: Mixed language text
            
        Returns:
            Tuple of (hindi_text, english_text)
        """
        hindi_parts = re.findall(r'[\u0900-\u097F\s]+', text)
        english_parts = re.findall(r'[a-zA-Z\s]+', text)
        
        hindi_text = ' '.join(hindi_parts).strip()
        english_text = ' '.join(english_parts).strip()
        
        return hindi_text, english_text

# Global instance
multilang = MultiLanguageSupport()

def detect_and_translate_symptoms(symptoms: str, target_lang: str = 'en') -> Tuple[str, str]:
    """
    Detect language and translate symptoms if needed.
    
    Args:
        symptoms: Input symptoms text
        target_lang: Target language for translation
        
    Returns:
        Tuple of (translated_symptoms, detected_language)
    """
    detected_lang = multilang.detect_language(symptoms)
    
    if detected_lang != target_lang:
        translated_symptoms = multilang.translate_text(symptoms, target_lang)
        return translated_symptoms, detected_lang
    else:
        return symptoms, detected_lang

def get_ui_text(key: str, language: str = 'en') -> str:
    """Convenience function to get UI text."""
    return multilang.get_ui_text(key, language)

def translate_symptoms(symptoms: str, target_lang: str = 'en') -> str:
    """Convenience function to translate symptoms."""
    return multilang.translate_symptoms(symptoms, target_lang)
