"""
AI Medical Assistant - Portfolio Project
Personal learning project with text, image, and audio input capabilities.
"""

import streamlit as st
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import base64
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.unified_predictor import UnifiedDiseasePredictor, predict_disease
from utils.multilang_support import MultiLanguageSupport, get_ui_text
from utils.audio_processor import AudioProcessor, record_and_transcribe, transcribe_uploaded_audio
from utils.query_logger import QueryLogger, query_logger
from web_app.utils import (
    save_prediction_history, load_prediction_history, create_prediction_summary,
    format_confidence, get_confidence_color, create_model_status_display,
    validate_symptoms_input, validate_image_file, create_prediction_metrics,
    create_download_link, create_prediction_export, create_analytics_dashboard
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
multilang = MultiLanguageSupport()
audio_processor = AudioProcessor()

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant - Portfolio Project",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
def load_custom_css():
    """Load custom CSS for styling the app."""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
    }
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Section styling */
    .section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    .section h2 {
        color: var(--primary-color);
        font-size: 1.8rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    .section h3 {
        color: var(--dark-color);
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    
    /* Input styling */
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 12px;
        font-size: 16px;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Audio section styling */
    .audio-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .audio-button {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .audio-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-disease {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Language selector styling */
    .language-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 2px solid #e9ecef;
    }
    
    /* Portfolio section */
    .portfolio-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .tech-item {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .prediction-disease {
            font-size: 2rem;
        }
        
        .tech-stack {
            flex-direction: column;
            align-items: center;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create the main header section."""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Medical Assistant</h1>
        <p>Portfolio Project - Multi-Modal Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

def create_portfolio_section():
    """Create portfolio information section."""
    st.markdown('<div class="portfolio-section">', unsafe_allow_html=True)
    st.markdown("""
    <h2>🎯 Portfolio Project Overview</h2>
    <p>This is a personal learning project demonstrating advanced AI/ML techniques for disease prediction using multiple input modalities.</p>
    
    <div class="tech-stack">
        <div class="tech-item">🤖 Machine Learning</div>
        <div class="tech-item">🧠 Deep Learning</div>
        <div class="tech-item">📝 Natural Language Processing</div>
        <div class="tech-item">👁️ Computer Vision</div>
        <div class="tech-item">🎤 Speech Recognition</div>
        <div class="tech-item">🌐 Multi-language Support</div>
        <div class="tech-item">📊 Data Analytics</div>
        <div class="tech-item">☁️ Cloud Deployment</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with model status and information."""
    st.sidebar.markdown("## 🔧 System Status")
    
    # Initialize predictor to check model status
    try:
        predictor = UnifiedDiseasePredictor()
        model_status = predictor.get_model_status()
        
        # NLP Model Status
        nlp_status = model_status['nlp_model']
        if nlp_status['available']:
            st.sidebar.success("✅ NLP Model: Ready")
        else:
            st.sidebar.warning("⚠️ NLP Model: Demo Mode")
        
        # CV Model Status
        cv_status = model_status['cv_model']
        if cv_status['available']:
            st.sidebar.success("✅ CV Model: Ready")
        else:
            st.sidebar.warning("⚠️ CV Model: Demo Mode")
        
        # Audio Status
        st.sidebar.success("✅ Audio Processing: Ready")
        
    except Exception as e:
        st.sidebar.error(f"Error checking model status: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    # Load and display analytics
    history = load_prediction_history()
    analytics = create_analytics_dashboard(history)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Predictions", analytics['total_predictions'])
    with col2:
        if analytics['confidence_stats']['mean'] > 0:
            st.metric("Avg Confidence", f"{analytics['confidence_stats']['mean']:.1%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Project Features")
    st.sidebar.markdown("""
    - **Text Analysis**: NLP models for symptom analysis
    - **Image Analysis**: CV models for medical images
    - **Audio Input**: Voice-based symptom description
    - **Multi-language**: English & Hindi support
    - **Unified Prediction**: Combined AI insights
    - **Analytics**: Usage statistics and insights
    """)

def create_language_selector():
    """Create language selector."""
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    language = multilang.get_language_selector()
    st.markdown('</div>', unsafe_allow_html=True)
    return language

def create_input_section(language: str = 'en'):
    """Create the input section for symptoms, image, and audio."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Input Your Symptoms</h2>', unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎤 Voice Input", "🖼️ Image Upload"])
    
    with tab1:
        # Text input for symptoms
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        symptoms = st.text_area(
            "Describe your symptoms in detail:",
            height=150,
            placeholder="Enter your symptoms here... (e.g., 'I have been experiencing chest pain, shortness of breath, and fatigue for the past week')",
            help="Be as detailed as possible. Include duration, severity, and any other relevant information."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Audio input section
        st.markdown('<div class="audio-section">', unsafe_allow_html=True)
        st.markdown("### 🎤 Voice Input")
        st.markdown("Click the button below to record your symptoms using your microphone.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                with st.spinner("Recording... Speak now!"):
                    # Record audio
                    text, confidence, audio_data = record_and_transcribe(duration=10, language=language)
                    
                    if text:
                        st.success(f"Transcribed: {text}")
                        st.session_state.voice_symptoms = text
                        st.session_state.voice_confidence = confidence
                    else:
                        st.error("Could not transcribe audio. Please try again.")
        
        # Display transcribed text
        if 'voice_symptoms' in st.session_state:
            st.text_area("Transcribed Symptoms:", value=st.session_state.voice_symptoms, height=100)
            symptoms = st.session_state.voice_symptoms
        else:
            symptoms = ""
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Image upload section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, DICOM"
        )
        
        image_path = None
        if uploaded_file is not None:
            # Validate image file
            is_valid, error_msg = validate_image_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Save uploaded file temporarily
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                image_path = temp_dir / uploaded_file.name
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                # Clean up old files
                for old_file in temp_dir.glob("*"):
                    if old_file != image_path:
                        old_file.unlink()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return symptoms, image_path, predict_button

def create_prediction_section(symptoms: str, image_path: Optional[str], predict_button: bool, language: str = 'en'):
    """Create the prediction results section."""
    if not predict_button:
        return
    
    # Validate input
    if not symptoms.strip():
        st.warning("⚠️ Please enter your symptoms before analyzing.")
        return
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2>🔍 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Show loading spinner
    with st.spinner("🤖 AI is analyzing your symptoms..."):
        try:
            # Make prediction with language support
            result = predict_disease(symptoms, str(image_path) if image_path else None, language=language)
            
            # Save to history
            save_prediction_history(result)
            
            # Create summary
            summary = create_prediction_summary(result)
            metrics = create_prediction_metrics(result)
            
            # Display main prediction
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-disease">{summary.get("predicted_disease", "Unknown")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-confidence">Confidence: {metrics["confidence_level"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display language detection info
            if result.get('detected_language') != language:
                st.info(f"🌐 Detected Language: {result.get('detected_language', 'Unknown')}")
            
            # Display model contributions
            if summary.get('source') == 'Combined NLP + CV':
                st.markdown("### 🤖 Model Contributions")
                col1, col2 = st.columns(2)
                
                with col1:
                    nlp_contrib = result['unified_prediction'].get('nlp_contribution', {})
                    st.info(f"**NLP Model**: {nlp_contrib.get('disease', 'Unknown')} ({nlp_contrib.get('confidence', 0.0):.1%})")
                
                with col2:
                    cv_contrib = result['unified_prediction'].get('cv_contribution', {})
                    st.info(f"**CV Model**: {cv_contrib.get('disease', 'Unknown')} ({cv_contrib.get('confidence', 0.0):.1%})")
            
            # Display top predictions
            top_predictions = summary.get('top_predictions', [])
            if top_predictions:
                st.markdown("### 📊 Top Predictions")
                for i, pred in enumerate(top_predictions[:3], 1):
                    st.write(f"{i}. **{pred['disease']}** ({pred['confidence']:.1%})")
            
            # Display related symptoms
            related_symptoms = summary.get('related_symptoms', [])
            if related_symptoms:
                st.markdown("### 🔗 Related Symptoms")
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                for symptom in related_symptoms:
                    st.markdown(f"• {symptom}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display precautions
            precautions = summary.get('precautions', [])
            if precautions:
                st.markdown("### ⚠️ Recommendations")
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                for precaution in precautions:
                    st.markdown(f"• {precaution}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display technical details
            with st.expander("🔧 Technical Details"):
                st.json({
                    "timestamp": summary.get('timestamp'),
                    "model_availability": summary.get('model_availability', {}),
                    "input_symptoms": summary.get('input_symptoms'),
                    "translated_symptoms": result.get('translated_symptoms'),
                    "detected_language": result.get('detected_language'),
                    "input_image": summary.get('input_image'),
                    "prediction_source": summary.get('source'),
                    "processing_time": result.get('processing_time'),
                    "query_id": result.get('query_id')
                })
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            logger.error(f"Prediction error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_analytics_section():
    """Create the analytics section."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2>📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Load analytics data
    history = load_prediction_history()
    analytics = create_analytics_dashboard(history)
    
    if analytics['total_predictions'] == 0:
        st.info("No predictions yet. Make some predictions to see analytics!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", analytics['total_predictions'])
    
    with col2:
        st.metric("Avg Confidence", f"{analytics['confidence_stats']['mean']:.1%}")
    
    with col3:
        st.metric("Max Confidence", f"{analytics['confidence_stats']['max']:.1%}")
    
    with col4:
        st.metric("Min Confidence", f"{analytics['confidence_stats']['min']:.1%}")
    
    # Disease distribution chart
    if analytics['disease_distribution']:
        st.markdown("### 🎯 Disease Distribution")
        df_diseases = pd.DataFrame(list(analytics['disease_distribution'].items()), 
                                 columns=['Disease', 'Count'])
        
        fig = px.pie(df_diseases, values='Count', names='Disease', 
                    title="Predicted Diseases Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage chart
    st.markdown("### 🤖 Model Usage")
    model_usage = analytics['model_usage']
    df_usage = pd.DataFrame(list(model_usage.items()), 
                           columns=['Model Type', 'Count'])
    
    fig = px.bar(df_usage, x='Model Type', y='Count', 
                title="Model Usage Statistics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.markdown("### 📋 Recent Predictions")
    if analytics['recent_predictions']:
        df_recent = pd.DataFrame(analytics['recent_predictions'])
        df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df_recent = df_recent[['timestamp', 'predicted_disease', 'confidence', 'symptoms']]
        df_recent.columns = ['Time', 'Predicted Disease', 'Confidence', 'Symptoms']
        st.dataframe(df_recent, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_about_section():
    """Create the about section."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2>About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is a **personal learning project** demonstrating advanced AI/ML techniques for disease prediction using multiple input modalities. The project showcases various machine learning concepts and technologies suitable for a portfolio.
    
    ### 🤖 Technical Features
    
    **Multi-Modal Input Processing:**
    - **Text Analysis**: NLP models (BERT, BioBERT) for symptom text analysis
    - **Image Analysis**: Computer Vision models (ResNet, EfficientNet) for medical image processing
    - **Audio Processing**: Speech-to-text conversion for voice-based symptom input
    - **Multi-language Support**: English and Hindi language processing
    
    **Advanced ML Techniques:**
    - **Transfer Learning**: Pre-trained models fine-tuned for medical domain
    - **Ensemble Methods**: Combining multiple model predictions
    - **Data Augmentation**: Image and text augmentation techniques
    - **Model Optimization**: Hyperparameter tuning and model compression
    
    **Full-Stack Development:**
    - **Backend**: Python, PyTorch, TensorFlow, HuggingFace Transformers
    - **Frontend**: Streamlit with custom CSS styling
    - **Data Processing**: Pandas, NumPy, OpenCV, scikit-learn
    - **Deployment**: Streamlit Cloud, HuggingFace Spaces, Docker
    
    ### 🛠️ Technology Stack
    
    **Machine Learning & AI:**
    - PyTorch, TensorFlow
    - HuggingFace Transformers
    - OpenCV, scikit-image
    - NLTK, spaCy
    
    **Web Development:**
    - Streamlit
    - HTML/CSS/JavaScript
    - Plotly for visualizations
    
    **Data Processing:**
    - Pandas, NumPy
    - scikit-learn
    - Albumentations
    
    **Deployment & DevOps:**
    - Docker
    - Streamlit Cloud
    - HuggingFace Spaces
    - Git/GitHub
    
    ### 📊 Learning Outcomes
    
    This project demonstrates proficiency in:
    - **Deep Learning**: CNN, RNN, Transformer architectures
    - **NLP**: Text preprocessing, tokenization, language models
    - **Computer Vision**: Image processing, medical imaging
    - **MLOps**: Model deployment, monitoring, logging
    - **Full-Stack Development**: End-to-end application development
    - **Cloud Computing**: Deployment and scaling
    
    ### 🎓 Educational Purpose
    
    This project is created for:
    - **Personal Learning**: Understanding AI/ML concepts
    - **Portfolio Enhancement**: Demonstrating technical skills
    - **Resume Building**: Showcasing relevant experience
    - **Skill Development**: Hands-on practice with modern technologies
    
    ### ⚠️ Important Note
    
    This is an **educational project** and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    # Load custom CSS
    load_custom_css()
    
    # Create header
    create_header()
    
    # Create portfolio section
    create_portfolio_section()
    
    # Create language selector
    language = create_language_selector()
    
    # Create sidebar
    create_sidebar()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Create main content
        symptoms, image_path, predict_button = create_input_section(language)
        
        # Create prediction section
        create_prediction_section(symptoms, image_path, predict_button, language)
    
    with tab2:
        # Create analytics section
        create_analytics_section()
    
    with tab3:
        # Create about section
        create_about_section()

if __name__ == "__main__":
    main()
