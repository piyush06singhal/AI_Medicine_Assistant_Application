"""
Multi-language AI Medical Assistant - Streamlit Web Application
Supports English and Hindi with comprehensive logging and analytics.
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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.unified_predictor import UnifiedDiseasePredictor, predict_disease
from utils.multilang_support import MultiLanguageSupport, get_ui_text
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

# Initialize multi-language support
multilang = MultiLanguageSupport()

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
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
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .prediction-disease {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_header(language: str = 'en'):
    """Create the main header section."""
    title = get_ui_text('symptoms', language)  # Using symptoms as title placeholder
    subtitle = get_ui_text('unified_prediction', language)
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🏥 AI Medical Assistant</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar(language: str = 'en'):
    """Create the sidebar with model status and information."""
    st.sidebar.markdown(f"## {get_ui_text('system_status', language)}")
    
    # Initialize predictor to check model status
    try:
        predictor = UnifiedDiseasePredictor()
        model_status = predictor.get_model_status()
        
        # NLP Model Status
        nlp_status = model_status['nlp_model']
        if nlp_status['available']:
            st.sidebar.success(f"✅ NLP {get_ui_text('model_ready', language)}")
        else:
            st.sidebar.error(f"❌ NLP {get_ui_text('model_not_available', language)}")
        
        # CV Model Status
        cv_status = model_status['cv_model']
        if cv_status['available']:
            st.sidebar.success(f"✅ CV {get_ui_text('model_ready', language)}")
        else:
            st.sidebar.error(f"❌ CV {get_ui_text('model_not_available', language)}")
        
        # Model Weights
        st.sidebar.markdown(f"### ⚖️ {get_ui_text('model_contributions', language)}")
        nlp_weight = st.sidebar.slider("NLP Weight", 0.0, 1.0, 0.6, 0.1)
        cv_weight = st.sidebar.slider("CV Weight", 0.0, 1.0, 0.4, 0.1)
        
        if st.sidebar.button("Update Weights"):
            predictor.update_weights(nlp_weight, cv_weight)
            st.sidebar.success("Weights updated!")
        
    except Exception as e:
        st.sidebar.error(f"Error checking model status: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_ui_text('quick_stats', language)}")
    
    # Load and display analytics
    history = load_prediction_history()
    analytics = create_analytics_dashboard(history)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(get_ui_text('predictions_today', language), analytics['total_predictions'])
    with col2:
        if analytics['confidence_stats']['mean'] > 0:
            st.metric(get_ui_text('model_accuracy', language), f"{analytics['confidence_stats']['mean']:.1%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_ui_text('about', language)}")
    st.sidebar.markdown(f"""
    {get_ui_text('nlp_models', language)}
    {get_ui_text('cv_models', language)}
    {get_ui_text('unified_prediction', language)}
    """)

def create_language_selector():
    """Create language selector."""
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    language = multilang.get_language_selector()
    st.markdown('</div>', unsafe_allow_html=True)
    return language

def create_input_section(language: str = 'en'):
    """Create the input section for symptoms and image upload."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<h2>{get_ui_text("symptoms", language)}</h2>', unsafe_allow_html=True)
    
    # Text input for symptoms
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    symptoms = st.text_area(
        f"{get_ui_text('symptoms', language)}:",
        height=150,
        placeholder=get_ui_text('enter_symptoms', language),
        help=get_ui_text('be_detailed', language)
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image upload section
    st.markdown(f'<h3>{get_ui_text("upload_image", language)}</h3>', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        get_ui_text('upload_image', language),
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'],
        help=get_ui_text('supported_formats', language)
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
        predict_button = st.button(f"🔍 {get_ui_text('analyze', language)}", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return symptoms, image_path, predict_button

def create_prediction_section(symptoms: str, image_path: Optional[str], predict_button: bool, language: str = 'en'):
    """Create the prediction results section."""
    if not predict_button:
        return
    
    # Validate input
    is_valid, error_msg = validate_symptoms_input(symptoms)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<h2>{get_ui_text("analysis_results", language)}</h2>', unsafe_allow_html=True)
    
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
            st.markdown(f'<div class="prediction-confidence">{get_ui_text("confidence", language)}: {metrics["confidence_level"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display language detection info
            if result.get('detected_language') != language:
                st.info(f"🌐 {get_ui_text('detected_language', language)}: {result.get('detected_language', 'Unknown')}")
            
            # Display model contributions
            if summary.get('source') == 'Combined NLP + CV':
                st.markdown(f"### 🤖 {get_ui_text('model_contributions', language)}")
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
                st.markdown(f"### 📊 {get_ui_text('top_predictions', language)}")
                for i, pred in enumerate(top_predictions[:3], 1):
                    st.write(f"{i}. **{pred['disease']}** ({pred['confidence']:.1%})")
            
            # Display related symptoms
            related_symptoms = summary.get('related_symptoms', [])
            if related_symptoms:
                st.markdown(f"### 🔗 {get_ui_text('related_symptoms', language)}")
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                for symptom in related_symptoms:
                    st.markdown(f"• {symptom}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display precautions
            precautions = summary.get('precautions', [])
            if precautions:
                st.markdown(f"### ⚠️ {get_ui_text('precautions', language)}")
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                for precaution in precautions:
                    st.markdown(f"• {precaution}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display query ID for logging
            if result.get('query_id'):
                st.info(f"📝 Query ID: {result['query_id']} (Logged for model improvement)")
            
            # Download results
            st.markdown(f"### 📥 {get_ui_text('export_results', language)}")
            export_data = create_prediction_export(result)
            st.download_button(
                label=f"📄 {get_ui_text('download_report', language)}",
                data=export_data,
                file_name=f"medical_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # Display technical details
            with st.expander(f"🔧 {get_ui_text('technical_details', language)}"):
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

def create_warning_section(language: str = 'en'):
    """Create the medical disclaimer section."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<h2>⚠️ {get_ui_text("important_disclaimer", language)}</h2>', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="warning-box">
        <h3>🚨 {get_ui_text("ai_information", language)}</h3>
        <p><strong>{get_ui_text("important_note", language)}</strong></p>
        <p><strong>{get_ui_text("not_for_clinical", language)}</strong></p>
        
        <p><strong>{get_ui_text("consult_healthcare", language)}</strong></p>
        <ul>
            <li>{get_ui_text("medical_diagnosis", language)}</li>
            <li>{get_ui_text("emergency_situations", language)}</li>
            <li>{get_ui_text("serious_symptoms", language)}</li>
            <li>{get_ui_text("health_concerns", language)}</li>
        </ul>
        
        <p><strong>{get_ui_text("emergency_call", language)}</strong></p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_analytics_section(language: str = 'en'):
    """Create the analytics section."""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<h2>📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
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

def create_footer(language: str = 'en'):
    """Create the footer section."""
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏥 AI Medical Assistant | {get_ui_text('powered_by', language)}</p>
        <p>{get_ui_text('educational_purposes', language)} | {get_ui_text('not_clinical_use', language)}</p>
        <p>{get_ui_text('copyright', language)}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Load custom CSS
    load_custom_css()
    
    # Create language selector
    language = create_language_selector()
    
    # Create header
    create_header(language)
    
    # Create sidebar
    create_sidebar(language)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        # Create main content
        symptoms, image_path, predict_button = create_input_section(language)
        
        # Create prediction section
        create_prediction_section(symptoms, image_path, predict_button, language)
        
        # Create warning section
        create_warning_section(language)
    
    with tab2:
        # Create analytics section
        create_analytics_section(language)
    
    with tab3:
        # Create about section
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown(f'<h2>About AI Medical Assistant</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        ### 🏥 What is AI Medical Assistant?
        
        The AI Medical Assistant is an advanced healthcare technology platform that combines 
        Natural Language Processing (NLP) and Computer Vision (CV) to provide intelligent 
        disease prediction and analysis.
        
        ### 🌐 Multi-language Support
        
        The system supports multiple languages including:
        - **English**: Full support for symptom input and interface
        - **Hindi**: Complete Hindi interface and symptom input support
        - **Auto-translation**: Automatic detection and translation of symptoms
        
        ### 🤖 How It Works
        
        1. **Text Analysis**: Our NLP models analyze symptom descriptions using state-of-the-art 
           language models trained on medical literature.
        
        2. **Image Analysis**: Our CV models examine medical images (X-rays, skin lesions, etc.) 
           using deep learning architectures like ResNet, EfficientNet, and Vision Transformers.
        
        3. **Unified Prediction**: We combine insights from both models to provide comprehensive 
           disease predictions with confidence scores.
        
        4. **Query Logging**: All user queries are logged for model improvement and analytics.
        
        ### 🔬 Technology Stack
        
        - **NLP Models**: BERT, BioBERT, ClinicalBERT
        - **CV Models**: ResNet, EfficientNet, DenseNet, Vision Transformers
        - **Frameworks**: PyTorch, TensorFlow, HuggingFace Transformers
        - **Web Interface**: Streamlit
        - **Data Processing**: Pandas, NumPy, OpenCV
        - **Translation**: Google Translate API
        - **Logging**: Comprehensive query logging system
        
        ### ⚠️ Important Notes
        
        - This system is for **educational and research purposes only**
        - **Not intended for clinical use or medical diagnosis**
        - Always consult qualified healthcare professionals
        - Use only for learning and research
        - All queries are logged for model improvement
        
        ### 📞 Support
        
        For technical support or questions, please contact our team.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create footer
    create_footer(language)

if __name__ == "__main__":
    main()
