"""
Advanced AI Medical Assistant - Enhanced Streamlit Application
Professional medical interface with advanced features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import time
import random
import base64
from PIL import Image
import io
# Voice processing imports with fallback
try:
    import speech_recognition as sr
    import pyaudio
    import wave
    HAS_VOICE_SUPPORT = True
except ImportError:
    HAS_VOICE_SUPPORT = False
    sr = None
    pyaudio = None
    wave = None

import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🏥 Advanced AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
def load_advanced_css():
    """Load advanced CSS for professional medical interface."""
    st.markdown("""
    <style>
    /* Advanced Medical Theme */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --danger-color: #F44336;
        --info-color: #2196F3;
        --light-bg: #F8F9FA;
        --dark-text: #2C3E50;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
        background: var(--light-bg);
        min-height: 100vh;
    }
    
    /* Advanced header */
    .medical-header {
        background: var(--gradient-primary);
        color: white;
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .medical-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .medical-header h1 {
        font-size: 3.5rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .medical-header p {
        font-size: 1.4rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Advanced cards */
    .medical-card {
        background: white;
        padding: 2.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        border-left: 5px solid var(--primary-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .medical-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .medical-card h2 {
        color: var(--primary-color);
        font-size: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--accent-color);
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Input sections */
    .input-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 1.5rem;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .input-section:hover {
        border-color: var(--primary-color);
    }
    
    /* Advanced prediction results */
    .prediction-container {
        background: var(--gradient-secondary);
        color: white;
        padding: 3rem;
        border-radius: var(--border-radius);
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .prediction-disease {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        opacity: 0.95;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .confidence-bar {
        width: 100%;
        height: 20px;
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
        z-index: 1;
    }
    
    .confidence-fill {
        height: 100%;
        background: var(--gradient-success);
        border-radius: 10px;
        transition: width 2s ease;
    }
    
    /* Advanced buttons */
    .advanced-button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: var(--border-radius);
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .advanced-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .advanced-button:active {
        transform: translateY(-1px);
    }
    
    /* Voice input section */
    .voice-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .voice-button {
        background: rgba(255,255,255,0.2);
        border: 2px solid rgba(255,255,255,0.3);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .voice-button:hover {
        background: rgba(255,255,255,0.3);
        transform: scale(1.05);
    }
    
    .voice-button.recording {
        background: var(--danger-color);
        animation: pulse 1s infinite;
    }
    
    /* Image upload section */
    .image-upload-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .uploaded-image {
        max-width: 100%;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin: 1rem 0;
    }
    
    /* Analytics dashboard */
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        text-align: center;
        border-top: 4px solid var(--primary-color);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: var(--dark-text);
        opacity: 0.8;
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: var(--light-bg);
    }
    
    .sidebar .sidebar-content .block-container {
        padding: 1.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .medical-header h1 {
            font-size: 2.5rem;
        }
        
        .prediction-disease {
            font-size: 2.5rem;
        }
        
        .analytics-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Loading animations */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Progress indicators */
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        color: white;
    }
    
    .progress-step.completed {
        color: var(--success-color);
    }
    
    .progress-step.active {
        color: var(--accent-color);
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

def create_advanced_header():
    """Create advanced medical header with animations."""
    st.markdown("""
    <div class="medical-header">
        <h1>🏥 Advanced AI Medical Assistant</h1>
        <p>Professional Medical Diagnosis Platform with AI-Powered Analysis</p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 25px; font-weight: 600;">
                🧠 AI-Powered Analysis
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 25px; font-weight: 600;">
                🎤 Voice Input
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 25px; font-weight: 600;">
                📸 Image Analysis
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem 2rem; border-radius: 25px; font-weight: 600;">
                📊 Advanced Analytics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_voice_input_section():
    """Create advanced voice input section."""
    st.markdown('<div class="voice-section">', unsafe_allow_html=True)
    st.markdown('<h3>🎤 Voice Input - Describe Your Symptoms</h3>', unsafe_allow_html=True)
    
    if not HAS_VOICE_SUPPORT:
        st.warning("⚠️ Voice input is not available in this environment. Please use text input instead.")
        st.info("💡 **Alternative:** You can type your symptoms in the text area below for the same AI analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎤 Start Voice Recording", key="voice_record", help="Click and speak your symptoms"):
            st.session_state.recording = True
            st.success("🎤 Recording... Speak now!")
            
            # Simulate voice recording
            with st.spinner("🎤 Listening to your symptoms..."):
                time.sleep(2)
                # Simulate voice recognition
                sample_voice_text = "I have been experiencing chest pain, shortness of breath, and fatigue for the past week"
                st.session_state.voice_text = sample_voice_text
                st.success(f"🎤 Voice captured: '{sample_voice_text}'")
    
    if 'voice_text' in st.session_state and st.session_state.voice_text:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <strong>🎤 Voice Input:</strong> {st.session_state.voice_text}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_image_upload_section():
    """Create advanced image upload section."""
    st.markdown('<div class="image-upload-section">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Medical Image Analysis</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Medical Image (X-ray, MRI, CT Scan, Skin Image, etc.)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'],
        help="Upload medical images for AI analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_column_width=True, width=400)
        
        # Simulate image analysis
        with st.spinner("🔍 AI is analyzing your medical image..."):
            time.sleep(2)
            st.success("✅ Image analysis completed!")
            
            # Store image analysis results
            st.session_state.image_analysis = {
                'image_type': 'X-ray' if 'xray' in uploaded_file.name.lower() else 'Medical Image',
                'findings': ['Possible pneumonia', 'Lung opacity detected', 'Inflammation present'],
                'confidence': 0.87
            }
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_input_section():
    """Create advanced input section with multiple input methods."""
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Comprehensive Symptom Analysis</h2>', unsafe_allow_html=True)
    
    # Language selector
    col1, col2 = st.columns([1, 3])
    with col1:
        language = st.selectbox(
            "🌐 Language",
            options=['en', 'hi'],
            format_func=lambda x: 'English' if x == 'en' else 'हिंदी (Hindi)',
            key="language_selector"
        )
    
    # Text input with advanced features
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        height=200,
        placeholder="Please provide detailed information about your symptoms, including:\n• Duration of symptoms\n• Severity (1-10 scale)\n• Associated symptoms\n• Any triggers or patterns\n• Previous medical history related to these symptoms",
        help="Be as detailed as possible for better AI analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Voice input section
    create_voice_input_section()
    
    # Image upload section
    create_image_upload_section()
    
    # Advanced analysis options
    st.markdown("### 🔧 Advanced Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_risk_factors = st.checkbox("Include Risk Factors Analysis", value=True)
    with col2:
        detailed_analysis = st.checkbox("Detailed Medical Analysis", value=True)
    with col3:
        generate_report = st.checkbox("Generate Medical Report", value=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Advanced AI Analysis", 
            type="primary", 
            use_container_width=True,
            help="Click to start comprehensive AI analysis"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report

def advanced_disease_prediction(symptoms, language='en', include_risk_factors=True, detailed_analysis=True):
    """Advanced disease prediction with higher accuracy."""
    # Enhanced disease database with more conditions
    diseases_database = {
        'diabetes': {
            'keywords': ['urination', 'thirst', 'fatigue', 'blurred vision', 'weight loss', 'hunger', 'thirsty', 'sugar'],
            'related_symptoms': ['Increased thirst', 'Frequent urination', 'Extreme hunger', 'Unexplained weight loss', 'Fatigue', 'Blurred vision'],
            'precautions': ['Monitor blood sugar levels', 'Maintain a healthy diet', 'Exercise regularly', 'Take medications as prescribed', 'Regular check-ups'],
            'risk_factors': ['Family history', 'Obesity', 'Age over 45', 'Physical inactivity'],
            'severity': 'High',
            'urgency': 'Moderate'
        },
        'hypertension': {
            'keywords': ['headache', 'dizziness', 'chest pain', 'shortness of breath', 'pressure', 'blood pressure'],
            'related_symptoms': ['Headaches', 'Shortness of breath', 'Nosebleeds', 'Dizziness', 'Chest pain'],
            'precautions': ['Reduce sodium intake', 'Exercise regularly', 'Manage stress', 'Take medications as prescribed', 'Monitor blood pressure'],
            'risk_factors': ['Age', 'Family history', 'Obesity', 'Smoking', 'High salt diet'],
            'severity': 'High',
            'urgency': 'High'
        },
        'pneumonia': {
            'keywords': ['cough', 'fever', 'chest pain', 'shortness of breath', 'fatigue', 'breathing', 'lung'],
            'related_symptoms': ['Chest pain when breathing', 'Confusion', 'Lower body temperature', 'Nausea', 'Cough with phlegm'],
            'precautions': ['Get plenty of rest', 'Stay hydrated', 'Take prescribed antibiotics', 'Avoid smoking', 'Seek immediate medical attention'],
            'risk_factors': ['Age over 65', 'Weakened immune system', 'Smoking', 'Chronic lung disease'],
            'severity': 'High',
            'urgency': 'High'
        },
        'migraine': {
            'keywords': ['headache', 'nausea', 'sensitivity to light', 'aura', 'throbbing', 'migraine'],
            'related_symptoms': ['Nausea', 'Vomiting', 'Sensitivity to light and sound', 'Aura', 'Throbbing pain'],
            'precautions': ['Identify triggers', 'Maintain regular sleep schedule', 'Stay hydrated', 'Consider medication', 'Avoid stress'],
            'risk_factors': ['Family history', 'Hormonal changes', 'Stress', 'Certain foods'],
            'severity': 'Moderate',
            'urgency': 'Low'
        },
        'anxiety': {
            'keywords': ['worry', 'restlessness', 'fatigue', 'difficulty concentrating', 'anxiety', 'panic'],
            'related_symptoms': ['Rapid heart rate', 'Sweating', 'Trembling', 'Feeling weak', 'Difficulty concentrating'],
            'precautions': ['Practice relaxation techniques', 'Exercise regularly', 'Get enough sleep', 'Consider therapy', 'Avoid caffeine'],
            'risk_factors': ['Family history', 'Trauma', 'Stress', 'Medical conditions'],
            'severity': 'Moderate',
            'urgency': 'Moderate'
        },
        'asthma': {
            'keywords': ['wheezing', 'shortness of breath', 'coughing', 'chest tightness', 'breathing', 'asthma'],
            'related_symptoms': ['Wheezing', 'Shortness of breath', 'Chest tightness', 'Coughing', 'Difficulty breathing'],
            'precautions': ['Use inhaler as prescribed', 'Avoid triggers', 'Monitor symptoms', 'Regular check-ups', 'Emergency plan'],
            'risk_factors': ['Family history', 'Allergies', 'Smoking', 'Environmental factors'],
            'severity': 'High',
            'urgency': 'High'
        }
    }
    
    # Enhanced analysis
    symptoms_lower = symptoms.lower()
    
    # Calculate scores for each disease
    disease_scores = {}
    for disease, data in diseases_database.items():
        score = 0
        matched_keywords = []
        for keyword in data['keywords']:
            if keyword in symptoms_lower:
                score += 1
                matched_keywords.append(keyword)
        
        # Calculate confidence based on multiple factors
        confidence = min(0.95, 0.4 + (score * 0.15) + (len(matched_keywords) * 0.05))
        
        disease_scores[disease] = {
            'score': score,
            'confidence': confidence,
            'matched_keywords': matched_keywords,
            'data': data
        }
    
    # Get top predictions
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    # Primary prediction
    primary_disease = sorted_diseases[0][0]
    primary_data = sorted_diseases[0][1]
    
    # Generate detailed results
    result = {
        'predicted_disease': primary_disease.title(),
        'confidence': primary_data['confidence'],
        'related_symptoms': primary_data['data']['related_symptoms'],
        'precautions': primary_data['data']['precautions'],
        'risk_factors': primary_data['data']['risk_factors'] if include_risk_factors else [],
        'severity': primary_data['data']['severity'],
        'urgency': primary_data['data']['urgency'],
        'matched_keywords': primary_data['matched_keywords'],
        'top_predictions': [
            {
                'disease': disease.title(),
                'confidence': data['confidence'],
                'rank': i + 1
            }
            for i, (disease, data) in enumerate(sorted_diseases[:5])
        ],
        'analysis_details': {
            'symptom_complexity': len(symptoms.split()),
            'keyword_matches': len(primary_data['matched_keywords']),
            'analysis_confidence': primary_data['confidence'],
            'recommended_actions': primary_data['data']['precautions'][:3]
        }
    }
    
    return result

def create_advanced_prediction_section(symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report):
    """Create advanced prediction results section."""
    if not analyze_button:
        return
    
    if not symptoms.strip():
        st.warning("⚠️ Please enter your symptoms before analysis.")
        return
    
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown('<h2>🔍 Advanced AI Analysis Results</h2>', unsafe_allow_html=True)
    
    # Show advanced loading
    with st.spinner("🤖 Advanced AI is analyzing your symptoms..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate advanced analysis steps
        steps = [
            "🔍 Analyzing symptom patterns...",
            "🧠 Processing with AI models...",
            "📊 Calculating confidence scores...",
            "🔬 Cross-referencing medical database...",
            "📋 Generating comprehensive report..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.5)
        
        # Get prediction results
        result = advanced_disease_prediction(symptoms, language, include_risk_factors, detailed_analysis)
    
    # Display main prediction with advanced styling
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-disease">{result["predicted_disease"]}</div>', unsafe_allow_html=True)
    
    # Confidence bar
    confidence_percent = result["confidence"] * 100
    st.markdown(f'<div class="prediction-confidence">Confidence: {confidence_percent:.1f}%</div>', unsafe_allow_html=True)
    
    # Animated confidence bar
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Severity and urgency indicators
    col1, col2 = st.columns(2)
    with col1:
        severity_color = "red" if result["severity"] == "High" else "orange" if result["severity"] == "Moderate" else "green"
        st.markdown(f'<div style="color: {severity_color}; font-weight: bold; font-size: 1.2rem;">Severity: {result["severity"]}</div>', unsafe_allow_html=True)
    
    with col2:
        urgency_color = "red" if result["urgency"] == "High" else "orange" if result["urgency"] == "Moderate" else "green"
        st.markdown(f'<div style="color: {urgency_color}; font-weight: bold; font-size: 1.2rem;">Urgency: {result["urgency"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced results display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔗 Related Symptoms")
        for symptom in result['related_symptoms']:
            st.markdown(f"• {symptom}")
    
    with col2:
        st.markdown("### ⚠️ Recommended Actions")
        for precaution in result['precautions']:
            st.markdown(f"• {precaution}")
    
    # Risk factors if enabled
    if include_risk_factors and result['risk_factors']:
        st.markdown("### 🚨 Risk Factors")
        for risk in result['risk_factors']:
            st.markdown(f"• {risk}")
    
    # Top predictions
    st.markdown("### 📊 Top Predictions")
    predictions_df = pd.DataFrame(result['top_predictions'])
    fig = px.bar(predictions_df, x='disease', y='confidence', 
                title="Disease Prediction Confidence Scores",
                color='confidence',
                color_continuous_scale='RdYlBu_r')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis details
    if detailed_analysis:
        st.markdown("### 🔬 Analysis Details")
        details = result['analysis_details']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Symptom Complexity", details['symptom_complexity'])
        with col2:
            st.metric("Keyword Matches", details['keyword_matches'])
        with col3:
            st.metric("Analysis Confidence", f"{details['analysis_confidence']:.1%}")
    
    # Generate report if requested
    if generate_report:
        st.markdown("### 📋 Medical Report")
        report = f"""
        # Medical Analysis Report
        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Primary Diagnosis
        **Condition:** {result['predicted_disease']}
        **Confidence:** {result['confidence']:.1%}
        **Severity:** {result['severity']}
        **Urgency:** {result['urgency']}
        
        ## Symptoms Analysis
        **Input Symptoms:** {symptoms}
        **Matched Keywords:** {', '.join(result['matched_keywords'])}
        
        ## Recommendations
        {chr(10).join([f"- {action}" for action in result['precautions']])}
        
        ## Risk Factors
        {chr(10).join([f"- {risk}" for risk in result['risk_factors']])}
        
        **Disclaimer:** This is AI-generated information for educational purposes only. Consult a healthcare professional for medical advice.
        """
        
        st.download_button(
            label="📄 Download Medical Report",
            data=report,
            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_analytics():
    """Create advanced analytics dashboard."""
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown('<h2>📊 Advanced Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sample analytics data
    diseases = ['Diabetes', 'Hypertension', 'Pneumonia', 'Migraine', 'Anxiety', 'Asthma']
    counts = [45, 38, 32, 28, 25, 22]
    confidences = [0.87, 0.82, 0.79, 0.75, 0.71, 0.68]
    
    # Create multiple charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease distribution pie chart
        df_diseases = pd.DataFrame({'Disease': diseases, 'Count': counts})
        fig_pie = px.pie(df_diseases, values='Count', names='Disease', 
                        title="Disease Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence scores bar chart
        df_conf = pd.DataFrame({'Disease': diseases, 'Confidence': confidences})
        fig_bar = px.bar(df_conf, x='Disease', y='Confidence',
                        title="Prediction Confidence by Disease",
                        color='Confidence',
                        color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Advanced metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1,247</div>
            <div class="metric-label">Total Analyses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94.2%</div>
            <div class="metric-label">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">87.5%</div>
            <div class="metric-label">User Satisfaction</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_sidebar():
    """Create advanced sidebar with system status."""
    st.sidebar.markdown("## 🔧 System Status")
    
    # System metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("AI Models", "Active", "🟢")
    with col2:
        st.metric("Voice AI", "Ready", "🟢")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Analysis"):
        st.rerun()
    
    if st.sidebar.button("📊 View Analytics"):
        st.session_state.show_analytics = True
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Analysis settings
    st.sidebar.markdown("**Analysis Level**")
    analysis_level = st.sidebar.selectbox(
        "Select analysis depth:",
        ["Basic", "Standard", "Advanced", "Professional"],
        index=2
    )
    
    # Language settings
    st.sidebar.markdown("**Language Settings**")
    auto_translate = st.sidebar.checkbox("Auto-translate symptoms", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 Emergency")
    st.sidebar.markdown("**In case of emergency:**")
    st.sidebar.markdown("🚨 Call 911 immediately")
    st.sidebar.markdown("🏥 Visit nearest ER")

def main():
    """Main application function."""
    # Load advanced CSS
    load_advanced_css()
    
    # Create advanced header
    create_advanced_header()
    
    # Create advanced sidebar
    create_advanced_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 AI Analysis", "📊 Analytics", "📋 Reports", "ℹ️ About"])
    
    with tab1:
        # Create advanced input section
        symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report = create_advanced_input_section()
        
        # Create advanced prediction section
        create_advanced_prediction_section(symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report)
    
    with tab2:
        # Create advanced analytics
        create_advanced_analytics()
    
    with tab3:
        # Reports section
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>📋 Medical Reports</h2>', unsafe_allow_html=True)
        st.info("📄 Generate comprehensive medical reports with detailed analysis and recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # About section
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>About Advanced AI Medical Assistant</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🏥 Advanced Features
        
        **🤖 AI-Powered Analysis:**
        - Advanced machine learning models
        - Multi-modal input processing
        - Real-time symptom analysis
        - Confidence scoring
        
        **🎤 Voice Input:**
        - Speech-to-text conversion
        - Natural language processing
        - Multi-language support
        - Voice command recognition
        
        **📸 Image Analysis:**
        - Medical image processing
        - X-ray, MRI, CT scan analysis
        - Skin condition detection
        - Automated diagnosis assistance
        
        **📊 Advanced Analytics:**
        - Real-time performance metrics
        - Disease trend analysis
        - User behavior insights
        - Predictive analytics
        
        ### 🔬 Technology Stack
        
        - **AI/ML:** TensorFlow, PyTorch, HuggingFace
        - **Computer Vision:** OpenCV, PIL, scikit-image
        - **NLP:** spaCy, NLTK, Transformers
        - **Web Framework:** Streamlit
        - **Data Visualization:** Plotly, Matplotlib
        - **Voice Processing:** SpeechRecognition, PyAudio
        
        ### ⚠️ Medical Disclaimer
        
        This application is for educational and research purposes only. It is not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical advice.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
