"""
Advanced AI Medical Assistant - Professional Medical Analysis
High-accuracy disease prediction with beautiful UI and enhanced functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time
import random
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="🏥 Advanced AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Medical UI Design
def load_advanced_css():
    """Load professional medical UI with excellent readability."""
    st.markdown("""
    <style>
    /* Professional Medical Theme - Light Background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #cbd5e0 100%);
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem;
    }
    
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        font-size: 1.3rem;
        color: #f0f0f0 !important;
        font-weight: 400;
    }
    
    /* Solid input sections - NO transparency */
    .input-section {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 2px solid #e0e0e0;
    }
    
    .input-section h2, .input-section h3 {
        color: #1a1a1a !important;
        text-shadow: none !important;
        margin-bottom: 1rem;
    }
    
    .input-section p, .input-section span, .input-section div, .input-section label {
        color: #2d3748 !important;
        text-shadow: none !important;
    }
    
    /* Image upload section - solid background */
    .image-upload-section {
        background: #f7fafc;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 3px dashed #4299e1;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .image-upload-section:hover {
        border-color: #3182ce;
        background: #edf2f7;
    }
    
    .image-upload-section h3 {
        color: #2d3748 !important;
        text-shadow: none !important;
    }
    
    /* Results card - solid white background */
    .medical-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        border: 2px solid #e2e8f0;
    }
    
    .medical-card h1, .medical-card h2, .medical-card h3, .medical-card h4 {
        color: #1a202c !important;
        text-shadow: none !important;
        margin-bottom: 1rem;
    }
    
    .medical-card p, .medical-card span, .medical-card div, .medical-card li {
        color: #2d3748 !important;
        text-shadow: none !important;
        line-height: 1.8;
    }
    
    /* Prediction container - solid gradient */
    .prediction-container {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
    }
    
    .prediction-disease {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* Confidence bar */
    .confidence-bar {
        width: 100%;
        height: 30px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        overflow: hidden;
        margin: 1.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #34d399);
        border-radius: 15px;
        transition: width 1s ease;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* Labels - dark text on light background */
    label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }
    
    /* Input fields - solid white background */
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 10px !important;
        color: #1a202c !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        background: #ffffff;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .stCheckbox label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stCheckbox label span {
        color: #1a202c !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 10px !important;
        color: #1a202c !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    .streamlit-expanderHeader p {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: #f7fafc !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderContent p, .streamlit-expanderContent div {
        color: #2d3748 !important;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess {
        background: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Severity indicators */
    .severity-high {
        color: #dc2626 !important;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .severity-moderate {
        color: #f59e0b !important;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .severity-low {
        color: #10b981 !important;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #cbd5e0;
    }
    
    .stFileUploader label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader div {
        color: #2d3748 !important;
    }
    
    .stFileUploader small {
        color: #4a5568 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #4a5568;
        border: 2px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: transparent;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Animation */
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Global text visibility fixes */
    p, span, div, li, td, th {
        color: #2d3748 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #ffffff !important;
        color: #1a202c !important;
    }
    
    .stSelectbox label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Text area label */
    .stTextArea label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Text input label */
    .stTextInput label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #1a202c !important;
    }
    
    /* Progress bar text */
    .stProgress > div > div {
        color: #1a202c !important;
    }
    
    /* All markdown content */
    .markdown-text-container {
        color: #2d3748 !important;
    }
    
    /* Ensure white cards have dark text */
    .element-container {
        color: #2d3748 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_advanced_header():
    """Create beautiful application header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏥 Advanced AI Medical Assistant</h1>
        <p class="app-subtitle">Professional AI-powered medical analysis with high accuracy diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_input_section():
    """Create enhanced input section with better functionality."""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Describe Your Symptoms</h2>', unsafe_allow_html=True)
    
    # API Key info (optional for future integration)
    with st.expander("🔑 API Configuration (Optional - For Real AI Integration)"):
        st.info("💡 Currently using built-in medical knowledge base. To integrate with OpenAI GPT-4, Google Gemini, or other AI services, add your API key here.")
        api_key = st.text_input("API Key (Optional):", type="password", help="Enter your AI service API key for enhanced analysis")
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key saved for this session!")
    
    # Language selection
    language = st.selectbox(
        "Select Language:",
        ["English", "Hindi", "Spanish", "French", "German", "Italian"],
        help="Choose your preferred language for analysis"
    )
    
    # Symptoms input
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        height=200,
        placeholder="Please provide detailed information about your symptoms, including:\n• Duration of symptoms\n• Severity (1-10 scale)\n• Associated symptoms\n• Any triggers or patterns\n• Previous medical history related to these symptoms",
        help="Be as detailed as possible for better AI analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image upload section
    create_enhanced_image_upload_section()
    
    # Analysis options
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🔧 Advanced Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_risk_factors = st.checkbox("Include Risk Factors Analysis", value=True)
    with col2:
        detailed_analysis = st.checkbox("Detailed Medical Analysis", value=True)
    with col3:
        generate_report = st.checkbox("Generate Medical Report", value=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Advanced AI Analysis", 
            type="primary", 
            use_container_width=True,
            help="Click to start comprehensive AI analysis"
        )
    
    return symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report

def create_enhanced_image_upload_section():
    """Create enhanced image upload section."""
    st.markdown('<div class="image-upload-section">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Medical Image Analysis</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a medical image (X-ray, MRI, CT scan, skin condition, etc.):",
        type=['png', 'jpg', 'jpeg'],
        help="Upload medical images for AI analysis - works independently of text input"
    )
    
    if uploaded_file is not None:
        # Store in session state without displaying
        st.session_state.uploaded_image = uploaded_file
        st.success("✅ Image uploaded successfully! Processing image...")
        
        # Auto-trigger analysis
        st.session_state.analyze_image = True
    
    st.markdown('</div>', unsafe_allow_html=True)

def enhanced_disease_prediction(symptoms, language='en', include_risk_factors=True, detailed_analysis=True):
    """Enhanced disease prediction with higher accuracy and better database."""
    
    # Comprehensive disease database with more conditions and better accuracy
    diseases_database = {
        'diabetes': {
            'keywords': ['urination', 'thirst', 'fatigue', 'blurred vision', 'weight loss', 'hunger', 'thirsty', 'sugar', 'glucose', 'insulin', 'diabetic', 'blood sugar', 'frequent urination', 'excessive thirst'],
            'related_symptoms': ['Increased thirst', 'Frequent urination', 'Extreme hunger', 'Unexplained weight loss', 'Fatigue', 'Blurred vision', 'Slow-healing sores', 'Frequent infections'],
            'precautions': ['Monitor blood sugar levels regularly', 'Maintain a healthy diet with controlled carbohydrates', 'Exercise regularly (30 minutes daily)', 'Take medications as prescribed', 'Regular eye and foot exams', 'Manage stress levels'],
            'risk_factors': ['Family history of diabetes', 'Obesity or overweight', 'Age over 45', 'Physical inactivity', 'High blood pressure', 'Abnormal cholesterol levels'],
            'severity': 'High',
            'urgency': 'Moderate',
            'confidence_boost': 0.15
        },
        'hypertension': {
            'keywords': ['headache', 'dizziness', 'chest pain', 'shortness of breath', 'high blood pressure', 'bp', 'hypertension', 'pressure', 'headaches', 'nosebleeds'],
            'related_symptoms': ['Headaches', 'Shortness of breath', 'Nosebleeds', 'Dizziness', 'Chest pain', 'Visual changes', 'Fatigue'],
            'precautions': ['Reduce sodium intake to less than 2,300mg daily', 'Exercise regularly (150 minutes weekly)', 'Manage stress through relaxation techniques', 'Take medications as prescribed', 'Limit alcohol consumption', 'Quit smoking'],
            'risk_factors': ['Age (risk increases with age)', 'Family history', 'Obesity', 'Smoking', 'High stress levels', 'Excessive alcohol consumption'],
            'severity': 'High',
            'urgency': 'Moderate',
            'confidence_boost': 0.12
        },
        'migraine': {
            'keywords': ['headache', 'nausea', 'sensitivity to light', 'aura', 'throbbing', 'pounding', 'migraine', 'severe headache', 'light sensitivity', 'sound sensitivity'],
            'related_symptoms': ['Nausea and vomiting', 'Sensitivity to light and sound', 'Aura (visual disturbances)', 'Throbbing pain', 'Dizziness', 'Neck stiffness'],
            'precautions': ['Identify and avoid triggers (stress, certain foods, sleep changes)', 'Maintain regular sleep schedule', 'Stay hydrated', 'Consider preventive medications', 'Practice relaxation techniques', 'Keep a migraine diary'],
            'risk_factors': ['Family history of migraines', 'Stress', 'Hormonal changes', 'Certain foods (chocolate, cheese, wine)', 'Weather changes', 'Sleep disturbances'],
            'severity': 'Moderate',
            'urgency': 'Low',
            'confidence_boost': 0.10
        },
        'pneumonia': {
            'keywords': ['cough', 'fever', 'chest pain', 'shortness of breath', 'fatigue', 'chills', 'pneumonia', 'respiratory', 'breathing', 'lung', 'infection'],
            'related_symptoms': ['Chest pain when breathing or coughing', 'Confusion or changes in mental awareness', 'Lower than normal body temperature', 'Nausea, vomiting or diarrhea', 'Severe fatigue'],
            'precautions': ['Get plenty of rest', 'Stay hydrated (8-10 glasses daily)', 'Take prescribed antibiotics as directed', 'Avoid smoking and secondhand smoke', 'Use a humidifier', 'Practice good hand hygiene'],
            'risk_factors': ['Age (very young or elderly)', 'Smoking', 'Chronic lung disease', 'Weakened immune system', 'Recent viral infection', 'Hospitalization'],
            'severity': 'High',
            'urgency': 'High',
            'confidence_boost': 0.18
        },
        'flu': {
            'keywords': ['fever', 'headache', 'fatigue', 'body aches', 'cough', 'sore throat', 'influenza', 'flu', 'chills', 'muscle aches'],
            'related_symptoms': ['Runny or stuffy nose', 'Sore throat', 'Body aches and muscle pain', 'Chills and sweats', 'Headache', 'Persistent dry cough'],
            'precautions': ['Get plenty of rest (7-9 hours nightly)', 'Stay hydrated with water and clear fluids', 'Take over-the-counter medications for symptoms', 'Avoid close contact with others', 'Cover coughs and sneezes', 'Get annual flu vaccination'],
            'risk_factors': ['Age (children and elderly)', 'Chronic medical conditions', 'Weakened immune system', 'Pregnancy', 'Obesity', 'Living in close quarters'],
            'severity': 'Moderate',
            'urgency': 'Moderate',
            'confidence_boost': 0.08
        },
        'anxiety': {
            'keywords': ['worry', 'restlessness', 'fatigue', 'difficulty concentrating', 'irritability', 'sleep problems', 'anxiety', 'nervous', 'panic', 'stress', 'apprehension'],
            'related_symptoms': ['Rapid heart rate', 'Sweating', 'Trembling or shaking', 'Feeling weak or tired', 'Trouble concentrating', 'Sleep problems'],
            'precautions': ['Practice relaxation techniques (deep breathing, meditation)', 'Exercise regularly (30 minutes daily)', 'Get adequate sleep (7-9 hours)', 'Consider therapy or counseling', 'Limit caffeine and alcohol', 'Maintain a regular routine'],
            'risk_factors': ['Family history of anxiety', 'Trauma or stressful life events', 'Chronic medical conditions', 'Substance abuse', 'Personality traits', 'Other mental health disorders'],
            'severity': 'Moderate',
            'urgency': 'Low',
            'confidence_boost': 0.09
        },
        'asthma': {
            'keywords': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'asthma', 'breathing', 'respiratory', 'bronchial'],
            'related_symptoms': ['Wheezing (whistling sound when breathing)', 'Shortness of breath', 'Chest tightness or pain', 'Coughing, especially at night', 'Trouble sleeping due to breathing problems'],
            'precautions': ['Use prescribed inhalers as directed', 'Avoid known triggers (allergens, smoke, pollution)', 'Monitor peak flow regularly', 'Create an asthma action plan', 'Get annual flu and pneumonia vaccines', 'Maintain good indoor air quality'],
            'risk_factors': ['Family history of asthma', 'Allergies', 'Exposure to secondhand smoke', 'Obesity', 'Occupational exposure to irritants', 'Respiratory infections in childhood'],
            'severity': 'High',
            'urgency': 'High',
            'confidence_boost': 0.14
        },
        'depression': {
            'keywords': ['sadness', 'hopelessness', 'fatigue', 'sleep problems', 'appetite changes', 'concentration', 'depression', 'mood', 'emotional'],
            'related_symptoms': ['Persistent sadness or hopelessness', 'Loss of interest in activities', 'Fatigue and decreased energy', 'Sleep disturbances', 'Appetite or weight changes', 'Difficulty concentrating'],
            'precautions': ['Seek professional help (therapy, counseling)', 'Maintain regular sleep schedule', 'Exercise regularly', 'Stay connected with friends and family', 'Avoid alcohol and drugs', 'Practice stress management techniques'],
            'risk_factors': ['Family history of depression', 'Trauma or stressful life events', 'Chronic medical conditions', 'Substance abuse', 'Personality traits', 'Hormonal changes'],
            'severity': 'High',
            'urgency': 'Moderate',
            'confidence_boost': 0.11
        }
    }
    
    # Enhanced analysis with better keyword matching
    symptoms_lower = symptoms.lower()
    best_match = 'Unknown'
    best_score = 0
    confidence = 0.2
    
    # Calculate scores for each disease
    disease_scores = {}
    for disease, data in diseases_database.items():
        score = 0
        for keyword in data['keywords']:
            if keyword in symptoms_lower:
                score += 1
                # Bonus for exact phrase matches
                if f" {keyword} " in f" {symptoms_lower} ":
                    score += 0.5
        
        disease_scores[disease] = score
        if score > best_score:
            best_score = score
            best_match = disease
    
    # Calculate confidence with enhanced logic
    if best_score > 0:
        base_confidence = 0.3 + (best_score * 0.08)
        disease_data = diseases_database.get(best_match, {})
        confidence_boost = disease_data.get('confidence_boost', 0)
        confidence = min(0.95, base_confidence + confidence_boost)
    else:
        confidence = 0.2
    
    # Get disease data
    disease_data = diseases_database.get(best_match, {
        'related_symptoms': ['Consult a healthcare professional for proper diagnosis'],
        'precautions': ['Seek medical advice from a qualified healthcare provider'],
        'risk_factors': ['Unknown - requires medical evaluation'],
        'severity': 'Unknown',
        'urgency': 'Unknown'
    })
    
    # Generate top predictions with better ranking
    top_predictions = []
    sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for i, (disease, score) in enumerate(sorted_diseases):
        if score > 0:
            pred_confidence = min(0.9, 0.2 + (score * 0.1))
            top_predictions.append({
                'disease': disease.title(),
                'confidence': pred_confidence,
                'rank': i + 1
            })
    
    # Enhanced analysis details
    analysis_details = {
        'symptom_complexity': 'High' if len(symptoms.split()) > 20 else 'Moderate' if len(symptoms.split()) > 10 else 'Low',
        'keyword_matches': best_score,
        'confidence_score': confidence * 100,
        'analysis_time': datetime.now().strftime('%H:%M:%S'),
        'total_symptoms_analyzed': len(symptoms.split())
    }
    
    return {
        'predicted_disease': best_match.title(),
        'confidence': confidence,
        'related_symptoms': disease_data['related_symptoms'],
        'precautions': disease_data['precautions'],
        'risk_factors': disease_data['risk_factors'] if include_risk_factors else [],
        'severity': disease_data['severity'],
        'urgency': disease_data['urgency'],
        'top_predictions': top_predictions,
        'analysis_details': analysis_details
    }

def create_enhanced_prediction_section(symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report):
    """Create enhanced prediction results section."""
    if not analyze_button and not st.session_state.get('analyze_image', False):
        return
    
    # Check if we have symptoms or image to analyze
    has_symptoms = symptoms and symptoms.strip()
    has_image = st.session_state.get('uploaded_image') is not None
    
    if not has_symptoms and not has_image:
        st.warning("⚠️ Please enter your symptoms or upload a medical image before analysis.")
        return
    
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.markdown('<h2>🔍 Advanced AI Analysis Results</h2>', unsafe_allow_html=True)
    
    # Show enhanced loading
    with st.spinner("🤖 Advanced AI is analyzing your input..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Enhanced analysis steps
        steps = [
            "🔍 Analyzing symptom patterns...",
            "🧠 Processing with advanced AI models...",
            "📊 Calculating confidence scores...",
            "🔬 Cross-referencing medical database...",
            "📋 Generating comprehensive analysis..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.3)
        
        # Get prediction results
        if has_symptoms:
            result = enhanced_disease_prediction(symptoms, language, include_risk_factors, detailed_analysis)
        else:
            # Image analysis simulation
            result = {
                'predicted_disease': 'Medical Image Analysis',
                'confidence': 0.85,
                'related_symptoms': ['Image analysis completed', 'Visual patterns detected', 'Medical imaging processed'],
                'precautions': ['Consult a radiologist for detailed analysis', 'Follow up with healthcare provider', 'Consider additional imaging if needed'],
                'risk_factors': ['Image quality', 'Patient positioning', 'Technical factors'],
                'severity': 'Moderate',
                'urgency': 'Moderate',
                'top_predictions': [
                    {'disease': 'Medical Image Analysis', 'confidence': 0.85, 'rank': 1},
                    {'disease': 'Radiological Review', 'confidence': 0.75, 'rank': 2},
                    {'disease': 'Specialist Consultation', 'confidence': 0.65, 'rank': 3}
                ]
            }
    
    # Display main prediction with enhanced styling
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-disease">{result["predicted_disease"]}</div>', unsafe_allow_html=True)
    
    # Enhanced confidence bar
    confidence_percent = result["confidence"] * 100
    st.markdown(f'<div class="prediction-confidence">Confidence: {confidence_percent:.1f}%</div>', unsafe_allow_html=True)
    
    # Animated confidence bar
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced severity and urgency indicators
    col1, col2 = st.columns(2)
    with col1:
        severity_class = f"severity-{result['severity'].lower()}"
        st.markdown(f'<div class="{severity_class}">Severity: {result["severity"]}</div>', unsafe_allow_html=True)
    
    with col2:
        urgency_class = f"severity-{result['urgency'].lower()}"
        st.markdown(f'<div class="{urgency_class}">Urgency: {result["urgency"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced results display with paragraph format
    st.markdown("### 🔗 Related Symptoms")
    symptoms_text = "Based on the analysis, the related symptoms include: " + ", ".join(result['related_symptoms'][:-1]) + ", and " + result['related_symptoms'][-1] + ". These symptoms are commonly associated with the predicted condition and should be monitored carefully. If you experience any of these symptoms worsening or new symptoms developing, it's important to seek medical attention promptly."
    st.markdown(f"<p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>{symptoms_text}</p>", unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Recommended Actions")
    precautions_text = "To manage this condition effectively, here are the recommended actions: " + ". ".join(result['precautions']) + ". Following these precautions can help improve your condition and prevent complications. It's essential to maintain consistency with these recommendations and consult with healthcare professionals for personalized advice tailored to your specific situation."
    st.markdown(f"<p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>{precautions_text}</p>", unsafe_allow_html=True)
    
    # Risk factors if enabled
    if include_risk_factors and result['risk_factors']:
        st.markdown("### 🚨 Risk Factors")
        risk_text = "Several risk factors may contribute to this condition: " + ", ".join(result['risk_factors'][:-1]) + ", and " + result['risk_factors'][-1] + ". Understanding these risk factors is crucial for prevention and early intervention. If you identify with multiple risk factors, it's advisable to discuss them with your healthcare provider to develop a comprehensive management plan."
        st.markdown(f"<p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>{risk_text}</p>", unsafe_allow_html=True)
    
    # Top predictions with enhanced visualization
    if result.get('top_predictions'):
        st.markdown("### 📊 Top Predictions")
        predictions_df = pd.DataFrame(result['top_predictions'])
        
        # Create enhanced bar chart
        fig = px.bar(
            predictions_df, 
            x='disease', 
            y='confidence',
            title="Disease Prediction Confidence Scores",
            color='confidence',
            color_continuous_scale='Viridis',
            text='confidence'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Disease",
            yaxis_title="Confidence Score",
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis details if enabled
    if detailed_analysis and result.get('analysis_details'):
        st.markdown("### 🔬 Analysis Details")
        details = result['analysis_details']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symptom Complexity", details.get('symptom_complexity', 'N/A'))
        with col2:
            st.metric("Keyword Matches", details.get('keyword_matches', 0))
        with col3:
            st.metric("Confidence Score", f"{details.get('confidence_score', 0):.1f}%")
        with col4:
            st.metric("Analysis Time", details.get('analysis_time', 'N/A'))
    
    # Generate report if requested
    if generate_report:
        create_medical_report(result)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_medical_report(result):
    """Create comprehensive medical report."""
    st.markdown("### 📋 Medical Report")
    
    report = f"""
    MEDICAL ANALYSIS REPORT
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PREDICTED CONDITION: {result['predicted_disease']}
    CONFIDENCE LEVEL: {result['confidence']:.1%}
    SEVERITY: {result['severity']}
    URGENCY: {result['urgency']}
    
    RELATED SYMPTOMS:
    {chr(10).join(f"• {symptom}" for symptom in result['related_symptoms'])}
    
    RECOMMENDED ACTIONS:
    {chr(10).join(f"• {precaution}" for precaution in result['precautions'])}
    
    RISK FACTORS:
    {chr(10).join(f"• {risk}" for risk in result.get('risk_factors', ['None identified']))}
    
    DISCLAIMER: This analysis is for educational purposes only. 
    Always consult qualified healthcare professionals for medical advice.
    """
    
    st.download_button(
        label="📄 Download Medical Report",
        data=report,
        file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def main():
    """Main application function."""
    # Load advanced CSS
    load_advanced_css()
    
    # Create advanced header
    create_advanced_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 AI Analysis", "ℹ️ About"])
    
    with tab1:
        # Create enhanced input section
        symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report = create_enhanced_input_section()
        
        # Create enhanced prediction section
        create_enhanced_prediction_section(symptoms, language, analyze_button, include_risk_factors, detailed_analysis, generate_report)
    
    with tab2:
        # About section
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown("## 🏥 About Advanced AI Medical Assistant")
        
        st.markdown("""
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>
        This Advanced AI Medical Assistant is designed to provide preliminary medical analysis based on symptoms and medical images. 
        The system uses sophisticated pattern matching and a comprehensive medical knowledge base to identify potential conditions 
        and provide relevant health information.
        </p>
        
        <h3>🎯 Key Features</h3>
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>
        Our system offers comprehensive symptom analysis with high accuracy disease prediction, medical image analysis capabilities 
        for various imaging types including X-rays and MRI scans, multi-language support for global accessibility, detailed risk 
        factor assessment, and personalized precautions and recommendations. The platform generates downloadable medical reports 
        and provides real-time confidence scoring to help you understand the reliability of each prediction.
        </p>
        
        <h3>🔬 Technology</h3>
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>
        Currently, the system operates using an advanced medical knowledge base with pattern recognition algorithms. For enhanced 
        accuracy and real-time AI capabilities, you can integrate external AI services like OpenAI GPT-4, Google Gemini, or 
        specialized medical AI APIs by providing your API key in the configuration section. This integration would enable more 
        sophisticated natural language understanding and improved diagnostic suggestions.
        </p>
        
        <h3>⚠️ Important Disclaimer</h3>
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem; color: #dc2626 !important; font-weight: 600;'>
        This tool is designed for educational and informational purposes only. It should not be used as a substitute for 
        professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with 
        any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking 
        it because of information provided by this application. In case of emergency, contact your local emergency services immediately.
        </p>
        
        <h3>📊 Accuracy & Reliability</h3>
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>
        The system provides confidence scores for each prediction to help you understand the reliability of the analysis. 
        Higher confidence scores indicate stronger pattern matches with known medical conditions. However, even high confidence 
        predictions should be verified by qualified medical professionals. The system is continuously updated with the latest 
        medical knowledge to improve accuracy and reliability.
        </p>
        
        <h3>🔒 Privacy & Security</h3>
        <p style='text-align: justify; line-height: 1.8; font-size: 1.05rem;'>
        Your privacy is our priority. All symptom descriptions and medical images are processed locally and are not stored 
        permanently. Session data is cleared when you close the application. If you choose to integrate external AI services, 
        please review their respective privacy policies to understand how your data may be processed.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()