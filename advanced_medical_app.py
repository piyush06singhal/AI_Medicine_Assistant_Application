"""
Advanced AI Medical Assistant - Real AI-Powered Medical Analysis
Using Google Gemini AI for accurate medical insights
"""

import streamlit as st
from datetime import datetime
from PIL import Image
import io

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Hardcoded API Key
GEMINI_API_KEY = "AIzaSyD6HMYeylRgqmUER5mbeBHKjnfapDOX-ho"

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
        max-width: 1200px;
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
    
    /* Input sections */
    .input-section {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 2px solid #e0e0e0;
    }
    
    .input-section h2, .input-section h3 {
        color: #1a1a1a !important;
        margin-bottom: 1rem;
    }
    
    /* Results card */
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
        margin-bottom: 1rem;
    }
    
    .medical-card p, .medical-card span, .medical-card div, .medical-card li {
        color: #2d3748 !important;
        line-height: 1.8;
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
    
    /* Labels */
    label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Input fields */
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
    
    .stError {
        background: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Global text visibility */
    p, span, div, li, td, th {
        color: #2d3748 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
    }
    
    .stSelectbox > div > div {
        background: #ffffff !important;
        color: #1a202c !important;
    }
    
    .stSelectbox label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .stTextArea label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .stTextInput label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    .stSpinner > div {
        color: #1a202c !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create application header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏥 Advanced AI Medical Assistant</h1>
        <p class="app-subtitle">Real AI-powered medical analysis using Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

def get_ai_text_analysis(symptoms, language):
    """Get real AI analysis using Google Gemini for text symptoms."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are an expert medical AI assistant. Analyze the following symptoms and provide a comprehensive medical assessment in {language}.

Symptoms: {symptoms}

Please provide:
1. Most likely condition(s) based on these symptoms
2. Related symptoms to watch for
3. Recommended precautions and actions
4. When to seek immediate medical attention
5. Risk factors associated with this condition

Important: This is for informational purposes only and should not replace professional medical advice. Always recommend consulting with a healthcare provider.

Format your response clearly with sections and use proper formatting."""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_ai_image_analysis(image, additional_info=""):
    """Get real AI analysis using Google Gemini Vision for medical images."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro-vision')
        
        prompt = f"""You are an expert medical AI assistant specializing in medical image analysis. Analyze this medical image carefully.

{f"Additional Information: {additional_info}" if additional_info else ""}

Please provide:
1. What you observe in the image (describe any visible conditions, abnormalities, or patterns)
2. Possible medical conditions based on the visual analysis
3. Recommended precautions and next steps
4. When to seek immediate medical attention
5. Important considerations for this type of condition

Important: This is for informational purposes only and should not replace professional medical diagnosis. Always recommend consulting with a qualified healthcare provider or specialist for proper diagnosis and treatment.

Format your response clearly with sections."""
        
        response = model.generate_content([prompt, image])
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main application function."""
    load_advanced_css()
    create_header()
    
    if not GEMINI_AVAILABLE:
        st.error("❌ Google Gemini library not installed. Please install it with: pip install google-generativeai")
        return
    
    # Main input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Describe Your Symptoms</h2>', unsafe_allow_html=True)
    
    # Language selection
    language = st.selectbox(
        "Select Language:",
        ["English", "Hindi", "Spanish", "French", "German", "Italian"],
        help="Choose your preferred language for analysis"
    )
    
    # Symptoms input
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        height=150,
        placeholder="Please provide detailed information about your symptoms, including:\n• Duration of symptoms\n• Severity (1-10 scale)\n• Associated symptoms\n• Any triggers or patterns\n• Previous medical history related to these symptoms",
        help="Be as detailed as possible for better AI analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Image upload section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>📸 Upload Medical Image (Optional)</h2>', unsafe_allow_html=True)
    st.info("Upload medical images like X-rays, MRI scans, CT scans, skin conditions, rashes, or any other medical images for AI analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a medical image:",
        type=['png', 'jpg', 'jpeg'],
        help="Upload medical images for AI-powered visual analysis"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)
        
        # Additional info for image
        additional_info = st.text_input(
            "Additional information about the image (optional):",
            placeholder="E.g., Location on body, duration, any pain or symptoms related to this image",
            help="Provide context to help AI analyze the image better"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze with AI", 
            type="primary", 
            use_container_width=True,
            help="Click to start AI-powered medical analysis"
        )
    
    # Show disclaimer
    st.warning("⚠️ **Medical Disclaimer**: This tool provides AI-generated information for educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.")
    
    # Process analysis
    if analyze_button:
        has_symptoms = symptoms and symptoms.strip()
        has_image = uploaded_file is not None
        
        if not has_symptoms and not has_image:
            st.error("❌ Please enter your symptoms or upload a medical image before analysis.")
            return
        
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>🔍 AI Analysis Results</h2>', unsafe_allow_html=True)
        
        # Analyze text symptoms
        if has_symptoms:
            with st.spinner("🤖 AI is analyzing your symptoms... This may take a few moments."):
                analysis = get_ai_text_analysis(symptoms, language)
                
                if analysis.startswith("Error:"):
                    st.error(f"❌ {analysis}")
                else:
                    st.markdown("### 📋 Symptom Analysis")
                    st.markdown(analysis)
                    st.success("✅ Symptom analysis completed!")
        
        # Analyze image
        if has_image:
            with st.spinner("🤖 AI is analyzing your medical image... This may take a few moments."):
                image = Image.open(uploaded_file)
                additional_context = additional_info if 'additional_info' in locals() else ""
                image_analysis = get_ai_image_analysis(image, additional_context)
                
                if image_analysis.startswith("Error:"):
                    st.error(f"❌ {image_analysis}")
                else:
                    st.markdown("---")
                    st.markdown("### 🖼️ Medical Image Analysis")
                    st.markdown(image_analysis)
                    st.success("✅ Image analysis completed!")
        
        # Add timestamp
        st.markdown(f"<p style='text-align: right; color: #718096; font-size: 0.9rem; margin-top: 2rem;'>Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
