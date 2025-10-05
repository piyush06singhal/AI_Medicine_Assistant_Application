"""
Advanced AI Medical Assistant - Real AI-Powered Medical Analysis
Using Google Gemini AI for accurate medical insights
"""

import streamlit as st
import os
from datetime import datetime
import time

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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

def get_ai_analysis(symptoms, api_key):
    """Get real AI analysis using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are an expert medical AI assistant. Analyze the following symptoms and provide a comprehensive medical assessment.

Symptoms: {symptoms}

Please provide:
1. Most likely condition(s) based on these symptoms
2. Related symptoms to watch for
3. Recommended precautions and actions
4. When to seek immediate medical attention
5. Risk factors associated with this condition

Important: This is for informational purposes only and should not replace professional medical advice. Always recommend consulting with a healthcare provider.

Format your response clearly with sections."""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main application function."""
    load_advanced_css()
    create_header()
    
    # Check for API key in environment or session
    api_key = os.getenv('GEMINI_API_KEY') or st.session_state.get('gemini_api_key', '')
    
    # API Key input if not set
    if not api_key:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h2>🔑 API Configuration Required</h2>', unsafe_allow_html=True)
        st.info("To use real AI analysis, please enter your Google Gemini API key. Get one free at: https://aistudio.google.com/app/apikey")
        
        api_key_input = st.text_input(
            "Enter your Google Gemini API Key:",
            type="password",
            help="Your API key is stored only for this session and never saved"
        )
        
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            api_key = api_key_input
            st.success("✅ API Key configured! You can now use AI analysis.")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This tool provides AI-generated information for educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.")
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
        height=200,
        placeholder="Please provide detailed information about your symptoms, including:\n• Duration of symptoms\n• Severity (1-10 scale)\n• Associated symptoms\n• Any triggers or patterns\n• Previous medical history related to these symptoms",
        help="Be as detailed as possible for better AI analysis"
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
        if not symptoms or not symptoms.strip():
            st.error("❌ Please enter your symptoms before analysis.")
            return
        
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>🔍 AI Analysis Results</h2>', unsafe_allow_html=True)
        
        with st.spinner("🤖 AI is analyzing your symptoms... This may take a few moments."):
            if not GEMINI_AVAILABLE:
                st.error("❌ Google Gemini library not installed. Please install it with: pip install google-generativeai")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            analysis = get_ai_analysis(symptoms, api_key)
            
            if analysis.startswith("Error:"):
                st.error(f"❌ {analysis}")
                st.info("💡 Please check your API key and try again. Make sure you have a valid Google Gemini API key.")
            else:
                st.markdown(f"""
                <div style='background: #f7fafc; padding: 2rem; border-radius: 10px; border-left: 4px solid #667eea;'>
                    {analysis.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Analysis completed successfully!")
                
                # Add timestamp
                st.markdown(f"<p style='text-align: right; color: #718096; font-size: 0.9rem;'>Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
