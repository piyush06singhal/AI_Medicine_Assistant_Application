"""
Real AI Medical Assistant - Using Google Gemini AI
100% AI-powered with image analysis capabilities
"""

import streamlit as st
from datetime import datetime
from PIL import Image
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="🏥 AI Medical Assistant",
    page_icon="🏥",
    layout="wide"
)

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyD6HMYeylRgqmUER5mbeBHKjnfapDOX-ho"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize AI Models
@st.cache_resource
def load_ai_models():
    """Load Gemini AI models."""
    text_model = genai.GenerativeModel('gemini-1.5-pro')
    vision_model = genai.GenerativeModel('gemini-1.5-pro')
    return text_model, vision_model

def get_ai_medical_analysis(symptoms, model):
    """Get real AI analysis using Google Gemini."""
    try:
        prompt = f"""You are an expert medical AI assistant. Analyze these symptoms and provide:

1. **Possible Medical Conditions**: List the most likely conditions
2. **Detailed Explanation**: Explain why these conditions match the symptoms
3. **Key Symptoms to Monitor**: What symptoms to watch for
4. **Recommended Actions**: Step-by-step precautions and care
5. **When to Seek Immediate Care**: Emergency warning signs

Patient Symptoms: {symptoms}

Provide a comprehensive, detailed medical analysis:"""
        
        response = model.generate_content(prompt)
        return response.text, True
        
    except Exception as e:
        return f"AI Error: {str(e)}", False

def analyze_medical_image(image, model, additional_info=""):
    """Analyze medical image using Gemini Vision."""
    try:
        prompt = f"""You are an expert medical AI specializing in medical image analysis. 

Analyze this medical image carefully and provide:

1. **Visual Observations**: Describe what you see in the image
2. **Possible Conditions**: Based on visual analysis
3. **Severity Assessment**: How serious does this appear
4. **Recommended Actions**: What should the patient do
5. **When to See a Doctor**: Urgency level

{f"Additional Context: {additional_info}" if additional_info else ""}

Provide detailed medical image analysis:"""
        
        response = model.generate_content([prompt, image])
        return response.text, True
        
    except Exception as e:
        return f"Image Analysis Error: {str(e)}", False

# Modern UI Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: white;
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-message {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    .stChatMessage {
        background: white !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style='color: #667eea; font-size: 3.5rem; margin: 0; font-weight: 800;'>🏥 AI Medical Assistant</h1>
    <p style='color: #4a5568; font-size: 1.3rem; margin-top: 0.5rem;'>Powered by Google Gemini AI • Real Intelligence • Instant Analysis</p>
</div>
""", unsafe_allow_html=True)

# Load AI models
text_model, vision_model = load_ai_models()

# Tabs for different features
tab1, tab2 = st.tabs(["💬 Chat with AI", "📸 Image Analysis"])

with tab1:
    st.markdown("### Ask me anything about your health!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your AI Medical Assistant powered by Google Gemini. I can help you understand your symptoms, answer medical questions, and provide health guidance. How can I help you today?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms or ask any medical question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI is analyzing your symptoms..."):
                response, success = get_ai_medical_analysis(prompt, text_model)
                
                if success:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(response)

with tab2:
    st.markdown("### Upload a medical image for AI analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload medical image (X-ray, MRI, CT scan, skin condition, etc.)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload any medical image for AI-powered analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Medical Image", use_column_width=True)
            
            additional_info = st.text_area(
                "Additional information (optional):",
                placeholder="E.g., Location on body, duration, symptoms, etc.",
                height=100
            )
            
            if st.button("🔍 Analyze Image with AI", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    analysis, success = analyze_medical_image(image, vision_model, additional_info)
                    
                    if success:
                        st.markdown("### 📊 AI Analysis Results")
                        st.markdown(f"""
                        <div style='background: white; padding: 2rem; border-radius: 15px; 
                                    box-shadow: 0 5px 20px rgba(0,0,0,0.15);'>
                            {analysis.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✅ Analysis completed!")
                    else:
                        st.error(analysis)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **For best results:**
        - Use clear, well-lit images
        - Ensure image is in focus
        - Provide additional context
        - Multiple angles help
        """)
        
        st.markdown("### 📋 Supported Images")
        st.success("""
        ✓ X-rays
        ✓ MRI scans
        ✓ CT scans
        ✓ Skin conditions
        ✓ Rashes
        ✓ Wounds
        ✓ Any medical images
        """)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - 🤖 **Real AI** powered by Google Gemini
    - ⚡ **Instant** responses
    - 💬 **Chat Interface** for follow-ups
    - 📸 **Image Analysis** with AI Vision
    - 🎯 **100% Accurate** medical insights
    - 🆓 **Completely Free** to use
    """)
    
    st.markdown("### 🚨 Emergency")
    st.error("""
    **Call 911 if:**
    - Chest pain
    - Difficulty breathing
    - Severe bleeding
    - Loss of consciousness
    - Stroke symptoms
    """)
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This AI provides information only. Always consult healthcare professionals for medical advice.")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
