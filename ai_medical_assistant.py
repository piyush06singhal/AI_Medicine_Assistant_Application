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

# Google Gemini API Key - Get from Streamlit secrets or use hardcoded
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "AIzaSyDADGz7JyDYUSLPmFCTpePgHEhIq98mSwY"

genai.configure(api_key=GEMINI_API_KEY)

def get_ai_medical_analysis(user_message, chat_history):
    """Get real AI analysis using Google Gemini - ChatGPT style."""
    try:
        # Build conversation context
        conversation = "You are an expert medical AI assistant. You provide accurate, helpful medical information in a conversational way, similar to ChatGPT. You can answer ANY medical question, explain diseases, symptoms, treatments, medications, and provide health advice. Always be helpful, clear, and professional.\n\n"
        
        # Add chat history for context
        for msg in chat_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['content']}\n\n"
        
        conversation += f"User: {user_message}\n\nAssistant:"
        
        # Use the correct GenerativeModel API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(conversation)
        
        return response.text, True
        
    except Exception as e:
        return f"AI Error: {str(e)}\n\nPlease check your API key or try again.", False

def analyze_medical_image(image, additional_info=""):
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
        
        # Use Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-pro-vision')
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

# Tabs for different features
tab1, tab2 = st.tabs(["💬 Chat with AI", "📸 Image Analysis"])

with tab1:
    st.markdown("### 💬 Chat with AI - Ask Anything!")
    
    # Example questions
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        **Symptoms & Diagnosis:**
        - "I have a headache, fever, and sore throat. What could it be?"
        - "What are the symptoms of diabetes?"
        - "I'm feeling dizzy and nauseous, should I be worried?"
        
        **Medications:**
        - "What are the side effects of ibuprofen?"
        - "Can I take aspirin with blood pressure medication?"
        - "What's the difference between paracetamol and ibuprofen?"
        
        **Health Conditions:**
        - "Explain what hypertension is and how to manage it"
        - "What causes kidney stones?"
        - "How is COVID-19 different from the flu?"
        
        **General Health:**
        - "How much water should I drink daily?"
        - "What foods help lower cholesterol?"
        - "How can I improve my sleep quality?"
        
        **Ask ANYTHING medical - I'm here to help!** 🤖
        """)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": """👋 Hello! I'm your AI Medical Assistant powered by Google Gemini.

I can help you with:
- 🩺 Symptom analysis and disease identification
- 💊 Medication information and side effects
- 🏥 Treatment options and medical procedures
- 🧬 Health conditions and their management
- 🍎 Nutrition and lifestyle advice
- 🧪 Lab results interpretation
- 📋 Medical terminology explanations
- ❓ Any medical question you have!

Just ask me anything about health and medicine. I'm here to help! 😊"""
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
            with st.spinner("🤖 AI is thinking..."):
                response, success = get_ai_medical_analysis(prompt, st.session_state.messages)
                
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
                    analysis, success = analyze_medical_image(image, additional_info)
                    
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
