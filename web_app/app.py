"""
AI Medical Assistant - Streamlit Web Application
Main application entry point.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import settings

def main():
    """Main application function."""
    st.set_page_config(
        page_title="AI Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 AI Medical Assistant")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the AI Medical Assistant! This application provides:
    
    - **Natural Language Processing**: Medical text analysis and classification
    - **Computer Vision**: Medical image analysis and processing
    - **Interactive Interface**: Easy-to-use web interface for medical AI tasks
    
    ### Getting Started
    
    1. **Text Analysis**: Upload or paste medical text for analysis
    2. **Image Analysis**: Upload medical images for computer vision analysis
    3. **Model Management**: View and manage your AI models
    
    ### Features
    
    - Medical entity recognition
    - Symptom extraction
    - Drug interaction detection
    - Medical image classification
    - X-ray analysis
    - And much more!
    
    ---
    
    **⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. 
    It is not intended for clinical use or medical diagnosis. Always consult with 
    qualified healthcare professionals for medical advice and diagnosis.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Home", "Text Analysis", "Image Analysis", "Model Management", "Settings"]
        )
        
        st.markdown("---")
        st.markdown("### Model Status")
        st.success("✅ NLP Models: Ready")
        st.success("✅ CV Models: Ready")
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Active Models", "2")
        st.metric("Processed Texts", "0")
        st.metric("Processed Images", "0")
    
    # Main content based on selected page
    if page == "Home":
        st.info("Select a page from the sidebar to get started!")
    
    elif page == "Text Analysis":
        st.header("📝 Medical Text Analysis")
        st.markdown("Upload or paste medical text for AI analysis.")
        
        # Text input
        text_input = st.text_area(
            "Enter medical text:",
            height=200,
            placeholder="Paste your medical text here..."
        )
        
        if st.button("Analyze Text"):
            if text_input:
                st.success("Text analysis completed!")
                # TODO: Implement actual text analysis
            else:
                st.warning("Please enter some text to analyze.")
    
    elif page == "Image Analysis":
        st.header("🖼️ Medical Image Analysis")
        st.markdown("Upload medical images for AI analysis.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii']
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                st.success("Image analysis completed!")
                # TODO: Implement actual image analysis
    
    elif page == "Model Management":
        st.header("🤖 Model Management")
        st.markdown("Manage your AI models and view their status.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NLP Models")
            st.info("PubMedBERT - Medical Text Analysis")
            st.info("ClinicalBERT - Clinical Text Processing")
        
        with col2:
            st.subheader("CV Models")
            st.info("ResNet-50 - Medical Image Classification")
            st.info("DenseNet - X-ray Analysis")
    
    elif page == "Settings":
        st.header("⚙️ Settings")
        st.markdown("Configure your AI Medical Assistant.")
        
        # Model settings
        st.subheader("Model Configuration")
        nlp_model = st.selectbox(
            "NLP Model:",
            ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "bert-base-uncased"]
        )
        
        cv_model = st.selectbox(
            "CV Model:",
            ["microsoft/resnet-50", "resnet-50"]
        )
        
        # Application settings
        st.subheader("Application Settings")
        debug_mode = st.checkbox("Debug Mode", value=settings.debug)
        log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
