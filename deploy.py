"""
Deployment script for AI Medical Assistant
Automates deployment to Streamlit Cloud and HuggingFace Spaces
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_git_status():
    """Check if git repository is clean."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            logger.warning("Uncommitted changes detected:")
            print(result.stdout)
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking git status: {e}")
        return False

def create_deployment_files():
    """Create necessary files for deployment."""
    logger.info("Creating deployment files...")
    
    # Create .streamlit/config.toml
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    config_content = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    with open(streamlit_dir / "config.toml", "w") as f:
        f.write(config_content)
    
    # Create app.py for HuggingFace Spaces
    app_content = '''import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from web_app.multilang_app import main

if __name__ == "__main__":
    main()
'''
    
    with open("app.py", "w") as f:
        f.write(app_content)
    
    # Create .gitignore if not exists
    if not Path(".gitignore").exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Temporary files
temp_uploads/
temp/
*.tmp

# Model files (large)
models/
*.pkl
*.joblib
*.h5
*.pb
*.onnx
*.pt
*.pth

# Cache
.cache/
huggingface_cache/
transformers_cache/
"""
        
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
    
    logger.info("Deployment files created successfully")

def prepare_for_streamlit_cloud():
    """Prepare repository for Streamlit Cloud deployment."""
    logger.info("Preparing for Streamlit Cloud deployment...")
    
    # Check git status
    if not check_git_status():
        response = input("Uncommitted changes detected. Continue? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create deployment files
    create_deployment_files()
    
    # Commit changes
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Prepare for Streamlit Cloud deployment'], check=True)
        subprocess.run(['git', 'push'], check=True)
        logger.info("Changes committed and pushed to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error committing changes: {e}")
        return False

def prepare_for_huggingface_spaces():
    """Prepare repository for HuggingFace Spaces deployment."""
    logger.info("Preparing for HuggingFace Spaces deployment...")
    
    # Create deployment files
    create_deployment_files()
    
    # Create README for HuggingFace Spaces
    hf_readme_content = """---
title: AI Medical Assistant
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
license: mit
---

# AI Medical Assistant

An advanced AI-powered medical assistant that combines Natural Language Processing (NLP) and Computer Vision (CV) for comprehensive disease prediction.

## Features

- **Multi-language Support**: English and Hindi
- **Text Analysis**: NLP models for symptom analysis
- **Image Analysis**: CV models for medical image processing
- **Unified Prediction**: Combined AI insights
- **Query Logging**: For model improvement
- **Analytics Dashboard**: Usage statistics and insights

## Usage

1. Enter your symptoms in the text area
2. Optionally upload a medical image
3. Select your preferred language
4. Click "Analyze Symptoms" to get predictions
5. View results with confidence scores and recommendations

## Disclaimer

This application is for educational and research purposes only. It is not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## Technology Stack

- **NLP Models**: BERT, BioBERT, ClinicalBERT
- **CV Models**: ResNet, EfficientNet, DenseNet
- **Frameworks**: PyTorch, TensorFlow, Streamlit
- **Translation**: Google Translate API
"""
    
    with open("README_HF.md", "w") as f:
        f.write(hf_readme_content)
    
    logger.info("HuggingFace Spaces preparation completed")
    return True

def deploy_to_streamlit_cloud():
    """Deploy to Streamlit Cloud."""
    logger.info("Deploying to Streamlit Cloud...")
    
    if not prepare_for_streamlit_cloud():
        logger.error("Failed to prepare for Streamlit Cloud deployment")
        return False
    
    print("\n" + "="*60)
    print("STREAMLIT CLOUD DEPLOYMENT")
    print("="*60)
    print("1. Go to https://share.streamlit.io")
    print("2. Sign in with your GitHub account")
    print("3. Click 'New app'")
    print("4. Select your repository: your-username/AI_Medicine_Assistant")
    print("5. Set main file path: web_app/multilang_app.py")
    print("6. Set requirements file: requirements_deployment.txt")
    print("7. Click 'Deploy!'")
    print("\nYour app will be available at: https://your-app-name.streamlit.app")
    print("="*60)
    
    return True

def deploy_to_huggingface_spaces():
    """Deploy to HuggingFace Spaces."""
    logger.info("Deploying to HuggingFace Spaces...")
    
    if not prepare_for_huggingface_spaces():
        logger.error("Failed to prepare for HuggingFace Spaces deployment")
        return False
    
    print("\n" + "="*60)
    print("HUGGINGFACE SPACES DEPLOYMENT")
    print("="*60)
    print("1. Go to https://huggingface.co/spaces")
    print("2. Click 'Create new Space'")
    print("3. Choose 'Streamlit' as SDK")
    print("4. Set visibility to 'Public'")
    print("5. Name your space: ai-medical-assistant")
    print("6. Upload your code or connect to Git repository")
    print("7. Wait for deployment to complete")
    print("\nYour space will be available at: https://huggingface.co/spaces/your-username/ai-medical-assistant")
    print("="*60)
    
    return True

def create_dockerfile():
    """Create Dockerfile for containerized deployment."""
    dockerfile_content = """FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgcc-s1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "web_app/multilang_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("Dockerfile created")

def main():
    """Main deployment function."""
    print("🚀 AI Medical Assistant - Deployment Script")
    print("=" * 50)
    print("Choose deployment option:")
    print("1. Streamlit Cloud")
    print("2. HuggingFace Spaces")
    print("3. Create Dockerfile")
    print("4. Prepare for both platforms")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        deploy_to_streamlit_cloud()
    elif choice == '2':
        deploy_to_huggingface_spaces()
    elif choice == '3':
        create_dockerfile()
        print("Dockerfile created successfully!")
    elif choice == '4':
        prepare_for_streamlit_cloud()
        prepare_for_huggingface_spaces()
        print("Preparation completed for both platforms!")
    elif choice == '5':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
