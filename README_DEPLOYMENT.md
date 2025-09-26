# AI Medical Assistant - Deployment Guide

This guide provides step-by-step instructions for deploying the AI Medical Assistant on free platforms like Streamlit Cloud and HuggingFace Spaces.

## 🚀 Deployment Options

### 1. Streamlit Cloud (Recommended)

Streamlit Cloud is the easiest way to deploy Streamlit apps with free hosting.

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Your code pushed to a GitHub repository

#### Step-by-Step Instructions

1. **Prepare Your Repository**
   ```bash
   # Ensure your project structure is correct
   AI_Medicine_Assistant/
   ├── web_app/
   │   ├── multilang_app.py
   │   ├── medical_assistant_app.py
   │   └── enhanced_app.py
   ├── utils/
   │   ├── unified_predictor.py
   │   ├── query_logger.py
   │   └── multilang_support.py
   ├── requirements_deployment.txt
   └── README.md
   ```

2. **Create Streamlit Cloud App**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `your-username/AI_Medicine_Assistant`
   - Set main file path: `web_app/multilang_app.py`
   - Set requirements file: `requirements_deployment.txt`
   - Click "Deploy!"

3. **Configure Environment Variables** (Optional)
   ```
   NLP_MODEL_PATH=./models/medical_bert
   CV_MODEL_PATH=./models/medical_cnn
   LOG_LEVEL=INFO
   ```

4. **Access Your App**
   - Your app will be available at: `https://your-app-name.streamlit.app`
   - Share this URL with users

#### Streamlit Cloud Configuration

Create a `.streamlit/config.toml` file:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 2. HuggingFace Spaces

HuggingFace Spaces provides free hosting for ML applications.

#### Prerequisites
- HuggingFace account
- Your code in a Git repository

#### Step-by-Step Instructions

1. **Create a New Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Set visibility to "Public"
   - Name your space: `ai-medical-assistant`

2. **Upload Your Code**
   ```bash
   # Clone your space
   git clone https://huggingface.co/spaces/your-username/ai-medical-assistant
   cd ai-medical-assistant
   
   # Copy your files
   cp -r /path/to/AI_Medicine_Assistant/* .
   
   # Commit and push
   git add .
   git commit -m "Add AI Medical Assistant"
   git push
   ```

3. **Configure Space Settings**
   - Create `README.md` with space description
   - Ensure `app.py` points to your main Streamlit app
   - Add `requirements.txt` with dependencies

4. **Access Your Space**
   - Your space will be available at: `https://huggingface.co/spaces/your-username/ai-medical-assistant`

#### HuggingFace Spaces Configuration

Create `app.py` in the root directory:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from web_app.multilang_app import main

if __name__ == "__main__":
    main()
```

### 3. Other Deployment Options

#### Heroku (Paid)
```bash
# Create Procfile
echo "web: streamlit run web_app/multilang_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.9.16" > runtime.txt

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### Railway (Free Tier Available)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

## 📁 Required Files for Deployment

### Essential Files
```
AI_Medicine_Assistant/
├── web_app/
│   ├── multilang_app.py          # Main multi-language app
│   ├── medical_assistant_app.py  # Basic app
│   ├── enhanced_app.py          # Enhanced app
│   └── utils.py                 # Web utilities
├── utils/
│   ├── unified_predictor.py     # Unified prediction system
│   ├── query_logger.py          # Query logging
│   └── multilang_support.py     # Multi-language support
├── nlp_models/                  # NLP model files
├── cv_models/                   # CV model files
├── requirements_deployment.txt  # Deployment requirements
├── packages.txt                 # System packages (HuggingFace)
├── README.md                    # Project documentation
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

### Optional Files
```
├── .gitignore                   # Git ignore rules
├── LICENSE                      # License file
├── CHANGELOG.md                 # Version history
└── docs/                       # Additional documentation
```

## 🔧 Configuration

### Environment Variables

Set these environment variables in your deployment platform:

```bash
# Model paths
NLP_MODEL_PATH=./models/medical_bert
CV_MODEL_PATH=./models/medical_cnn

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# Translation
GOOGLE_TRANSLATE_API_KEY=your_api_key  # Optional

# Security
SECRET_KEY=your_secret_key
```

### Model Files

For deployment, you have several options:

1. **Use Pre-trained Models** (Recommended for demo)
   - Models will be downloaded automatically
   - No additional setup required

2. **Upload Custom Models**
   - Upload model files to your repository
   - Update model paths in configuration

3. **Use Model Hub**
   - Store models on HuggingFace Hub
   - Load models from hub URLs

## 🚀 Quick Deployment Commands

### Streamlit Cloud
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy AI Medical Assistant"
git push origin main

# 2. Deploy on Streamlit Cloud
# Go to share.streamlit.io and follow the UI
```

### HuggingFace Spaces
```bash
# 1. Create space
huggingface-cli repo create ai-medical-assistant --type space

# 2. Clone and setup
git clone https://huggingface.co/spaces/your-username/ai-medical-assistant
cd ai-medical-assistant

# 3. Copy files and deploy
cp -r /path/to/AI_Medicine_Assistant/* .
git add .
git commit -m "Deploy AI Medical Assistant"
git push
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Add to the top of your main app file
   import sys
   from pathlib import Path
   sys.path.append(str(Path(__file__).parent.parent))
   ```

2. **Model Loading Issues**
   ```python
   # Use try-catch for model loading
   try:
       predictor = UnifiedDiseasePredictor()
   except Exception as e:
       st.error(f"Model loading failed: {e}")
   ```

3. **Memory Issues**
   ```python
   # Use smaller models for deployment
   model_name = "distilbert-base-uncased"  # Instead of large models
   ```

4. **Translation Issues**
   ```python
   # Handle translation errors gracefully
   try:
       translated = translate_text(text)
   except Exception:
       translated = text  # Fallback to original
   ```

### Performance Optimization

1. **Lazy Loading**
   ```python
   @st.cache_resource
   def load_predictor():
       return UnifiedDiseasePredictor()
   ```

2. **Model Caching**
   ```python
   @st.cache_data
   def predict_disease_cached(symptoms, image_path):
       return predictor.predict_disease(symptoms, image_path)
   ```

3. **Resource Management**
   ```python
   # Clean up temporary files
   import tempfile
   import os
   
   # Clean up after each session
   for file in tempfile.gettempdir():
       if file.startswith("temp_"):
           os.remove(file)
   ```

## 📊 Monitoring and Analytics

### Query Logging
The app automatically logs all user queries for:
- Model improvement
- Usage analytics
- Performance monitoring

### Access Logs
```python
# View query statistics
from utils.query_logger import query_logger
stats = query_logger.get_query_stats(days=30)
print(f"Total queries: {stats['total_queries']}")
```

### Export Data
```python
# Export query data
export_path = query_logger.export_query_data(
    start_date="2024-01-01",
    end_date="2024-01-31",
    format="csv"
)
```

## 🔒 Security Considerations

1. **Data Privacy**
   - All user data is anonymized
   - No personal information is stored
   - Logs are encrypted

2. **API Keys**
   - Store sensitive keys as environment variables
   - Never commit API keys to repository

3. **Rate Limiting**
   - Implement rate limiting for API calls
   - Monitor usage patterns

## 📈 Scaling

### For High Traffic
1. **Use CDN** for static assets
2. **Implement caching** for predictions
3. **Use load balancers** for multiple instances
4. **Monitor resource usage**

### For Production
1. **Use dedicated servers**
2. **Implement proper logging**
3. **Set up monitoring and alerts**
4. **Regular backups**

## 🆘 Support

### Getting Help
1. Check the troubleshooting section
2. Review error logs
3. Test locally first
4. Contact support team

### Common Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [HuggingFace Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Deployment Best Practices](https://docs.streamlit.io/deploy)

---

**Note**: This deployment guide assumes you have basic knowledge of Git, Python, and web deployment. For advanced deployment scenarios, consult the platform-specific documentation.
