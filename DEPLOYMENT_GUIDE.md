# AI Medical Assistant - Deployment Guide

This guide will help you deploy the AI Medical Assistant to Streamlit Cloud and HuggingFace Spaces.

## 🚀 Quick Deployment

### For Streamlit Cloud:

1. **Push to GitHub**: Make sure your code is pushed to a GitHub repository
2. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Deploy App**: 
   - Click "New app"
   - Select your repository
   - Set main file path: `standalone_app.py`
   - Set requirements file: `requirements_deployment.txt`
   - Click "Deploy!"

### For HuggingFace Spaces:

1. **Go to HuggingFace Spaces**: Visit [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Create New Space**:
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Set visibility to "Public"
   - Name your space: `ai-medical-assistant`
3. **Upload Files**:
   - Upload `app.py`
   - Upload `requirements_deployment.txt`
   - Upload `README_HF.md` (rename to `README.md`)
4. **Deploy**: The space will automatically deploy

## 📁 Required Files for Deployment

### Core Files:
- `standalone_app.py` - Main Streamlit application
- `app.py` - Entry point for HuggingFace Spaces
- `requirements_deployment.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

### Optional Files:
- `README_HF.md` - Documentation for HuggingFace Spaces
- `DEPLOYMENT_GUIDE.md` - This deployment guide

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`):
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

[browser]
gatherUsageStats = false
```

### Requirements (`requirements_deployment.txt`):
```
streamlit>=1.25.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
nltk>=3.8.0
tqdm>=4.65.0
```

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure you're using `standalone_app.py` which has no external dependencies
2. **Memory Issues**: The app uses lightweight dependencies to avoid memory constraints
3. **Translation Issues**: The app includes fallback translation without external APIs
4. **Model Loading**: The app uses simulation models that don't require large model files

### Local Testing:

Before deploying, test locally:
```bash
streamlit run standalone_app.py
```

### Deployment Checklist:

- [ ] All required files are in the repository
- [ ] `requirements_deployment.txt` contains only necessary packages
- [ ] `standalone_app.py` runs without errors locally
- [ ] No large model files are included (use simulation instead)
- [ ] All imports are working correctly

## 🌐 Deployment URLs

After successful deployment:

- **Streamlit Cloud**: `https://your-app-name.streamlit.app`
- **HuggingFace Spaces**: `https://huggingface.co/spaces/your-username/ai-medical-assistant`

## 📝 Features Included

### ✅ Working Features:
- Multi-language support (English/Hindi)
- Symptom analysis with keyword matching
- Disease prediction simulation
- Analytics dashboard
- Responsive design
- Medical disclaimers

### 🔄 Simulation Features:
- Disease prediction based on symptom keywords
- Confidence scoring
- Related symptoms and precautions
- Language detection and translation
- Analytics and metrics

## 🚨 Important Notes

1. **Educational Purpose**: This is for portfolio/educational use only
2. **Not for Medical Use**: Do not use for actual medical diagnosis
3. **Simulation Models**: Uses keyword-based simulation, not real ML models
4. **Lightweight**: Optimized for cloud deployment with minimal dependencies

## 📞 Support

If you encounter issues:

1. Check the logs in your deployment platform
2. Test locally first with `streamlit run standalone_app.py`
3. Ensure all files are properly uploaded
4. Verify requirements.txt is correct

## 🔄 Updates

To update your deployment:

1. Make changes to `standalone_app.py`
2. Push to GitHub (for Streamlit Cloud)
3. Update files in HuggingFace Spaces
4. The deployment will automatically update

---

**Happy Deploying! 🚀**
