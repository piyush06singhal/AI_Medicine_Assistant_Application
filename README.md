# AI Medical Assistant

A comprehensive AI-powered medical assistant with Natural Language Processing (NLP) and Computer Vision (CV) capabilities for medical text analysis and image processing.

## 🏗️ Project Structure

```
AI_Medicine_Assistant/
├── nlp_models/                 # Natural Language Processing models
│   ├── models/                 # Pre-trained and custom NLP models
│   ├── preprocessing/          # Text preprocessing utilities
│   └── training/               # Model training scripts
├── cv_models/                  # Computer Vision models
│   ├── models/                 # Pre-trained and custom CV models
│   ├── preprocessing/          # Image preprocessing utilities
│   └── training/               # Model training scripts
├── data/                       # Data storage
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   └── external/               # External data sources
├── web_app/                    # Streamlit web application
│   ├── static/                 # Static assets (CSS, JS, images)
│   ├── templates/              # HTML templates
│   └── api/                    # API endpoints
├── utils/                      # Utility functions
│   ├── data_processing/        # Data processing utilities
│   └── model_utils/            # Model management utilities
├── config/                     # Configuration files
│   └── environments/           # Environment-specific configs
├── tests/                      # Test files
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── setup_env.bat              # Windows setup script
├── setup_env.sh               # Linux/macOS setup script
├── Makefile                   # Common tasks
└── README.md                  # This file
```

## 🚀 Features

### Natural Language Processing
- Medical text analysis and classification
- Named Entity Recognition (NER) for medical entities
- Medical question answering
- Symptom extraction and analysis
- Drug interaction detection
- Clinical note summarization

### Computer Vision
- Medical image classification
- X-ray analysis
- MRI/CT scan processing
- Skin lesion detection
- Medical image segmentation
- Anomaly detection in medical images

### Web Application
- Interactive Streamlit interface
- Real-time model inference
- File upload and processing
- Results visualization
- User-friendly medical assistant interface

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA (optional, for GPU acceleration)

### Quick Setup

#### Windows
```bash
# Clone the repository
git clone <repository-url>
cd AI_Medicine_Assistant

# Run the setup script
setup_env.bat
```

#### Linux/macOS
```bash
# Clone the repository
git clone <repository-url>
cd AI_Medicine_Assistant

# Make setup script executable and run
chmod +x setup_env.sh
./setup_env.sh
```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

## 🎯 Usage

### Starting the Application

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Start the Streamlit app
streamlit run web_app/app.py
```

### Using the Makefile

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Start the application
make run
```

## 📦 Dependencies

### Core ML/AI Frameworks
- **PyTorch** (2.0.0+) - Deep learning framework
- **TensorFlow** (2.13.0+) - Machine learning platform
- **HuggingFace Transformers** (4.30.0+) - Pre-trained models
- **OpenCV** (4.8.0+) - Computer vision
- **scikit-learn** (1.3.0+) - Machine learning utilities

### Medical AI Libraries
- **medspacy** - Medical text processing
- **scispacy** - Scientific text processing
- **pymed** - PubMed API client
- **pydicom** - DICOM medical imaging
- **SimpleITK** - Medical image processing

### Web Application
- **Streamlit** (1.25.0+) - Web app framework
- **FastAPI** (0.100.0+) - API framework
- **Pydantic** (2.0.0+) - Data validation

### Development Tools
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking

## 🔧 Configuration

The application uses environment variables for configuration. Copy `env.example` to `.env` and modify as needed:

```bash
# Application Settings
DEBUG=False
API_HOST=localhost
API_PORT=8000

# Model Settings
NLP_MODEL_NAME=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
CV_MODEL_NAME=microsoft/resnet-50

# HuggingFace Settings
HF_TOKEN=your_huggingface_token_here
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_nlp_models.py
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running the application
- **Code Documentation**: Generated using Sphinx
- **User Guide**: See `docs/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests before committing
make test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Disclaimer

This software is for research and educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 🙏 Acknowledgments

- HuggingFace for pre-trained models
- Medical AI research community
- Open source contributors

## 📞 Support

For support, email team@aimedicalassistant.com or create an issue in the repository.

## 🔄 Version History

- **v1.0.0** - Initial release with basic NLP and CV capabilities
- **v1.1.0** - Added web interface and API endpoints
- **v1.2.0** - Enhanced medical text processing

---

**Note**: This is a template project structure. Customize the models, data processing, and web interface according to your specific medical AI requirements.
