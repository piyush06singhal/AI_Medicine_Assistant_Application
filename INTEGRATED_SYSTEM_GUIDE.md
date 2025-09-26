# AI Medical Assistant - Integrated System Guide

This guide explains how to use the complete AI Medical Assistant system that combines NLP and Computer Vision models for comprehensive disease prediction.

## 🏗️ System Overview

The integrated system consists of:
- **Unified Predictor**: Combines NLP and CV models for comprehensive predictions
- **Streamlit Web App**: User-friendly interface for input and results
- **Web Utilities**: Helper functions for the web application
- **History Tracking**: Analytics and prediction history
- **Export Functionality**: Download results and reports

## 📁 File Structure

```
utils/
└── unified_predictor.py          # Unified prediction system

web_app/
├── medical_assistant_app.py      # Basic Streamlit app
├── enhanced_app.py              # Enhanced app with analytics
└── utils.py                     # Web app utilities

# Main scripts
run_web_app.py                   # Web app launcher
test_integrated_system.py        # Integration tests
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Additional dependencies for web app
pip install streamlit plotly
```

### 2. Test the System

```bash
# Run integration tests
python test_integrated_system.py
```

### 3. Start the Web App

```bash
# Run the web app launcher
python run_web_app.py

# Or run directly
streamlit run web_app/medical_assistant_app.py
streamlit run web_app/enhanced_app.py
```

## 🔧 Usage Examples

### Basic Disease Prediction

```python
from utils.unified_predictor import predict_disease

# Text-only prediction
result = predict_disease("frequent urination, excessive thirst, fatigue, blurred vision")

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Related Symptoms: {result['related_symptoms']}")
print(f"Precautions: {result['precautions']}")
```

### Combined Text and Image Prediction

```python
from utils.unified_predictor import predict_disease

# Combined prediction
result = predict_disease(
    symptoms="chest pain, shortness of breath, coughing",
    image_path="path/to/chest_xray.jpg"
)

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Source: {result['unified_prediction']['source']}")
```

### Advanced Usage with Custom Models

```python
from utils.unified_predictor import UnifiedDiseasePredictor

# Initialize with custom model paths
predictor = UnifiedDiseasePredictor(
    nlp_model_path="./models/medical_bert",
    cv_model_path="./models/medical_cnn"
)

# Make prediction
result = predictor.predict_disease(
    symptoms="headache, dizziness, chest pain",
    image_path="path/to/medical_image.jpg",
    return_probabilities=True,
    top_k=3
)

# Access detailed results
print(f"Top Prediction: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")

if result['top_predictions']:
    print("All Predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['rank']}. {pred['disease']} ({pred['confidence']:.3f})")
```

## 🌐 Web Application

### Basic App Features

The basic Streamlit app (`medical_assistant_app.py`) provides:
- **Symptom Input**: Text area for detailed symptom description
- **Image Upload**: Optional medical image upload
- **Prediction Display**: Results with confidence scores
- **Related Information**: Symptoms and precautions
- **Medical Disclaimer**: Important safety warnings

### Enhanced App Features

The enhanced app (`enhanced_app.py`) includes:
- **Analytics Dashboard**: Prediction history and statistics
- **Model Status**: Real-time model availability
- **Export Functionality**: Download results and reports
- **History Tracking**: Save and view past predictions
- **Interactive Charts**: Visual analytics with Plotly

### Web App Usage

1. **Start the App**:
   ```bash
   python run_web_app.py
   ```

2. **Choose Version**:
   - Basic App: Simple interface
   - Enhanced App: Full features with analytics

3. **Input Symptoms**:
   - Enter detailed symptom description
   - Upload medical image (optional)
   - Click "Analyze Symptoms"

4. **View Results**:
   - Predicted disease with confidence
   - Related symptoms and precautions
   - Top predictions and model contributions

5. **Export Results**:
   - Download prediction report
   - View analytics dashboard
   - Track prediction history

## 🔍 Unified Prediction System

### How It Works

The unified predictor combines insights from both NLP and CV models:

1. **Text Analysis**: NLP models analyze symptom descriptions
2. **Image Analysis**: CV models examine medical images
3. **Weighted Combination**: Results are combined using configurable weights
4. **Unified Output**: Single prediction with confidence and recommendations

### Prediction Weights

```python
# Default weights
nlp_weight = 0.6  # 60% weight for NLP predictions
cv_weight = 0.4   # 40% weight for CV predictions

# Update weights
predictor.update_weights(nlp_weight=0.7, cv_weight=0.3)
```

### Model Availability

The system handles different model availability scenarios:

- **Both Models**: Combined prediction with weighted voting
- **NLP Only**: Text-based prediction
- **CV Only**: Image-based prediction
- **No Models**: Fallback to generic recommendations

## 📊 Analytics and History

### Prediction History

```python
from web_app.utils import save_prediction_history, load_prediction_history

# Save prediction
save_prediction_history(prediction_data)

# Load history
history = load_prediction_history()

# Create analytics
analytics = create_analytics_dashboard(history)
```

### Analytics Dashboard

The analytics dashboard provides:
- **Total Predictions**: Count of all predictions made
- **Confidence Statistics**: Mean, median, min, max confidence
- **Disease Distribution**: Pie chart of predicted diseases
- **Model Usage**: Bar chart of model usage patterns
- **Recent Predictions**: Table of recent prediction history

## 🎨 Web App Styling

### Custom CSS Features

The web app includes comprehensive styling:
- **Gradient Headers**: Beautiful gradient backgrounds
- **Card Layouts**: Clean card-based design
- **Responsive Design**: Mobile-friendly interface
- **Color Coding**: Confidence-based color schemes
- **Interactive Elements**: Hover effects and animations

### Styling Components

- **Main Header**: Gradient background with title
- **Input Sections**: Styled text areas and file uploaders
- **Prediction Results**: Highlighted results with confidence
- **Info Boxes**: Color-coded information displays
- **Warning Boxes**: Important medical disclaimers
- **Metrics**: Styled metric displays

## 🔧 Configuration

### Model Paths

```python
# Custom model paths
predictor = UnifiedDiseasePredictor(
    nlp_model_path="./models/custom_nlp",
    cv_model_path="./models/custom_cv"
)
```

### Web App Configuration

```python
# Streamlit page config
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Environment Variables

```bash
# Optional environment variables
NLP_MODEL_PATH=./models/medical_bert
CV_MODEL_PATH=./models/medical_cnn
HISTORY_FILE=prediction_history.json
```

## 🧪 Testing

### Run Tests

```bash
# Test integrated system
python test_integrated_system.py

# Test specific components
python -c "
from test_integrated_system import test_unified_predictor
test_unified_predictor()
"
```

### Test Coverage

The test suite covers:
- **Unified Predictor**: Core prediction functionality
- **Web Utils**: Utility functions
- **History Tracking**: Prediction history
- **Export Functionality**: Download and export
- **Image Validation**: File upload validation
- **Web App Imports**: Application imports
- **Integration Workflow**: End-to-end testing

## 📈 Performance Optimization

### Model Loading

```python
# Lazy loading for better performance
predictor = UnifiedDiseasePredictor()
# Models are loaded only when needed
```

### Caching

```python
# Streamlit caching for repeated operations
@st.cache_data
def load_prediction_history():
    return load_prediction_history()
```

### Memory Management

```python
# Clean up temporary files
import tempfile
import os

# Clean up after processing
for file in tempfile.gettempdir():
    if file.startswith("temp_"):
        os.remove(file)
```

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This system is for research and educational purposes only.**
- Not intended for clinical use or medical diagnosis
- Always consult qualified healthcare professionals
- Use only for research and educational purposes

### Data Privacy

- Ensure medical data is properly anonymized
- Follow HIPAA and other privacy regulations
- Use secure data storage and transmission

### Model Limitations

- Performance depends on training data quality
- May not generalize to all medical conditions
- Requires regular retraining with new data
- Should be validated by medical professionals

## 🔧 Troubleshooting

### Common Issues

1. **Models Not Found**
   ```python
   # Check model paths
   import os
   print(os.path.exists("./models/medical_bert"))
   print(os.path.exists("./models/medical_cnn"))
   ```

2. **Web App Not Starting**
   ```bash
   # Check Streamlit installation
   pip install streamlit
   # Check port availability
   streamlit run app.py --server.port 8502
   ```

3. **Prediction Errors**
   ```python
   # Check model status
   predictor = UnifiedDiseasePredictor()
   status = predictor.get_model_status()
   print(status)
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python run_web_app.py --verbose
```

## 📚 API Reference

### UnifiedDiseasePredictor Class

```python
class UnifiedDiseasePredictor:
    def __init__(self, nlp_model_path: str, cv_model_path: str)
    def predict_disease(self, symptoms: str, image_path: Optional[str] = None) -> Dict
    def get_model_status(self) -> Dict
    def update_weights(self, nlp_weight: float, cv_weight: float)
```

### Web App Utilities

```python
# History functions
save_prediction_history(prediction_data: Dict, history_file: str)
load_prediction_history(history_file: str) -> List[Dict]

# Validation functions
validate_symptoms_input(symptoms: str) -> Tuple[bool, str]
validate_image_file(uploaded_file) -> Tuple[bool, str]

# Utility functions
format_confidence(confidence: float) -> str
create_prediction_summary(prediction_data: Dict) -> Dict
create_analytics_dashboard(history: List[Dict]) -> Dict
```

## 🤝 Contributing

To contribute to the integrated system:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test scripts for examples

---

**Remember**: This system is for educational and research purposes only. Always consult medical professionals for actual medical advice and diagnosis.
