# Medical Disease Prediction System - Usage Guide

This guide explains how to use the AI Medical Assistant's disease prediction system built with BERT/BioBERT models.

## 🏗️ System Overview

The disease prediction system consists of:
- **Data Loader**: Handles CSV/JSON medical symptom datasets
- **Model Trainer**: Trains BERT/BioBERT models for disease classification
- **Disease Predictor**: Provides predictions with related symptoms and precautions
- **Sample Dataset**: Medical symptoms dataset for testing

## 📁 File Structure

```
nlp_models/
├── preprocessing/
│   └── data_loader.py          # Data loading and preprocessing
├── training/
│   └── model_trainer.py        # Model training utilities
└── models/
    └── disease_predictor.py    # Disease prediction and inference

data/raw/
└── medical_symptoms_dataset.csv # Sample medical symptoms dataset

# Main scripts
train_disease_model.py          # Training script
demo_disease_prediction.py      # Demonstration script
test_disease_prediction.py      # Test script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements_disease_prediction.txt

# Or install from main requirements
pip install -r requirements.txt
```

### 2. Test the System

```bash
# Run tests to verify everything works
python test_disease_prediction.py
```

### 3. Train the Model

```bash
# Train the disease prediction model
python train_disease_model.py

# With custom parameters
python train_disease_model.py \
    --data_path data/raw/medical_symptoms_dataset.csv \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir ./models/medical_bert \
    --num_epochs 5 \
    --batch_size 16
```

### 4. Test Predictions

```bash
# Run interactive demo
python demo_disease_prediction.py
```

## 🔧 Usage Examples

### Basic Disease Prediction

```python
from nlp_models.models.disease_predictor import predict_disease_from_text

# Simple prediction
symptoms = "frequent urination, excessive thirst, fatigue, blurred vision"
result = predict_disease_from_text(symptoms)

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Related Symptoms: {result['related_symptoms']}")
print(f"Precautions: {result['precautions']}")
```

### Advanced Usage with Custom Model

```python
from nlp_models.models.disease_predictor import DiseasePredictor

# Initialize predictor with custom model
predictor = DiseasePredictor("./models/medical_bert")

# Make prediction
result = predictor.predict_disease_from_text(
    "headache, dizziness, chest pain, shortness of breath",
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

### Batch Predictions

```python
# Predict for multiple symptom texts
symptoms_list = [
    "frequent urination, excessive thirst, fatigue",
    "wheezing, shortness of breath, chest tightness",
    "severe headache, nausea, sensitivity to light"
]

results = predictor.batch_predict(symptoms_list)

for i, result in enumerate(results):
    print(f"Case {i+1}: {result['predicted_disease']} ({result['confidence']:.3f})")
```

## 📊 Data Format

### Input Dataset Format

The system expects CSV or JSON files with the following structure:

**CSV Format:**
```csv
symptoms,disease
"frequent urination, excessive thirst, fatigue","diabetes"
"headache, dizziness, chest pain","hypertension"
"wheezing, shortness of breath, coughing","asthma"
```

**JSON Format:**
```json
[
  {
    "symptoms": "frequent urination, excessive thirst, fatigue",
    "disease": "diabetes"
  },
  {
    "symptoms": "headache, dizziness, chest pain",
    "disease": "hypertension"
  }
]
```

### Custom Dataset

To use your own dataset:

1. **Prepare your data** in CSV or JSON format
2. **Update column names** if needed:
   ```python
   data_loader = MedicalDataLoader(
       data_path="your_dataset.csv",
       text_column="symptom_text",  # Your symptom column name
       label_column="disease_name"  # Your disease column name
   )
   ```

## 🎯 Model Training

### Training Parameters

```python
# Customize training parameters
training_results = trainer.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    output_dir="./models/custom_medical_bert",
    num_epochs=10,           # Number of training epochs
    batch_size=32,           # Batch size
    learning_rate=1e-5,      # Learning rate
    warmup_steps=1000,       # Warmup steps
    weight_decay=0.01,       # Weight decay
    class_weights=class_weights  # For imbalanced data
)
```

### Model Evaluation

```python
# Evaluate trained model
evaluation_results = trainer.evaluate(
    test_texts=test_texts,
    test_labels=test_labels,
    model_path="./models/medical_bert"
)

print(f"Accuracy: {evaluation_results['accuracy']:.3f}")
print(f"F1 Score: {evaluation_results['f1_score']:.3f}")
```

## 🔍 Available Models

### HuggingFace Models

The system supports various BERT-based models:

- **BioBERT**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **ClinicalBERT**: `emilyalsentzer/Bio_ClinicalBERT`
- **General BERT**: `bert-base-uncased`
- **Medical BERT**: `dmis-lab/biobert-base-cased-v1.1`

### Model Selection

```python
# Use different models
trainer = MedicalModelTrainer(
    model_name="emilyalsentzer/Bio_ClinicalBERT",  # Clinical BERT
    max_length=512,
    num_classes=num_classes
)
```

## 📈 Performance Optimization

### GPU Support

```python
# Enable GPU training
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

### Memory Optimization

```python
# Reduce memory usage
trainer = MedicalModelTrainer(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    max_length=256,  # Reduce sequence length
    num_classes=num_classes
)

# Use smaller batch size
training_results = trainer.train(
    # ... other parameters
    batch_size=8,  # Smaller batch size
    gradient_accumulation_steps=4  # Accumulate gradients
)
```

## 🧪 Testing and Validation

### Run Tests

```bash
# Test all components
python test_disease_prediction.py

# Test specific functionality
python -c "
from test_disease_prediction import test_data_loading
test_data_loading()
"
```

### Validation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Per-class F1 scores
- **Classification Report**: Detailed per-class metrics

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

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or sequence length
   batch_size = 8
   max_length = 256
   ```

2. **Model Not Found**
   ```python
   # Check model path
   import os
   print(os.path.exists("./models/medical_bert"))
   ```

3. **Data Loading Errors**
   ```python
   # Check file format and column names
   import pandas as pd
   data = pd.read_csv("your_data.csv")
   print(data.columns)
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python train_disease_model.py --verbose
```

## 📚 API Reference

### DiseasePredictor Class

```python
class DiseasePredictor:
    def __init__(self, model_path: str)
    def predict_disease_from_text(self, symptoms: str, return_probabilities: bool = True, top_k: int = 3) -> Dict
    def batch_predict(self, symptoms_list: List[str]) -> List[Dict]
    def get_model_info(self) -> Dict
```

### MedicalDataLoader Class

```python
class MedicalDataLoader:
    def __init__(self, data_path: str, text_column: str = "symptoms", label_column: str = "disease")
    def load_data(self) -> pd.DataFrame
    def preprocess_data(self) -> Tuple[List[str], List[str], List[int]]
    def split_data(self, texts: List[str], labels: List[int], test_size: float = 0.2, val_size: float = 0.1) -> Dict
```

### MedicalModelTrainer Class

```python
class MedicalModelTrainer:
    def __init__(self, model_name: str, max_length: int = 512, num_classes: int = None)
    def train(self, train_texts: List[str], train_labels: List[int], val_texts: List[str], val_labels: List[int], **kwargs) -> Dict
    def evaluate(self, test_texts: List[str], test_labels: List[int], model_path: Optional[str] = None) -> Dict
```

## 🤝 Contributing

To contribute to the disease prediction system:

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
