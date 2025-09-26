# Medical Image Disease Prediction System - Usage Guide

This guide explains how to use the AI Medical Assistant's medical image prediction system built with CNN models and transfer learning.

## 🏗️ System Overview

The medical image prediction system consists of:
- **Image Loader**: Handles medical image datasets (X-ray, skin lesions, etc.)
- **CNN Models**: Custom CNN and transfer learning models (ResNet, EfficientNet, DenseNet, ViT)
- **Image Trainer**: Training and evaluation utilities for medical image classification
- **Image Predictor**: Provides predictions with related symptoms and precautions
- **Sample Dataset**: Synthetic medical images for testing

## 📁 File Structure

```
cv_models/
├── preprocessing/
│   └── image_loader.py          # Image loading and preprocessing
├── models/
│   ├── medical_cnn.py           # CNN model architectures
│   └── image_predictor.py       # Image prediction and inference
└── training/
    └── image_trainer.py         # Training and evaluation utilities

data/raw/
├── medical_images/              # Sample medical image dataset
│   ├── pneumonia/               # X-ray images
│   ├── tuberculosis/            # X-ray images
│   ├── lung_cancer/             # X-ray images
│   ├── normal_lung/             # X-ray images
│   ├── melanoma/                # Skin lesion images
│   ├── basal_cell_carcinoma/    # Skin lesion images
│   ├── squamous_cell_carcinoma/ # Skin lesion images
│   ├── normal_skin/             # Skin lesion images
│   └── dataset.csv              # Dataset metadata
└── create_sample_image_dataset.py # Sample dataset generator

# Main scripts
train_image_model.py             # Training script
demo_image_prediction.py         # Demonstration script
test_image_prediction.py         # Test script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Additional dependencies for medical images
pip install opencv-python albumentations pydicom
```

### 2. Create Sample Dataset

```bash
# Create sample medical images for testing
python data/raw/create_sample_image_dataset.py
```

### 3. Test the System

```bash
# Run tests to verify everything works
python test_image_prediction.py
```

### 4. Train the Model

```bash
# Train the medical image classification model
python train_image_model.py

# With custom parameters
python train_image_model.py \
    --data_dir data/raw/medical_images \
    --model_type resnet50 \
    --output_dir ./models/medical_cnn \
    --num_epochs 10 \
    --batch_size 32
```

### 5. Test Predictions

```bash
# Run interactive demo
python demo_image_prediction.py
```

## 🔧 Usage Examples

### Basic Medical Image Prediction

```python
from cv_models.models.image_predictor import predict_disease_from_image

# Simple prediction
image_path = "path/to/medical_image.jpg"
result = predict_disease_from_image(image_path)

print(f"Predicted Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Related Symptoms: {result['related_symptoms']}")
print(f"Precautions: {result['precautions']}")
```

### Advanced Usage with Custom Model

```python
from cv_models.models.image_predictor import MedicalImagePredictor

# Initialize predictor with custom model
predictor = MedicalImagePredictor("./models/medical_cnn")

# Make prediction
result = predictor.predict_disease_from_image(
    "path/to/medical_image.jpg",
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
# Predict for multiple images
image_paths = [
    "image1.jpg",
    "image2.jpg", 
    "image3.jpg"
]

results = predictor.batch_predict(image_paths)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['predicted_disease']} ({result['confidence']:.3f})")
```

## 📊 Data Format

### Folder Structure Format

```
medical_images/
├── pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── tuberculosis/
│   ├── image1.jpg
│   └── ...
└── normal_lung/
    ├── image1.jpg
    └── ...
```

### CSV Format

```csv
image_path,class_name,disease_type
pneumonia/image1.jpg,pneumonia,xray
melanoma/image1.jpg,melanoma,skin
normal_lung/image1.jpg,normal_lung,xray
```

### Supported Image Formats

- **Standard**: JPG, JPEG, PNG, BMP, TIFF
- **Medical**: DICOM (.dcm), NIfTI (.nii, .nii.gz)
- **Other**: Any format supported by OpenCV or PIL

## 🎯 Model Training

### Available Models

#### CNN Models
- **Custom CNN**: `cnn` - Custom convolutional neural network
- **ResNet**: `resnet18`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **Vision Transformer**: `vit_base_patch16_224`, `vit_large_patch16_224`

### Training Parameters

```python
# Customize training parameters
python train_image_model.py \
    --data_dir data/raw/medical_images \
    --model_type resnet50 \
    --output_dir ./models/medical_cnn \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --pretrained \
    --freeze_backbone
```

### Training with Custom Dataset

```python
from cv_models.preprocessing.image_loader import MedicalImageLoader
from cv_models.models.medical_cnn import create_medical_model
from cv_models.training.image_trainer import MedicalImageTrainer

# Load your dataset
loader = MedicalImageLoader(data_dir="your_dataset", image_size=(224, 224))
data_df = loader.load_from_folder_structure()
image_paths, labels = loader.preprocess_images(data_df)

# Split data
splits = loader.split_data(image_paths, labels)
datasets = loader.create_datasets(splits)
dataloaders = loader.create_dataloaders(datasets, batch_size=32)

# Create model
model = create_medical_model(
    model_type='resnet50',
    num_classes=len(loader.get_class_names()),
    pretrained=True
)

# Train model
trainer = MedicalImageTrainer(model=model, class_names=loader.get_class_names())
results = trainer.train(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    num_epochs=10
)
```

## 🔍 Model Evaluation

### Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Per-class F1 scores
- **Confusion Matrix**: Visual representation of predictions
- **Classification Report**: Detailed per-class metrics

### Evaluation Code

```python
# Evaluate trained model
evaluation_results = trainer.evaluate(test_loader=dataloaders['test'])

print(f"Test Accuracy: {evaluation_results['test_accuracy']:.2f}%")
print(f"Precision: {evaluation_results['precision']:.4f}")
print(f"Recall: {evaluation_results['recall']:.4f}")
print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
```

## 🖼️ Image Preprocessing

### Data Augmentation

The system includes comprehensive data augmentation:

```python
# Training augmentations
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Validation augmentations (no augmentation)
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### Custom Preprocessing

```python
# Custom image preprocessing
def preprocess_medical_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply medical-specific preprocessing
    # (e.g., contrast enhancement, noise reduction)
    
    # Resize and normalize
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    
    return image
```

## 🏥 Medical Image Types

### X-ray Images
- **Pneumonia**: Chest X-rays with lung infections
- **Tuberculosis**: Chest X-rays with TB lesions
- **Lung Cancer**: Chest X-rays with tumors
- **Normal Lung**: Healthy chest X-rays

### Skin Lesion Images
- **Melanoma**: Malignant skin lesions
- **Basal Cell Carcinoma**: Common skin cancer
- **Squamous Cell Carcinoma**: Another type of skin cancer
- **Normal Skin**: Healthy skin

### DICOM Support

```python
# Load DICOM images
import pydicom

def load_dicom_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array
    
    # Normalize to 0-255
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    return image
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
trainer = MedicalImageTrainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Use smaller batch size
dataloaders = loader.create_dataloaders(
    datasets, 
    batch_size=16,  # Smaller batch size
    num_workers=2   # Fewer workers
)
```

### Model Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🧪 Testing and Validation

### Run Tests

```bash
# Test all components
python test_image_prediction.py

# Test specific functionality
python -c "
from test_image_prediction import test_image_loading
test_image_loading()
"
```

### Validation Pipeline

```python
# Validate model performance
def validate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
```

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This system is for research and educational purposes only.**
- Not intended for clinical use or medical diagnosis
- Always consult qualified healthcare professionals
- Use only for research and educational purposes

### Data Privacy

- Ensure medical images are properly anonymized
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
   # Reduce batch size or image size
   batch_size = 16
   image_size = (128, 128)
   ```

2. **Model Not Found**
   ```python
   # Check model path
   import os
   print(os.path.exists("./models/medical_cnn"))
   ```

3. **Image Loading Errors**
   ```python
   # Check image format and path
   import cv2
   image = cv2.imread("image_path.jpg")
   print(image is not None)
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python train_image_model.py --verbose
```

## 📚 API Reference

### MedicalImagePredictor Class

```python
class MedicalImagePredictor:
    def __init__(self, model_path: str, device: str = 'auto')
    def predict_disease_from_image(self, image_path: str, return_probabilities: bool = True, top_k: int = 3) -> Dict
    def batch_predict(self, image_paths: List[str]) -> List[Dict]
    def get_model_info(self) -> Dict
```

### MedicalImageLoader Class

```python
class MedicalImageLoader:
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224))
    def load_from_folder_structure(self, class_folders: Optional[List[str]] = None) -> pd.DataFrame
    def load_from_csv(self, csv_path: str, image_path_col: str = 'image_path', label_col: str = 'label') -> pd.DataFrame
    def preprocess_images(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]
    def split_data(self, image_paths: List[str], labels: List[int]) -> Dict
    def get_transforms(self, phase: str = 'train') -> A.Compose
```

### MedicalImageTrainer Class

```python
class MedicalImageTrainer:
    def __init__(self, model: nn.Module, device: str = 'auto', class_names: Optional[List[str]] = None)
    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> Dict[str, Any]
    def evaluate(self, test_loader: DataLoader, model_path: Optional[str] = None) -> Dict[str, Any]
    def save_model(self, path: str)
    def load_model(self, path: str)
```

## 🤝 Contributing

To contribute to the medical image prediction system:

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
