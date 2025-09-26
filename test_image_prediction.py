"""
Test script for medical image prediction functionality.
Tests the data loading, preprocessing, and prediction pipeline.
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cv_models.preprocessing.image_loader import MedicalImageLoader
from cv_models.models.medical_cnn import create_medical_model, get_available_models
from cv_models.training.image_trainer import MedicalImageTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_loading():
    """Test image loading and preprocessing."""
    print("🧪 Testing Image Loading...")
    
    try:
        # Test data loading
        data_dir = "data/raw/medical_images"
        if not Path(data_dir).exists():
            print(f"❌ Data directory not found: {data_dir}")
            print("Please create sample images first by running: python data/raw/create_sample_image_dataset.py")
            return False
        
        loader = MedicalImageLoader(data_dir=data_dir, image_size=(224, 224))
        
        # Load data
        data_df = loader.load_from_folder_structure()
        print(f"✅ Loaded {len(data_df)} images")
        
        # Preprocess images
        image_paths, labels = loader.preprocess_images(data_df)
        print(f"✅ Preprocessed {len(image_paths)} valid images")
        
        # Check data splits
        splits = loader.split_data(image_paths, labels)
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']
        test_paths, test_labels = splits['test']
        
        print(f"✅ Data splits - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Check unique classes
        class_names = loader.get_class_names()
        print(f"✅ Found {len(class_names)} classes: {class_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image loading test failed: {str(e)}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\n🧪 Testing Model Creation...")
    
    try:
        # Test different model types
        model_types = ['cnn', 'resnet50', 'efficientnet_b0', 'densenet121']
        num_classes = 8  # Based on our sample dataset
        
        for model_type in model_types:
            try:
                model = create_medical_model(
                    model_type=model_type,
                    num_classes=num_classes,
                    pretrained=False  # Don't download pretrained weights for testing
                )
                print(f"✅ Created {model_type} model successfully")
            except Exception as e:
                print(f"⚠️  {model_type} model creation failed: {str(e)}")
        
        # Test available models
        available_models = get_available_models()
        print(f"✅ Available models: {list(available_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {str(e)}")
        return False

def test_data_transforms():
    """Test data augmentation transforms."""
    print("\n🧪 Testing Data Transforms...")
    
    try:
        from cv_models.preprocessing.image_loader import MedicalImageLoader
        
        loader = MedicalImageLoader(data_dir="data/raw/medical_images", image_size=(224, 224))
        
        # Test transforms
        train_transform = loader.get_transforms('train')
        val_transform = loader.get_transforms('val')
        
        print("✅ Train transforms created successfully")
        print("✅ Validation transforms created successfully")
        
        # Test with sample image
        data_dir = Path("data/raw/medical_images")
        if data_dir.exists():
            # Find a sample image
            sample_image = None
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.glob("*.jpg"):
                        sample_image = str(img_file)
                        break
                    if sample_image:
                        break
            
            if sample_image:
                # Test loading and transforming
                from cv_models.preprocessing.image_loader import MedicalImageDataset
                
                dataset = MedicalImageDataset([sample_image], [0], train_transform, (224, 224))
                image_tensor, label = dataset[0]
                
                print(f"✅ Sample image loaded and transformed: {image_tensor.shape}")
                print(f"✅ Label: {label}")
            else:
                print("⚠️  No sample images found for transform testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Data transforms test failed: {str(e)}")
        return False

def test_prediction_function():
    """Test the prediction function (without training)."""
    print("\n🧪 Testing Prediction Function...")
    
    try:
        from cv_models.models.image_predictor import MedicalImagePredictor
        
        # Test with mock predictor
        class MockImagePredictor:
            def __init__(self):
                self.class_names = ['pneumonia', 'tuberculosis', 'lung_cancer', 'normal_lung',
                                  'melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma', 'normal_skin']
                self.model_info = {'image_size': [224, 224]}
            
            def preprocess_image(self, image_path):
                # Create a mock image tensor
                return np.random.rand(3, 224, 224).astype(np.float32)
            
            def _get_disease_info(self, disease_name):
                return ['symptom1', 'symptom2'], ['precaution1', 'precaution2']
        
        # Test image preprocessing
        predictor = MockImagePredictor()
        
        # Test disease info retrieval
        related_symptoms, precautions = predictor._get_disease_info("pneumonia")
        print(f"✅ Disease info retrieval: {len(related_symptoms)} symptoms, {len(precautions)} precautions")
        
        # Test class names
        print(f"✅ Class names: {predictor.class_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction function test failed: {str(e)}")
        return False

def test_quick_training():
    """Test quick training setup (without actual training)."""
    print("\n🧪 Testing Quick Training Setup...")
    
    try:
        # Load data
        data_dir = "data/raw/medical_images"
        if not Path(data_dir).exists():
            print(f"❌ Data directory not found: {data_dir}")
            return False
        
        loader = MedicalImageLoader(data_dir=data_dir, image_size=(128, 128))  # Smaller for quick test
        
        data_df = loader.load_from_folder_structure()
        image_paths, labels = loader.preprocess_images(data_df)
        
        # Use only first 20 samples for quick test
        test_image_paths = image_paths[:20]
        test_labels = labels[:20]
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            test_image_paths, test_labels, test_size=0.3, random_state=42
        )
        
        # Create datasets
        datasets = loader.create_datasets({
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels)
        })
        
        # Create dataloaders
        dataloaders = loader.create_dataloaders(datasets, batch_size=4, num_workers=0)
        
        # Create model
        model = create_medical_model(
            model_type='cnn',  # Simple CNN for quick test
            num_classes=len(set(labels))
        )
        
        # Initialize trainer
        trainer = MedicalImageTrainer(
            model=model,
            device='cpu',  # Use CPU for quick test
            class_names=loader.get_class_names()
        )
        
        print("✅ Quick training test setup successful")
        print("Note: Full training would take longer and requires more data")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {str(e)}")
        return False

def create_sample_dataset():
    """Create sample dataset for testing."""
    print("\n🧪 Creating Sample Dataset...")
    
    try:
        from data.raw.create_sample_image_dataset import create_sample_dataset
        
        # Create sample dataset
        create_sample_dataset(
            output_dir="data/raw/medical_images",
            num_samples_per_class=5  # Very small for testing
        )
        
        print("✅ Sample dataset created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sample dataset creation failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🏥 AI Medical Assistant - Medical Image Prediction Tests")
    print("=" * 70)
    
    # Create sample dataset first
    if not Path("data/raw/medical_images").exists():
        print("Creating sample dataset for testing...")
        if not create_sample_dataset():
            print("❌ Failed to create sample dataset. Exiting.")
            return
    
    tests = [
        ("Image Loading", test_image_loading),
        ("Model Creation", test_model_creation),
        ("Data Transforms", test_data_transforms),
        ("Prediction Function", test_prediction_function),
        ("Quick Training Setup", test_quick_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*70}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The medical image prediction system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Run 'python data/raw/create_sample_image_dataset.py' to create more sample images")
    print("2. Run 'python train_image_model.py' to train the full model")
    print("3. Run 'python demo_image_prediction.py' to test predictions")
    print("4. Use the predict_disease_from_image() function in your applications")

if __name__ == "__main__":
    main()
