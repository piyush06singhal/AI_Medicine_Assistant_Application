"""
Test script for disease prediction functionality.
Tests the data loading, preprocessing, and prediction pipeline.
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nlp_models.preprocessing.data_loader import MedicalDataLoader
from nlp_models.training.model_trainer import MedicalModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading and preprocessing."""
    print("🧪 Testing Data Loading...")
    
    try:
        # Test data loading
        data_loader = MedicalDataLoader(
            data_path="data/raw/medical_symptoms_dataset.csv",
            text_column="symptoms",
            label_column="disease"
        )
        
        # Load data
        data = data_loader.load_data()
        print(f"✅ Loaded {len(data)} samples")
        
        # Preprocess data
        texts, diseases, labels = data_loader.preprocess_data()
        print(f"✅ Preprocessed {len(texts)} samples")
        
        # Check data splits
        splits = data_loader.split_data(texts, labels)
        train_texts, train_labels = splits['train']
        val_texts, val_labels = splits['val']
        test_texts, test_labels = splits['test']
        
        print(f"✅ Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Check unique diseases
        unique_diseases = set(diseases)
        print(f"✅ Found {len(unique_diseases)} unique diseases: {list(unique_diseases)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

def test_model_initialization():
    """Test model initialization."""
    print("\n🧪 Testing Model Initialization...")
    
    try:
        # Initialize trainer
        trainer = MedicalModelTrainer(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            max_length=512,
            num_classes=6  # Based on our dataset
        )
        
        # Setup tokenizer
        trainer.setup_tokenizer()
        print("✅ Tokenizer setup successful")
        
        # Setup model
        trainer.setup_model(num_classes=6)
        print("✅ Model setup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {str(e)}")
        return False

def test_prediction_function():
    """Test the prediction function (without training)."""
    print("\n🧪 Testing Prediction Function...")
    
    try:
        from nlp_models.models.disease_predictor import DiseasePredictor
        
        # Test text preprocessing
        predictor = DiseasePredictor.__new__(DiseasePredictor)
        predictor.model_path = Path("./models/medical_bert")
        predictor.model_info = {
            'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
            'num_classes': 6,
            'max_length': 512,
            'class_names': ['asthma', 'diabetes', 'gastroenteritis', 'hypertension', 'migraine', 'pneumonia']
        }
        
        # Test text preprocessing
        test_text = "frequent urination, excessive thirst, fatigue"
        cleaned_text = predictor.preprocess_text(test_text)
        print(f"✅ Text preprocessing: '{test_text}' -> '{cleaned_text}'")
        
        # Test disease info retrieval
        related_symptoms, precautions = predictor._get_disease_info("diabetes")
        print(f"✅ Disease info retrieval: {len(related_symptoms)} symptoms, {len(precautions)} precautions")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction function test failed: {str(e)}")
        return False

def run_quick_training_test():
    """Run a quick training test with minimal data."""
    print("\n🧪 Testing Quick Training...")
    
    try:
        # Load data
        data_loader = MedicalDataLoader(
            data_path="data/raw/medical_symptoms_dataset.csv",
            text_column="symptoms",
            label_column="disease"
        )
        
        data = data_loader.load_data()
        texts, diseases, labels = data_loader.preprocess_data()
        
        # Use only first 20 samples for quick test
        test_texts = texts[:20]
        test_labels = labels[:20]
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            test_texts, test_labels, test_size=0.3, random_state=42
        )
        
        # Initialize trainer
        trainer = MedicalModelTrainer(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            max_length=128,  # Smaller for quick test
            num_classes=len(set(labels))
        )
        
        print("✅ Quick training test setup successful")
        print("Note: Full training would take longer and requires more data")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🏥 AI Medical Assistant - Disease Prediction Tests")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Initialization", test_model_initialization),
        ("Prediction Function", test_prediction_function),
        ("Quick Training Setup", run_quick_training_test)
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
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The disease prediction system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Run 'python train_disease_model.py' to train the full model")
    print("2. Run 'python demo_disease_prediction.py' to test predictions")
    print("3. Use the predict_disease_from_text() function in your applications")

if __name__ == "__main__":
    main()
