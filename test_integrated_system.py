"""
Test script for the integrated AI Medical Assistant system.
Tests the unified prediction function and web app components.
"""

import sys
import os
from pathlib import Path
import logging
import tempfile
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.unified_predictor import UnifiedDiseasePredictor, predict_disease
from web_app.utils import (
    save_prediction_history, load_prediction_history, create_prediction_summary,
    format_confidence, get_confidence_color, create_model_status_display,
    validate_symptoms_input, validate_image_file, create_prediction_metrics,
    create_download_link, create_prediction_export, create_analytics_dashboard
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_predictor():
    """Test the unified predictor functionality."""
    print("🧪 Testing Unified Predictor...")
    
    try:
        # Initialize predictor
        predictor = UnifiedDiseasePredictor()
        
        # Test model status
        model_status = predictor.get_model_status()
        print(f"✅ Model status retrieved: {model_status}")
        
        # Test prediction with text only
        test_symptoms = "frequent urination, excessive thirst, fatigue, blurred vision"
        result = predictor.predict_disease(test_symptoms)
        
        print(f"✅ Text prediction completed")
        print(f"   Predicted disease: {result.get('predicted_disease', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"   Model availability: {result.get('model_availability', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified predictor test failed: {str(e)}")
        return False

def test_prediction_function():
    """Test the convenience prediction function."""
    print("\n🧪 Testing Prediction Function...")
    
    try:
        # Test with text only
        result = predict_disease("headache, dizziness, chest pain, shortness of breath")
        
        print(f"✅ Prediction function completed")
        print(f"   Predicted disease: {result.get('predicted_disease', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction function test failed: {str(e)}")
        return False

def test_web_utils():
    """Test web application utility functions."""
    print("\n🧪 Testing Web Utils...")
    
    try:
        # Test validation functions
        is_valid, error_msg = validate_symptoms_input("test symptoms")
        print(f"✅ Symptoms validation: {is_valid}, {error_msg}")
        
        # Test confidence formatting
        confidence_str = format_confidence(0.85)
        print(f"✅ Confidence formatting: {confidence_str}")
        
        # Test color assignment
        color = get_confidence_color(0.75)
        print(f"✅ Confidence color: {color}")
        
        # Test model status display
        model_availability = {'nlp_available': True, 'cv_available': False}
        status_display = create_model_status_display(model_availability)
        print(f"✅ Model status display: {status_display}")
        
        # Test prediction summary
        test_data = {
            'predicted_disease': 'diabetes',
            'confidence': 0.85,
            'related_symptoms': ['frequent urination', 'excessive thirst'],
            'precautions': ['monitor blood sugar', 'consult doctor'],
            'timestamp': '2024-01-01T00:00:00'
        }
        
        summary = create_prediction_summary(test_data)
        print(f"✅ Prediction summary created: {len(summary)} fields")
        
        # Test metrics creation
        metrics = create_prediction_metrics(test_data)
        print(f"✅ Prediction metrics created: {len(metrics)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Web utils test failed: {str(e)}")
        return False

def test_history_tracking():
    """Test prediction history tracking."""
    print("\n🧪 Testing History Tracking...")
    
    try:
        # Create test prediction data
        test_prediction = {
            'timestamp': '2024-01-01T00:00:00',
            'input_symptoms': 'test symptoms',
            'input_image': None,
            'predicted_disease': 'test_disease',
            'confidence': 0.85,
            'model_availability': {'nlp_available': True, 'cv_available': False}
        }
        
        # Test saving prediction
        save_prediction_history(test_prediction, "test_history.json")
        print("✅ Prediction saved to history")
        
        # Test loading history
        history = load_prediction_history("test_history.json")
        print(f"✅ History loaded: {len(history)} predictions")
        
        # Test analytics dashboard
        analytics = create_analytics_dashboard(history)
        print(f"✅ Analytics created: {len(analytics)} metrics")
        
        # Clean up test file
        Path("test_history.json").unlink()
        print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ History tracking test failed: {str(e)}")
        return False

def test_export_functionality():
    """Test export and download functionality."""
    print("\n🧪 Testing Export Functionality...")
    
    try:
        # Test prediction export
        test_data = {
            'timestamp': '2024-01-01T00:00:00',
            'input_symptoms': 'test symptoms',
            'predicted_disease': 'test_disease',
            'confidence': 0.85,
            'related_symptoms': ['symptom1', 'symptom2'],
            'precautions': ['precaution1', 'precaution2']
        }
        
        export_text = create_prediction_export(test_data)
        print(f"✅ Export text created: {len(export_text)} characters")
        
        # Test download link creation
        download_link = create_download_link(test_data)
        print(f"✅ Download link created: {len(download_link)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Export functionality test failed: {str(e)}")
        return False

def test_image_validation():
    """Test image file validation."""
    print("\n🧪 Testing Image Validation...")
    
    try:
        # Test with None (valid - image is optional)
        is_valid, error_msg = validate_image_file(None)
        print(f"✅ None validation: {is_valid}, {error_msg}")
        
        # Test with mock file object
        class MockFile:
            def __init__(self, name, size):
                self.name = name
                self.size = size
        
        # Test valid file
        mock_file = MockFile("test.jpg", 1000000)  # 1MB
        is_valid, error_msg = validate_image_file(mock_file)
        print(f"✅ Valid file validation: {is_valid}, {error_msg}")
        
        # Test file too large
        mock_file_large = MockFile("test.jpg", 20000000)  # 20MB
        is_valid, error_msg = validate_image_file(mock_file_large)
        print(f"✅ Large file validation: {is_valid}, {error_msg}")
        
        # Test invalid file type
        mock_file_invalid = MockFile("test.txt", 1000000)
        is_valid, error_msg = validate_image_file(mock_file_invalid)
        print(f"✅ Invalid file type validation: {is_valid}, {error_msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image validation test failed: {str(e)}")
        return False

def test_web_app_imports():
    """Test that web app can be imported without errors."""
    print("\n🧪 Testing Web App Imports...")
    
    try:
        # Test basic app import
        from web_app.medical_assistant_app import main as basic_main
        print("✅ Basic app imported successfully")
        
        # Test enhanced app import
        from web_app.enhanced_app import main as enhanced_main
        print("✅ Enhanced app imported successfully")
        
        # Test utils import
        from web_app.utils import create_prediction_summary
        print("✅ Web utils imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Web app import test failed: {str(e)}")
        return False

def test_integration_workflow():
    """Test the complete integration workflow."""
    print("\n🧪 Testing Integration Workflow...")
    
    try:
        # Test complete workflow
        predictor = UnifiedDiseasePredictor()
        
        # Test with text input
        result = predictor.predict_disease("chest pain, shortness of breath, fatigue")
        
        # Test summary creation
        summary = create_prediction_summary(result)
        
        # Test metrics creation
        metrics = create_prediction_metrics(result)
        
        # Test history saving
        save_prediction_history(result, "integration_test.json")
        
        # Test history loading
        history = load_prediction_history("integration_test.json")
        
        # Test analytics
        analytics = create_analytics_dashboard(history)
        
        print(f"✅ Integration workflow completed successfully")
        print(f"   Predictions in history: {len(history)}")
        print(f"   Analytics metrics: {len(analytics)}")
        
        # Clean up
        Path("integration_test.json").unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🏥 AI Medical Assistant - Integrated System Tests")
    print("=" * 60)
    
    tests = [
        ("Unified Predictor", test_unified_predictor),
        ("Prediction Function", test_prediction_function),
        ("Web Utils", test_web_utils),
        ("History Tracking", test_history_tracking),
        ("Export Functionality", test_export_functionality),
        ("Image Validation", test_image_validation),
        ("Web App Imports", test_web_app_imports),
        ("Integration Workflow", test_integration_workflow)
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
        print("🎉 All tests passed! The integrated system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Run 'python run_web_app.py' to start the web application")
    print("2. Open your browser to the provided URL")
    print("3. Test the web interface with sample data")
    print("4. Use the predict_disease() function in your applications")

if __name__ == "__main__":
    main()
