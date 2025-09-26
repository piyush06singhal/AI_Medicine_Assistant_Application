"""
Test script for new features: logging, multi-language support, and deployment
"""

import sys
import os
from pathlib import Path
import logging
import tempfile
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.query_logger import QueryLogger, log_user_query
from utils.multilang_support import MultiLanguageSupport, detect_and_translate_symptoms
from utils.unified_predictor import predict_disease

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_logging():
    """Test the query logging system."""
    print("🧪 Testing Query Logging System...")
    
    try:
        # Initialize query logger
        query_logger = QueryLogger(log_dir="test_logs", anonymize=True)
        
        # Test logging a query
        query_id = log_user_query(
            symptoms_text="frequent urination, excessive thirst, fatigue",
            image_path=None,
            predicted_disease="diabetes",
            confidence=0.85,
            model_availability={'nlp_available': True, 'cv_available': False},
            prediction_source="NLP only",
            processing_time=1.2,
            language="en"
        )
        
        print(f"✅ Query logged with ID: {query_id}")
        
        # Test getting statistics
        stats = query_logger.get_query_stats(days=1)
        print(f"✅ Query statistics: {stats}")
        
        # Test export functionality
        export_path = query_logger.export_query_data(format='json')
        if export_path:
            print(f"✅ Query data exported to: {export_path}")
        
        # Clean up
        import shutil
        if Path("test_logs").exists():
            shutil.rmtree("test_logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Query logging test failed: {str(e)}")
        return False

def test_multilang_support():
    """Test multi-language support functionality."""
    print("\n🧪 Testing Multi-language Support...")
    
    try:
        # Initialize multi-language support
        multilang = MultiLanguageSupport()
        
        # Test language detection
        english_text = "I have a headache and fever"
        hindi_text = "मुझे सिरदर्द और बुखार है"
        
        detected_en = multilang.detect_language(english_text)
        detected_hi = multilang.detect_language(hindi_text)
        
        print(f"✅ English detection: {detected_en}")
        print(f"✅ Hindi detection: {detected_hi}")
        
        # Test translation
        translated = multilang.translate_text(hindi_text, 'en')
        print(f"✅ Translation: {translated}")
        
        # Test UI text
        ui_text = multilang.get_ui_text('symptoms', 'hi')
        print(f"✅ Hindi UI text: {ui_text}")
        
        # Test mixed language detection
        mixed_text = "मुझे headache है और fever भी है"
        detected_mixed = multilang.detect_language(mixed_text)
        print(f"✅ Mixed language detection: {detected_mixed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-language support test failed: {str(e)}")
        return False

def test_unified_predictor_with_logging():
    """Test unified predictor with logging and multi-language support."""
    print("\n🧪 Testing Unified Predictor with New Features...")
    
    try:
        # Test English prediction
        result_en = predict_disease(
            symptoms="frequent urination, excessive thirst, fatigue",
            language="en"
        )
        
        print(f"✅ English prediction: {result_en.get('predicted_disease', 'Unknown')}")
        print(f"✅ Query ID: {result_en.get('query_id', 'None')}")
        print(f"✅ Processing time: {result_en.get('processing_time', 0):.3f}s")
        
        # Test Hindi prediction
        result_hi = predict_disease(
            symptoms="बार-बार पेशाब आना, अत्यधिक प्यास, थकान",
            language="hi"
        )
        
        print(f"✅ Hindi prediction: {result_hi.get('predicted_disease', 'Unknown')}")
        print(f"✅ Detected language: {result_hi.get('detected_language', 'Unknown')}")
        print(f"✅ Translated symptoms: {result_hi.get('translated_symptoms', 'None')}")
        
        # Test mixed language
        result_mixed = predict_disease(
            symptoms="मुझे headache है और fever भी है",
            language="en"
        )
        
        print(f"✅ Mixed language prediction: {result_mixed.get('predicted_disease', 'Unknown')}")
        print(f"✅ Detected language: {result_mixed.get('detected_language', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified predictor test failed: {str(e)}")
        return False

def test_web_app_imports():
    """Test that web apps can be imported with new features."""
    print("\n🧪 Testing Web App Imports...")
    
    try:
        # Test multi-language app import
        from web_app.multilang_app import main as multilang_main
        print("✅ Multi-language app imported successfully")
        
        # Test basic app import
        from web_app.medical_assistant_app import main as basic_main
        print("✅ Basic app imported successfully")
        
        # Test enhanced app import
        from web_app.enhanced_app import main as enhanced_main
        print("✅ Enhanced app imported successfully")
        
        # Test web utils import
        from web_app.utils import create_prediction_summary
        print("✅ Web utils imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Web app import test failed: {str(e)}")
        return False

def test_deployment_files():
    """Test deployment file creation and validation."""
    print("\n🧪 Testing Deployment Files...")
    
    try:
        # Check if deployment files exist
        deployment_files = [
            "requirements_deployment.txt",
            "packages.txt",
            "README_DEPLOYMENT.md",
            "deploy.py"
        ]
        
        for file in deployment_files:
            if Path(file).exists():
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                return False
        
        # Test requirements file content
        with open("requirements_deployment.txt", "r") as f:
            requirements = f.read()
            if "streamlit" in requirements and "googletrans" in requirements:
                print("✅ Requirements file contains necessary packages")
            else:
                print("❌ Requirements file missing necessary packages")
                return False
        
        # Test packages file content
        with open("packages.txt", "r") as f:
            packages = f.read()
            if "libgl1-mesa-glx" in packages:
                print("✅ Packages file contains necessary system packages")
            else:
                print("❌ Packages file missing necessary system packages")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment files test failed: {str(e)}")
        return False

def test_integration_workflow():
    """Test complete integration workflow with new features."""
    print("\n🧪 Testing Integration Workflow...")
    
    try:
        # Test complete workflow
        from utils.unified_predictor import UnifiedDiseasePredictor
        from utils.query_logger import QueryLogger
        from utils.multilang_support import MultiLanguageSupport
        
        # Initialize components
        predictor = UnifiedDiseasePredictor()
        query_logger = QueryLogger(log_dir="test_logs", anonymize=True)
        multilang = MultiLanguageSupport()
        
        # Test workflow with English
        result_en = predictor.predict_disease(
            symptoms="chest pain, shortness of breath",
            language="en"
        )
        
        print(f"✅ English workflow completed")
        print(f"   Predicted disease: {result_en.get('predicted_disease', 'Unknown')}")
        print(f"   Query ID: {result_en.get('query_id', 'None')}")
        
        # Test workflow with Hindi
        result_hi = predictor.predict_disease(
            symptoms="सीने में दर्द, सांस लेने में तकलीफ",
            language="hi"
        )
        
        print(f"✅ Hindi workflow completed")
        print(f"   Predicted disease: {result_hi.get('predicted_disease', 'Unknown')}")
        print(f"   Detected language: {result_hi.get('detected_language', 'Unknown')}")
        
        # Test query statistics
        stats = query_logger.get_query_stats(days=1)
        print(f"✅ Query statistics: {stats.get('total_queries', 0)} queries")
        
        # Clean up
        import shutil
        if Path("test_logs").exists():
            shutil.rmtree("test_logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {str(e)}")
        return False

def test_deployment_script():
    """Test deployment script functionality."""
    print("\n🧪 Testing Deployment Script...")
    
    try:
        # Test deployment script import
        from deploy import create_deployment_files, prepare_for_streamlit_cloud
        
        # Test deployment file creation
        create_deployment_files()
        
        # Check if files were created
        if Path(".streamlit/config.toml").exists():
            print("✅ Streamlit config created")
        else:
            print("❌ Streamlit config not created")
            return False
        
        if Path("app.py").exists():
            print("✅ HuggingFace app.py created")
        else:
            print("❌ HuggingFace app.py not created")
            return False
        
        # Clean up created files
        if Path(".streamlit").exists():
            import shutil
            shutil.rmtree(".streamlit")
        if Path("app.py").exists():
            os.remove("app.py")
        if Path(".gitignore").exists():
            os.remove(".gitignore")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment script test failed: {str(e)}")
        return False

def main():
    """Run all tests for new features."""
    print("🏥 AI Medical Assistant - New Features Test Suite")
    print("=" * 60)
    
    tests = [
        ("Query Logging", test_query_logging),
        ("Multi-language Support", test_multilang_support),
        ("Unified Predictor with Logging", test_unified_predictor_with_logging),
        ("Web App Imports", test_web_app_imports),
        ("Deployment Files", test_deployment_files),
        ("Integration Workflow", test_integration_workflow),
        ("Deployment Script", test_deployment_script)
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
        print("🎉 All tests passed! New features are working correctly.")
        print("\n📝 Next steps:")
        print("1. Run 'python deploy.py' to prepare for deployment")
        print("2. Deploy to Streamlit Cloud or HuggingFace Spaces")
        print("3. Test the multi-language web app")
        print("4. Monitor query logs for analytics")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n🌐 New Features Available:")
    print("- Multi-language support (English & Hindi)")
    print("- Comprehensive query logging")
    print("- Deployment scripts for free platforms")
    print("- Enhanced web app with analytics")
    print("- Query statistics and export functionality")

if __name__ == "__main__":
    main()
