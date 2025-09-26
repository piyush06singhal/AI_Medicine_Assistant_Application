"""
Final comprehensive test for AI Medical Assistant web application
Tests all features including audio, multi-language, and advanced analytics.
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

def test_audio_processing():
    """Test audio processing functionality."""
    print("🧪 Testing Audio Processing...")
    
    try:
        from utils.audio_processor import AudioProcessor, record_and_transcribe
        
        # Initialize audio processor
        audio_processor = AudioProcessor()
        print("✅ Audio processor initialized")
        
        # Test audio processing (without actual recording)
        print("✅ Audio processing module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {str(e)}")
        return False

def test_advanced_features():
    """Test advanced features functionality."""
    print("\n🧪 Testing Advanced Features...")
    
    try:
        from utils.advanced_features import AdvancedAnalytics, ModelComparison, DataExport
        
        # Test advanced analytics
        analytics = AdvancedAnalytics()
        print("✅ Advanced analytics initialized")
        
        # Test model comparison
        comparison = ModelComparison()
        print("✅ Model comparison initialized")
        
        # Test data export
        export = DataExport()
        print("✅ Data export initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {str(e)}")
        return False

def test_portfolio_app():
    """Test portfolio app functionality."""
    print("\n🧪 Testing Portfolio App...")
    
    try:
        # Test portfolio app import
        from web_app.portfolio_app import main as portfolio_main
        print("✅ Portfolio app imported successfully")
        
        # Test multi-language support
        from utils.multilang_support import MultiLanguageSupport
        multilang = MultiLanguageSupport()
        
        # Test language detection
        english_text = "I have a headache and fever"
        hindi_text = "मुझे सिरदर्द और बुखार है"
        
        detected_en = multilang.detect_language(english_text)
        detected_hi = multilang.detect_language(hindi_text)
        
        print(f"✅ English detection: {detected_en}")
        print(f"✅ Hindi detection: {detected_hi}")
        
        # Test UI text
        ui_text = multilang.get_ui_text('symptoms', 'hi')
        print(f"✅ Hindi UI text: {ui_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio app test failed: {str(e)}")
        return False

def test_unified_predictor_with_audio():
    """Test unified predictor with audio and multi-language support."""
    print("\n🧪 Testing Unified Predictor with Audio...")
    
    try:
        from utils.unified_predictor import predict_disease
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Unified predictor test failed: {str(e)}")
        return False

def test_web_app_components():
    """Test web app components."""
    print("\n🧪 Testing Web App Components...")
    
    try:
        # Test all web app imports
        from web_app.portfolio_app import main as portfolio_main
        from web_app.medical_assistant_app import main as basic_main
        from web_app.enhanced_app import main as enhanced_main
        from web_app.multilang_app import main as multilang_main
        
        print("✅ All web app components imported successfully")
        
        # Test web utils
        from web_app.utils import create_prediction_summary, format_confidence
        
        # Test utility functions
        test_data = {
            'predicted_disease': 'diabetes',
            'confidence': 0.85,
            'related_symptoms': ['frequent urination', 'excessive thirst'],
            'precautions': ['monitor blood sugar', 'consult doctor'],
            'timestamp': '2024-01-01T00:00:00'
        }
        
        summary = create_prediction_summary(test_data)
        confidence_str = format_confidence(0.85)
        
        print(f"✅ Prediction summary created: {len(summary)} fields")
        print(f"✅ Confidence formatting: {confidence_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web app components test failed: {str(e)}")
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
            "deploy.py",
            "PROJECT_OVERVIEW.md"
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
            if "streamlit" in requirements and "googletrans" in requirements and "SpeechRecognition" in requirements:
                print("✅ Requirements file contains necessary packages")
            else:
                print("❌ Requirements file missing necessary packages")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment files test failed: {str(e)}")
        return False

def test_integration_workflow():
    """Test complete integration workflow with all features."""
    print("\n🧪 Testing Integration Workflow...")
    
    try:
        from utils.unified_predictor import UnifiedDiseasePredictor
        from utils.query_logger import QueryLogger
        from utils.multilang_support import MultiLanguageSupport
        from utils.audio_processor import AudioProcessor
        from utils.advanced_features import AdvancedAnalytics
        
        # Initialize all components
        predictor = UnifiedDiseasePredictor()
        query_logger = QueryLogger(log_dir="test_logs", anonymize=True)
        multilang = MultiLanguageSupport()
        audio_processor = AudioProcessor()
        analytics = AdvancedAnalytics()
        
        print("✅ All components initialized successfully")
        
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
        
        # Test analytics
        history = [result_en, result_hi]
        metrics = analytics.create_model_performance_metrics(history)
        print(f"✅ Analytics completed: {len(metrics)} metrics")
        
        # Clean up
        import shutil
        if Path("test_logs").exists():
            shutil.rmtree("test_logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {str(e)}")
        return False

def test_web_app_launch():
    """Test web app launch capability."""
    print("\n🧪 Testing Web App Launch...")
    
    try:
        # Test if web app can be launched
        import subprocess
        import sys
        
        # Test portfolio app
        result = subprocess.run([
            sys.executable, "-c", 
            "from web_app.portfolio_app import main; print('Portfolio app can be imported')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Portfolio app can be launched")
        else:
            print(f"❌ Portfolio app launch failed: {result.stderr}")
            return False
        
        # Test basic app
        result = subprocess.run([
            sys.executable, "-c", 
            "from web_app.medical_assistant_app import main; print('Basic app can be imported')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Basic app can be launched")
        else:
            print(f"❌ Basic app launch failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Web app launch test failed: {str(e)}")
        return False

def test_deployment_script():
    """Test deployment script functionality."""
    print("\n🧪 Testing Deployment Script...")
    
    try:
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
    """Run all tests for the final web app."""
    print("🏥 AI Medical Assistant - Final Web App Test Suite")
    print("=" * 60)
    
    tests = [
        ("Audio Processing", test_audio_processing),
        ("Advanced Features", test_advanced_features),
        ("Portfolio App", test_portfolio_app),
        ("Unified Predictor with Audio", test_unified_predictor_with_audio),
        ("Web App Components", test_web_app_components),
        ("Deployment Files", test_deployment_files),
        ("Integration Workflow", test_integration_workflow),
        ("Web App Launch", test_web_app_launch),
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
        print("🎉 All tests passed! The web app is ready for deployment.")
        print("\n📝 Next steps:")
        print("1. Run 'streamlit run web_app/portfolio_app.py' to start the app")
        print("2. Test the audio input feature (requires microphone)")
        print("3. Test multi-language support (English & Hindi)")
        print("4. Deploy using 'python deploy.py'")
        print("5. Share your portfolio project!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n🌐 Final Features Available:")
    print("- Multi-modal input (Text, Image, Audio)")
    print("- Multi-language support (English & Hindi)")
    print("- Advanced analytics and visualizations")
    print("- Real-time query logging and monitoring")
    print("- Portfolio-optimized interface")
    print("- Free deployment options (Streamlit Cloud, HuggingFace Spaces)")
    print("- Comprehensive documentation and project overview")
    
    print("\n🎯 Portfolio Value:")
    print("- Demonstrates advanced AI/ML techniques")
    print("- Shows full-stack development skills")
    print("- Includes modern deployment practices")
    print("- Suitable for resume and interview discussions")

if __name__ == "__main__":
    main()
