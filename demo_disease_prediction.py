"""
Demonstration script for medical disease prediction.
Shows how to use the trained model for disease prediction.
"""

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nlp_models.models.disease_predictor import DiseasePredictor, predict_disease_from_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_disease_prediction():
    """Demonstrate disease prediction functionality."""
    
    print("🏥 AI Medical Assistant - Disease Prediction Demo")
    print("=" * 60)
    
    # Sample symptoms for testing
    test_symptoms = [
        "frequent urination, excessive thirst, fatigue, blurred vision",
        "headache, dizziness, chest pain, shortness of breath",
        "wheezing, shortness of breath, chest tightness, coughing",
        "severe headache, nausea, sensitivity to light and sound",
        "cough with phlegm, fever, chills, chest pain",
        "nausea, vomiting, diarrhea, abdominal pain"
    ]
    
    # Initialize predictor (assuming model is trained)
    model_path = "./models/medical_bert"
    
    try:
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first by running: python train_disease_model.py")
            return
        
        print(f"✅ Loading model from {model_path}")
        predictor = DiseasePredictor(model_path)
        
        print("\n🔍 Testing Disease Predictions:")
        print("-" * 40)
        
        for i, symptoms in enumerate(test_symptoms, 1):
            print(f"\n{i}. Symptoms: {symptoms}")
            print("-" * 30)
            
            # Make prediction
            result = predictor.predict_disease_from_text(symptoms)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display results
            print(f"🎯 Predicted Disease: {result['predicted_disease']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            
            if result['top_predictions']:
                print(f"📋 Top Predictions:")
                for pred in result['top_predictions'][:3]:
                    print(f"   {pred['rank']}. {pred['disease']} ({pred['confidence']:.3f})")
            
            print(f"🔗 Related Symptoms: {', '.join(result['related_symptoms'][:3])}")
            print(f"⚠️  Precautions: {', '.join(result['precautions'][:2])}")
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        logger.error(f"Demo error: {str(e)}")

def interactive_prediction():
    """Interactive disease prediction mode."""
    
    print("\n🎯 Interactive Disease Prediction")
    print("=" * 40)
    print("Enter your symptoms (or 'quit' to exit):")
    
    model_path = "./models/medical_bert"
    
    try:
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first by running: python train_disease_model.py")
            return
        
        predictor = DiseasePredictor(model_path)
        
        while True:
            symptoms = input("\n🔍 Enter symptoms: ").strip()
            
            if symptoms.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not symptoms:
                print("⚠️  Please enter some symptoms.")
                continue
            
            try:
                result = predictor.predict_disease_from_text(symptoms)
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"\n🎯 Predicted Disease: {result['predicted_disease']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"🔗 Related Symptoms: {', '.join(result['related_symptoms'][:3])}")
                print(f"⚠️  Precautions: {', '.join(result['precautions'][:2])}")
                
                if result['top_predictions']:
                    print(f"\n📋 Other possibilities:")
                    for pred in result['top_predictions'][1:3]:
                        print(f"   • {pred['disease']} ({pred['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Prediction error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main function."""
    print("Choose an option:")
    print("1. Run demo with sample symptoms")
    print("2. Interactive prediction mode")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        demo_disease_prediction()
    elif choice == '2':
        interactive_prediction()
    elif choice == '3':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
