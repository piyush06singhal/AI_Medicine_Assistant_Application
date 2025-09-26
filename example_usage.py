"""
Example usage of the Medical Disease Prediction System.
Demonstrates the complete workflow from data loading to prediction.
"""

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nlp_models.preprocessing.data_loader import MedicalDataLoader
from nlp_models.training.model_trainer import MedicalModelTrainer
from nlp_models.models.disease_predictor import DiseasePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_complete_workflow():
    """Demonstrate the complete workflow from data loading to prediction."""
    
    print("🏥 Medical Disease Prediction System - Complete Workflow Example")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    print("\n📊 Step 1: Loading and Preprocessing Data")
    print("-" * 50)
    
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
    
    # Split data
    splits = data_loader.split_data(texts, labels, test_size=0.2, val_size=0.1)
    train_texts, train_labels = splits['train']
    val_texts, val_labels = splits['val']
    test_texts, test_labels = splits['test']
    
    print(f"✅ Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Get unique diseases
    unique_diseases = set(diseases)
    print(f"✅ Found {len(unique_diseases)} unique diseases: {list(unique_diseases)}")
    
    # Step 2: Initialize and train model (quick demo)
    print("\n🤖 Step 2: Model Training (Demo Mode)")
    print("-" * 50)
    
    # For demo purposes, we'll use a smaller subset
    demo_train_texts = train_texts[:10]  # Use only 10 samples for demo
    demo_train_labels = train_labels[:10]
    demo_val_texts = val_texts[:5]
    demo_val_labels = val_labels[:5]
    
    print(f"📝 Demo training with {len(demo_train_texts)} samples")
    
    # Initialize trainer
    trainer = MedicalModelTrainer(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_length=128,  # Smaller for demo
        num_classes=len(unique_diseases)
    )
    
    print("✅ Model trainer initialized")
    print("ℹ️  Note: Full training would require more data and time")
    
    # Step 3: Demonstrate prediction function
    print("\n🔍 Step 3: Disease Prediction Demo")
    print("-" * 50)
    
    # Create a mock predictor for demonstration
    class MockDiseasePredictor:
        def __init__(self):
            self.disease_mapping = {
                'diabetes': {
                    'symptoms': ['frequent urination', 'excessive thirst', 'fatigue', 'blurred vision'],
                    'precautions': ['Monitor blood sugar', 'Maintain healthy diet', 'Exercise regularly']
                },
                'hypertension': {
                    'symptoms': ['headache', 'dizziness', 'chest pain', 'shortness of breath'],
                    'precautions': ['Reduce sodium intake', 'Exercise regularly', 'Manage stress']
                },
                'asthma': {
                    'symptoms': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing'],
                    'precautions': ['Avoid triggers', 'Use inhaler as prescribed', 'Monitor symptoms']
                }
            }
        
        def predict_disease_from_text(self, symptoms: str):
            # Simple keyword-based prediction for demo
            symptoms_lower = symptoms.lower()
            
            # Simple keyword matching
            if any(word in symptoms_lower for word in ['urination', 'thirst', 'glucose', 'sugar']):
                disease = 'diabetes'
                confidence = 0.85
            elif any(word in symptoms_lower for word in ['headache', 'dizziness', 'pressure', 'blood']):
                disease = 'hypertension'
                confidence = 0.80
            elif any(word in symptoms_lower for word in ['wheezing', 'breathing', 'chest', 'cough']):
                disease = 'asthma'
                confidence = 0.75
            else:
                disease = 'unknown'
                confidence = 0.50
            
            return {
                'predicted_disease': disease,
                'confidence': confidence,
                'related_symptoms': self.disease_mapping.get(disease, {}).get('symptoms', []),
                'precautions': self.disease_mapping.get(disease, {}).get('precautions', []),
                'input_symptoms': symptoms
            }
    
    # Test predictions
    predictor = MockDiseasePredictor()
    
    test_symptoms = [
        "frequent urination, excessive thirst, fatigue, blurred vision",
        "headache, dizziness, chest pain, shortness of breath",
        "wheezing, shortness of breath, chest tightness, coughing",
        "nausea, vomiting, diarrhea, abdominal pain"
    ]
    
    print("🔍 Testing Disease Predictions:")
    for i, symptoms in enumerate(test_symptoms, 1):
        result = predictor.predict_disease_from_text(symptoms)
        
        print(f"\n{i}. Symptoms: {symptoms}")
        print(f"   🎯 Predicted Disease: {result['predicted_disease']}")
        print(f"   📊 Confidence: {result['confidence']:.2f}")
        print(f"   🔗 Related Symptoms: {', '.join(result['related_symptoms'][:3])}")
        print(f"   ⚠️  Precautions: {', '.join(result['precautions'][:2])}")
    
    # Step 4: Show how to use the real prediction function
    print("\n💡 Step 4: Using the Real Prediction Function")
    print("-" * 50)
    
    print("""
    # To use the real prediction function after training:
    
    from nlp_models.models.disease_predictor import predict_disease_from_text
    
    # Simple usage
    symptoms = "frequent urination, excessive thirst, fatigue"
    result = predict_disease_from_text(symptoms, model_path="./models/medical_bert")
    
    print(f"Predicted Disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Related Symptoms: {result['related_symptoms']}")
    print(f"Precautions: {result['precautions']}")
    
    # Advanced usage with custom predictor
    from nlp_models.models.disease_predictor import DiseasePredictor
    
    predictor = DiseasePredictor("./models/medical_bert")
    result = predictor.predict_disease_from_text(
        symptoms,
        return_probabilities=True,
        top_k=3
    )
    """)
    
    print("\n✅ Complete workflow demonstration finished!")
    print("\n📝 Next Steps:")
    print("1. Run 'python train_disease_model.py' to train the full model")
    print("2. Run 'python demo_disease_prediction.py' to test predictions")
    print("3. Use the predict_disease_from_text() function in your applications")

def example_custom_dataset():
    """Example of using a custom dataset."""
    
    print("\n📊 Custom Dataset Example")
    print("-" * 30)
    
    # Create a custom dataset
    custom_data = [
        {"symptoms": "chest pain, shortness of breath, sweating", "disease": "heart_attack"},
        {"symptoms": "fever, cough, body aches, fatigue", "disease": "flu"},
        {"symptoms": "rash, itching, swelling, difficulty breathing", "disease": "allergic_reaction"},
        {"symptoms": "severe headache, neck stiffness, fever", "disease": "meningitis"},
        {"symptoms": "abdominal pain, nausea, vomiting, fever", "disease": "appendicitis"}
    ]
    
    # Save custom dataset
    import pandas as pd
    custom_df = pd.DataFrame(custom_data)
    custom_df.to_csv("data/raw/custom_symptoms_dataset.csv", index=False)
    
    print("✅ Created custom dataset with 5 samples")
    
    # Load custom dataset
    custom_loader = MedicalDataLoader(
        data_path="data/raw/custom_symptoms_dataset.csv",
        text_column="symptoms",
        label_column="disease"
    )
    
    data = custom_loader.load_data()
    texts, diseases, labels = custom_loader.preprocess_data()
    
    print(f"✅ Loaded custom dataset: {len(texts)} samples")
    print(f"✅ Diseases: {set(diseases)}")
    
    # Clean up
    Path("data/raw/custom_symptoms_dataset.csv").unlink()
    print("✅ Cleaned up temporary files")

def main():
    """Main function."""
    print("Choose an example:")
    print("1. Complete workflow demonstration")
    print("2. Custom dataset example")
    print("3. Both examples")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        example_complete_workflow()
    elif choice == '2':
        example_custom_dataset()
    elif choice == '3':
        example_complete_workflow()
        example_custom_dataset()
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
