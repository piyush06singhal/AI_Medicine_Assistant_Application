"""
Demonstration script for medical image disease prediction.
Shows how to use the trained model for medical image analysis.
"""

import sys
from pathlib import Path
import logging
import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cv_models.models.image_predictor import MedicalImagePredictor, predict_disease_from_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_image_prediction():
    """Demonstrate medical image prediction functionality."""
    
    print("🏥 AI Medical Assistant - Medical Image Prediction Demo")
    print("=" * 70)
    
    # Sample images for testing (you can replace these with real medical images)
    sample_images = [
        "data/raw/medical_images/pneumonia/pneumonia_000.jpg",
        "data/raw/medical_images/melanoma/melanoma_000.jpg",
        "data/raw/medical_images/normal_lung/normal_lung_000.jpg",
        "data/raw/medical_images/basal_cell_carcinoma/basal_cell_carcinoma_000.jpg"
    ]
    
    # Initialize predictor (assuming model is trained)
    model_path = "./models/medical_cnn"
    
    try:
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first by running: python train_image_model.py")
            return
        
        print(f"✅ Loading model from {model_path}")
        predictor = MedicalImagePredictor(model_path)
        
        print("\n🔍 Testing Medical Image Predictions:")
        print("-" * 50)
        
        for i, image_path in enumerate(sample_images, 1):
            if not Path(image_path).exists():
                print(f"⚠️  Sample image not found: {image_path}")
                continue
            
            print(f"\n{i}. Image: {Path(image_path).name}")
            print("-" * 30)
            
            # Make prediction
            result = predictor.predict_disease_from_image(image_path)
            
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
            
            # Display class probabilities if available
            if result.get('class_probabilities'):
                print(f"📈 Class Probabilities:")
                for class_name, prob in sorted(result['class_probabilities'].items(), 
                                             key=lambda x: x[1], reverse=True)[:3]:
                    print(f"   {class_name}: {prob:.3f}")
        
        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        logger.error(f"Demo error: {str(e)}")

def interactive_prediction():
    """Interactive medical image prediction mode."""
    
    print("\n🎯 Interactive Medical Image Prediction")
    print("=" * 50)
    print("Enter the path to a medical image (or 'quit' to exit):")
    
    model_path = "./models/medical_cnn"
    
    try:
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first by running: python train_image_model.py")
            return
        
        predictor = MedicalImagePredictor(model_path)
        
        while True:
            image_path = input("\n🔍 Enter image path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not image_path:
                print("⚠️  Please enter an image path.")
                continue
            
            if not Path(image_path).exists():
                print(f"❌ Image not found: {image_path}")
                continue
            
            try:
                result = predictor.predict_disease_from_image(image_path)
                
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

def create_sample_images():
    """Create sample medical images for testing."""
    print("\n🖼️  Creating Sample Medical Images...")
    
    try:
        from data.raw.create_sample_image_dataset import create_sample_dataset
        
        # Create sample dataset
        create_sample_dataset(
            output_dir="data/raw/medical_images",
            num_samples_per_class=10  # Small number for demo
        )
        
        print("✅ Sample images created successfully!")
        print("You can now run the demo with these sample images.")
        
    except Exception as e:
        print(f"❌ Error creating sample images: {str(e)}")

def test_image_loading():
    """Test image loading and preprocessing."""
    print("\n🧪 Testing Image Loading and Preprocessing...")
    
    try:
        from cv_models.preprocessing.image_loader import MedicalImageLoader
        
        # Test with sample dataset
        data_dir = "data/raw/medical_images"
        if not Path(data_dir).exists():
            print(f"❌ Data directory not found: {data_dir}")
            print("Please create sample images first.")
            return
        
        # Initialize loader
        loader = MedicalImageLoader(data_dir=data_dir, image_size=(224, 224))
        
        # Load data
        data_df = loader.load_from_folder_structure()
        print(f"✅ Loaded {len(data_df)} images")
        
        # Preprocess images
        image_paths, labels = loader.preprocess_images(data_df)
        print(f"✅ Preprocessed {len(image_paths)} valid images")
        
        # Get class names
        class_names = loader.get_class_names()
        print(f"✅ Found classes: {class_names}")
        
        # Test data splitting
        splits = loader.split_data(image_paths, labels)
        print(f"✅ Data splits - Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
        
        print("✅ Image loading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Image loading test failed: {str(e)}")

def main():
    """Main function."""
    print("Choose an option:")
    print("1. Run demo with sample images")
    print("2. Interactive prediction mode")
    print("3. Create sample medical images")
    print("4. Test image loading and preprocessing")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        demo_image_prediction()
    elif choice == '2':
        interactive_prediction()
    elif choice == '3':
        create_sample_images()
    elif choice == '4':
        test_image_loading()
    elif choice == '5':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
