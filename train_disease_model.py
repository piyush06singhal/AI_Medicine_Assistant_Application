"""
Main training script for medical disease prediction model.
Trains a BERT/BioBERT model on medical symptom data.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nlp_models.preprocessing.data_loader import MedicalDataLoader
from nlp_models.training.model_trainer import MedicalModelTrainer
from nlp_models.models.disease_predictor import DiseasePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medical disease prediction model')
    parser.add_argument('--data_path', type=str, 
                       default='data/raw/medical_symptoms_dataset.csv',
                       help='Path to the medical symptoms dataset')
    parser.add_argument('--model_name', type=str, 
                       default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                       help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, 
                       default='./models/medical_bert',
                       help='Output directory for the trained model')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting medical disease prediction model training...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        data_loader = MedicalDataLoader(
            data_path=args.data_path,
            text_column="symptoms",
            label_column="disease"
        )
        
        # Load data
        data = data_loader.load_data()
        logger.info(f"Loaded {len(data)} samples")
        
        # Preprocess data
        texts, diseases, labels = data_loader.preprocess_data()
        logger.info(f"Preprocessed {len(texts)} samples")
        
        # Split data
        splits = data_loader.split_data(
            texts, labels, 
            test_size=args.test_size, 
            val_size=args.val_size
        )
        
        train_texts, train_labels = splits['train']
        val_texts, val_labels = splits['val']
        test_texts, test_labels = splits['test']
        
        logger.info(f"Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Get class weights for imbalanced data
        class_weights = data_loader.get_class_weights(train_labels)
        logger.info(f"Class weights: {class_weights}")
        
        # Step 2: Initialize trainer
        logger.info("Step 2: Initializing model trainer...")
        trainer = MedicalModelTrainer(
            model_name=args.model_name,
            max_length=args.max_length,
            num_classes=len(set(labels))
        )
        
        # Step 3: Train the model
        logger.info("Step 3: Training the model...")
        training_results = trainer.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            class_weights=class_weights
        )
        
        logger.info(f"Training completed. Results: {training_results}")
        
        # Step 4: Evaluate the model
        logger.info("Step 4: Evaluating the model...")
        evaluation_results = trainer.evaluate(
            test_texts=test_texts,
            test_labels=test_labels,
            model_path=args.output_dir
        )
        
        logger.info(f"Evaluation results: {evaluation_results}")
        
        # Step 5: Save model information
        logger.info("Step 5: Saving model information...")
        disease_mapping = data_loader.get_disease_info()
        trainer.save_model_info(args.output_dir, data_loader.get_label_encoder(), disease_mapping)
        
        # Step 6: Test the prediction function
        logger.info("Step 6: Testing prediction function...")
        predictor = DiseasePredictor(args.output_dir)
        
        # Test with sample symptoms
        test_symptoms = [
            "frequent urination, excessive thirst, fatigue",
            "headache, dizziness, chest pain",
            "wheezing, shortness of breath, coughing"
        ]
        
        for symptoms in test_symptoms:
            result = predictor.predict_disease_from_text(symptoms)
            logger.info(f"Symptoms: {symptoms}")
            logger.info(f"Predicted: {result['predicted_disease']} (confidence: {result['confidence']:.3f})")
            logger.info(f"Related symptoms: {result['related_symptoms']}")
            logger.info("---")
        
        # Save training summary
        training_summary = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_info': {
                'total_samples': len(texts),
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                'test_samples': len(test_texts),
                'num_classes': len(set(labels)),
                'class_names': data_loader.get_label_encoder().classes_.tolist()
            },
            'model_info': {
                'model_name': args.model_name,
                'max_length': args.max_length,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate
            }
        }
        
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Training summary saved to: {output_dir / 'training_summary.json'}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
