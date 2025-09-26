"""
Main training script for medical image classification model.
Trains CNN models for disease prediction from medical images.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cv_models.preprocessing.image_loader import MedicalImageLoader
from cv_models.models.medical_cnn import create_medical_model, get_available_models
from cv_models.training.image_trainer import MedicalImageTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medical image classification model')
    parser.add_argument('--data_dir', type=str, 
                       default='data/raw/medical_images',
                       help='Directory containing the medical image dataset')
    parser.add_argument('--model_type', type=str, 
                       default='resnet50',
                       help='Type of model to train')
    parser.add_argument('--output_dir', type=str, 
                       default='./models/medical_cnn',
                       help='Output directory for the trained model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone for fine-tuning')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--data_format', type=str, default='folder',
                       choices=['folder', 'csv'],
                       help='Data format (folder structure or CSV)')
    parser.add_argument('--csv_path', type=str,
                       help='Path to CSV file (if data_format is csv)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting medical image classification model training...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        image_loader = MedicalImageLoader(
            data_dir=args.data_dir,
            image_size=tuple(args.image_size)
        )
        
        # Load data based on format
        if args.data_format == 'folder':
            data_df = image_loader.load_from_folder_structure()
        elif args.data_format == 'csv':
            if not args.csv_path:
                raise ValueError("CSV path must be provided when data_format is 'csv'")
            data_df = image_loader.load_from_csv(args.csv_path)
        else:
            raise ValueError(f"Unsupported data format: {args.data_format}")
        
        logger.info(f"Loaded {len(data_df)} images")
        
        # Preprocess images
        image_paths, labels = image_loader.preprocess_images(data_df)
        logger.info(f"Preprocessed {len(image_paths)} valid images")
        
        # Split data
        splits = image_loader.split_data(
            image_paths, labels,
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        train_paths, train_labels = splits['train']
        val_paths, val_labels = splits['val']
        test_paths, test_labels = splits['test']
        
        logger.info(f"Data splits - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Get class information
        class_names = image_loader.get_class_names()
        num_classes = len(class_names)
        logger.info(f"Classes: {class_names}")
        
        # Get class weights for imbalanced data
        class_weights = image_loader.get_class_weights(train_labels)
        logger.info(f"Class weights: {class_weights}")
        
        # Step 2: Create datasets and dataloaders
        logger.info("Step 2: Creating datasets and dataloaders...")
        datasets = image_loader.create_datasets(splits)
        dataloaders = image_loader.create_dataloaders(
            datasets, 
            batch_size=args.batch_size,
            num_workers=4
        )
        
        # Step 3: Create model
        logger.info("Step 3: Creating model...")
        model = create_medical_model(
            model_type=args.model_type,
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
        
        logger.info(f"Created model: {model.__class__.__name__}")
        
        # Step 4: Initialize trainer
        logger.info("Step 4: Initializing trainer...")
        trainer = MedicalImageTrainer(
            model=model,
            device='auto',
            class_names=class_names
        )
        
        # Step 5: Train the model
        logger.info("Step 5: Training the model...")
        training_results = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            class_weights=class_weights,
            save_best=True,
            save_dir=args.output_dir
        )
        
        logger.info(f"Training completed. Results: {training_results}")
        
        # Step 6: Evaluate the model
        logger.info("Step 6: Evaluating the model...")
        evaluation_results = trainer.evaluate(
            test_loader=dataloaders['test']
        )
        
        logger.info(f"Evaluation results: {evaluation_results}")
        
        # Step 7: Save model information
        logger.info("Step 7: Saving model information...")
        model_info = {
            'model_type': args.model_type,
            'num_classes': num_classes,
            'class_names': class_names,
            'image_size': args.image_size,
            'pretrained': args.pretrained,
            'freeze_backbone': args.freeze_backbone,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save training summary
        training_summary = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_info': {
                'total_samples': len(image_paths),
                'train_samples': len(train_paths),
                'val_samples': len(val_paths),
                'test_samples': len(test_paths),
                'num_classes': num_classes,
                'class_names': class_names
            },
            'model_info': {
                'model_type': args.model_type,
                'image_size': args.image_size,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'scheduler': args.scheduler
            }
        }
        
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Training summary saved to: {output_dir / 'training_summary.json'}")
        
        # Step 8: Test prediction function
        logger.info("Step 8: Testing prediction function...")
        from cv_models.models.image_predictor import MedicalImagePredictor
        
        predictor = MedicalImagePredictor(args.output_dir)
        
        # Test with a sample image if available
        if test_paths:
            sample_image = test_paths[0]
            result = predictor.predict_disease_from_image(sample_image)
            logger.info(f"Sample prediction for {sample_image}:")
            logger.info(f"  Predicted: {result['predicted_disease']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info(f"  Related symptoms: {result['related_symptoms'][:3]}")
            logger.info(f"  Precautions: {result['precautions'][:2]}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def list_available_models():
    """List available model types."""
    models = get_available_models()
    print("Available model types:")
    for model_type, variants in models.items():
        print(f"  {model_type}: {', '.join(variants)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--list-models':
        list_available_models()
    else:
        main()
