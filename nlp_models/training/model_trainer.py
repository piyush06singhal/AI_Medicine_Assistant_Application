"""
Medical Disease Prediction Model Trainer
Trains BERT/BioBERT models for disease prediction from symptoms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataset(Dataset):
    """PyTorch Dataset for medical symptom data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MedicalBERTClassifier(nn.Module):
    """BERT-based classifier for medical disease prediction."""
    
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.3):
        super(MedicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MedicalModelTrainer:
    """Trainer for medical disease prediction models."""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 max_length: int = 512, num_classes: int = None):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model name (BERT or BioBERT)
            max_length: Maximum sequence length
            num_classes: Number of disease classes
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_classes = num_classes
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        
    def setup_tokenizer(self):
        """Setup the tokenizer."""
        logger.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_model(self, num_classes: int):
        """Setup the model."""
        logger.info(f"Loading model {self.model_name}")
        self.model = MedicalBERTClassifier(
            model_name=self.model_name,
            num_classes=num_classes
        )
        self.num_classes = num_classes
    
    def create_datasets(self, train_texts: List[str], train_labels: List[int],
                       val_texts: List[str], val_labels: List[int],
                       test_texts: List[str], test_labels: List[int]) -> Tuple[Dataset, Dataset, Dataset]:
        """Create PyTorch datasets."""
        train_dataset = MedicalDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = MedicalDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        test_dataset = MedicalDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              output_dir: str = "./models/medical_bert",
              num_epochs: int = 5,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              class_weights: Optional[Dict[int, float]] = None) -> Dict:
        """
        Train the medical disease prediction model.
        
        Args:
            train_texts: Training text data
            train_labels: Training labels
            val_texts: Validation text data
            val_labels: Validation labels
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            class_weights: Class weights for imbalanced data
            
        Returns:
            Training results dictionary
        """
        # Setup tokenizer and model
        if self.tokenizer is None:
            self.setup_tokenizer()
        
        if self.model is None:
            self.setup_model(num_classes=len(set(train_labels)))
        
        # Create datasets
        train_dataset, val_dataset, _ = self.create_datasets(
            train_texts, train_labels, val_texts, val_labels, [], []
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training results
        results = {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'model_path': output_dir
        }
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        return results
    
    def evaluate(self, test_texts: List[str], test_labels: List[int],
                model_path: Optional[str] = None) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            test_texts: Test text data
            test_labels: Test labels
            model_path: Path to the trained model
            
        Returns:
            Evaluation results
        """
        if model_path:
            # Load the trained model
            self.model = MedicalBERTClassifier(
                model_name=self.model_name,
                num_classes=self.num_classes
            )
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
            self.model.eval()
        
        # Create test dataset
        test_dataset = MedicalDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for item in test_dataset:
                input_ids = item['input_ids'].unsqueeze(0)
                attention_mask = item['attention_mask'].unsqueeze(0)
                
                outputs = self.model(input_ids, attention_mask)
                pred = torch.argmax(outputs, dim=1).item()
                predictions.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Classification report
        class_report = classification_report(
            test_labels, predictions, output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results
    
    def save_model_info(self, output_dir: str, label_encoder, disease_info: Dict):
        """Save model information and metadata."""
        model_info = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'disease_mapping': disease_info,
            'class_names': label_encoder.classes_.tolist()
        }
        
        with open(f"{output_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {output_dir}/model_info.json")
