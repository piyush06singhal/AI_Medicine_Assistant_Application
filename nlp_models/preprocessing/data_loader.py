"""
Medical Symptom Data Loader and Preprocessor
Handles loading, cleaning, and preprocessing of medical symptom datasets.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """Handles loading and preprocessing of medical symptom datasets."""
    
    def __init__(self, data_path: str, text_column: str = "symptoms", label_column: str = "disease"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset file (CSV or JSON)
            text_column: Name of the column containing symptom text
            label_column: Name of the column containing disease labels
        """
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.label_column = label_column
        self.data = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV or JSON file."""
        try:
            if self.data_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.suffix.lower() == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.data = pd.DataFrame(json_data)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            logger.info(f"Loaded dataset with {len(self.data)} samples")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize medical text.
        
        Args:
            text: Raw symptom text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_data(self) -> Tuple[List[str], List[str], List[int]]:
        """
        Preprocess the loaded data.
        
        Returns:
            Tuple of (cleaned_texts, diseases, encoded_labels)
        """
        if self.data is None:
            self.load_data()
        
        # Clean text data
        logger.info("Cleaning text data...")
        self.data[f'{self.text_column}_cleaned'] = self.data[self.text_column].apply(self.clean_text)
        
        # Remove empty texts
        self.data = self.data[self.data[f'{self.text_column}_cleaned'].str.len() > 0]
        
        # Encode labels
        logger.info("Encoding disease labels...")
        encoded_labels = self.label_encoder.fit_transform(self.data[self.label_column])
        
        # Get unique diseases for reference
        unique_diseases = self.label_encoder.classes_
        logger.info(f"Found {len(unique_diseases)} unique diseases: {list(unique_diseases)}")
        
        return (
            self.data[f'{self.text_column}_cleaned'].tolist(),
            self.data[self.label_column].tolist(),
            encoded_labels.tolist()
        )
    
    def split_data(self, texts: List[str], labels: List[int], 
                   test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            texts: List of text samples
            labels: List of encoded labels
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation (from remaining data)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, test splits
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return splits
    
    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            labels: List of encoded labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced', classes=unique_labels, y=labels
        )
        
        return dict(zip(unique_labels, class_weights))
    
    def save_processed_data(self, output_path: str, texts: List[str], 
                           diseases: List[str], labels: List[int]):
        """
        Save processed data to file.
        
        Args:
            output_path: Path to save the processed data
            texts: List of cleaned texts
            diseases: List of disease names
            labels: List of encoded labels
        """
        processed_data = pd.DataFrame({
            'symptoms': texts,
            'disease': diseases,
            'encoded_label': labels
        })
        
        output_path = Path(output_path)
        if output_path.suffix.lower() == '.csv':
            processed_data.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.json':
            processed_data.to_json(output_path, orient='records', indent=2)
        
        logger.info(f"Processed data saved to {output_path}")
    
    def get_disease_info(self) -> Dict[str, int]:
        """Get mapping of disease names to encoded labels."""
        return dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
    
    def get_label_encoder(self) -> LabelEncoder:
        """Get the fitted label encoder."""
        return self.label_encoder
