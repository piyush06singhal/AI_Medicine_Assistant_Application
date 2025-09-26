"""
Data Processing Utilities for AI Medical Assistant
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing utilities for medical data."""
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not text:
            return ""
        
        # Basic text cleaning
        text = str(text).strip()
        text = text.lower()
        
        return text
    
    def process_medical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical data for analysis."""
        processed = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                processed[key] = self.clean_text(value)
            else:
                processed[key] = value
        
        return processed
