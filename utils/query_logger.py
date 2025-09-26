"""
Query Logging System for AI Medical Assistant
Logs user queries and predictions for model improvement and analytics.
"""

import json
import logging
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
import os
from dataclasses import dataclass, asdict
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryLog:
    """Data class for query logging."""
    query_id: str
    timestamp: str
    user_id: str
    session_id: str
    language: str
    symptoms_text: str
    image_uploaded: bool
    image_hash: Optional[str]
    image_size: Optional[int]
    predicted_disease: str
    confidence: float
    model_availability: Dict[str, bool]
    prediction_source: str
    processing_time: float
    user_feedback: Optional[str] = None
    anonymized: bool = True

class QueryLogger:
    """Logger for user queries and predictions."""
    
    def __init__(self, log_dir: str = "logs", anonymize: bool = True):
        """
        Initialize the query logger.
        
        Args:
            log_dir: Directory to store log files
            anonymize: Whether to anonymize sensitive data
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.anonymize = anonymize
        
        # Create subdirectories
        (self.log_dir / "queries").mkdir(exist_ok=True)
        (self.log_dir / "analytics").mkdir(exist_ok=True)
        (self.log_dir / "exports").mkdir(exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Setup file logging for queries."""
        log_file = self.log_dir / "query_logs.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        query_logger = logging.getLogger('query_logger')
        query_logger.addHandler(file_handler)
        query_logger.setLevel(logging.INFO)
        
        self.query_logger = query_logger
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize text by removing potential PII."""
        if not self.anonymize:
            return text
        
        # Simple anonymization - remove common PII patterns
        import re
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove potential names (simple heuristic)
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        return text
    
    def _generate_image_hash(self, image_path: str) -> str:
        """Generate hash for image file."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            logger.error(f"Error generating image hash: {e}")
            return "unknown"
    
    def _get_image_size(self, image_path: str) -> int:
        """Get image file size in bytes."""
        try:
            return os.path.getsize(image_path)
        except Exception as e:
            logger.error(f"Error getting image size: {e}")
            return 0
    
    def log_query(self, 
                  symptoms_text: str,
                  image_path: Optional[str] = None,
                  predicted_disease: str = "Unknown",
                  confidence: float = 0.0,
                  model_availability: Dict[str, bool] = None,
                  prediction_source: str = "Unknown",
                  processing_time: float = 0.0,
                  language: str = "en",
                  user_id: str = None,
                  session_id: str = None) -> str:
        """
        Log a user query and prediction.
        
        Args:
            symptoms_text: User's symptom description
            image_path: Path to uploaded image (optional)
            predicted_disease: Predicted disease
            confidence: Prediction confidence
            model_availability: Available models
            prediction_source: Source of prediction
            processing_time: Time taken for prediction
            language: Language of input
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Query ID
        """
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        # Generate user and session IDs if not provided
        if not user_id:
            user_id = f"user_{hash(symptoms_text) % 10000}"
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Anonymize symptoms text
        anonymized_symptoms = self._anonymize_text(symptoms_text)
        
        # Process image information
        image_uploaded = image_path is not None
        image_hash = None
        image_size = None
        
        if image_path and os.path.exists(image_path):
            image_hash = self._generate_image_hash(image_path)
            image_size = self._get_image_size(image_path)
        
        # Create query log
        query_log = QueryLog(
            query_id=query_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            language=language,
            symptoms_text=anonymized_symptoms,
            image_uploaded=image_uploaded,
            image_hash=image_hash,
            image_size=image_size,
            predicted_disease=predicted_disease,
            confidence=confidence,
            model_availability=model_availability or {},
            prediction_source=prediction_source,
            processing_time=processing_time,
            anonymized=self.anonymize
        )
        
        # Save to JSON file
        self._save_query_log(query_log)
        
        # Log to file
        self.query_logger.info(f"Query logged: {query_id} - {predicted_disease} ({confidence:.3f})")
        
        return query_id
    
    def _save_query_log(self, query_log: QueryLog):
        """Save query log to JSON file."""
        try:
            # Save individual query
            query_file = self.log_dir / "queries" / f"{query_log.query_id}.json"
            with open(query_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(query_log), f, indent=2, ensure_ascii=False)
            
            # Append to daily log
            daily_file = self.log_dir / f"queries_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            with open(daily_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(query_log), ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving query log: {e}")
    
    def get_query_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get query statistics for the last N days."""
        try:
            stats = {
                'total_queries': 0,
                'queries_by_language': {},
                'queries_by_disease': {},
                'queries_by_confidence': {'high': 0, 'medium': 0, 'low': 0},
                'image_upload_rate': 0.0,
                'model_usage': {'nlp_only': 0, 'cv_only': 0, 'both': 0, 'none': 0},
                'avg_processing_time': 0.0,
                'daily_queries': {}
            }
            
            # Load daily logs
            for i in range(days):
                date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                date = date.replace(day=date.day - i)
                daily_file = self.log_dir / f"queries_{date.strftime('%Y-%m-%d')}.jsonl"
                
                if daily_file.exists():
                    daily_count = 0
                    with open(daily_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                query_data = json.loads(line)
                                daily_count += 1
                                
                                # Update statistics
                                stats['total_queries'] += 1
                                
                                # Language stats
                                lang = query_data.get('language', 'en')
                                stats['queries_by_language'][lang] = stats['queries_by_language'].get(lang, 0) + 1
                                
                                # Disease stats
                                disease = query_data.get('predicted_disease', 'Unknown')
                                stats['queries_by_disease'][disease] = stats['queries_by_disease'].get(disease, 0) + 1
                                
                                # Confidence stats
                                confidence = query_data.get('confidence', 0.0)
                                if confidence >= 0.7:
                                    stats['queries_by_confidence']['high'] += 1
                                elif confidence >= 0.4:
                                    stats['queries_by_confidence']['medium'] += 1
                                else:
                                    stats['queries_by_confidence']['low'] += 1
                                
                                # Image upload stats
                                if query_data.get('image_uploaded', False):
                                    stats['image_upload_rate'] += 1
                                
                                # Model usage stats
                                model_avail = query_data.get('model_availability', {})
                                nlp_avail = model_avail.get('nlp_available', False)
                                cv_avail = model_avail.get('cv_available', False)
                                
                                if nlp_avail and cv_avail:
                                    stats['model_usage']['both'] += 1
                                elif nlp_avail:
                                    stats['model_usage']['nlp_only'] += 1
                                elif cv_avail:
                                    stats['model_usage']['cv_only'] += 1
                                else:
                                    stats['model_usage']['none'] += 1
                                
                                # Processing time
                                processing_time = query_data.get('processing_time', 0.0)
                                stats['avg_processing_time'] += processing_time
                    
                    stats['daily_queries'][date.strftime('%Y-%m-%d')] = daily_count
            
            # Calculate averages
            if stats['total_queries'] > 0:
                stats['image_upload_rate'] = stats['image_upload_rate'] / stats['total_queries']
                stats['avg_processing_time'] = stats['avg_processing_time'] / stats['total_queries']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            return {}
    
    def export_query_data(self, 
                         start_date: str = None, 
                         end_date: str = None,
                         format: str = 'csv') -> str:
        """
        Export query data for analysis.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to exported file
        """
        try:
            # Load query data
            queries = []
            
            # Determine date range
            if start_date and end_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                date_range = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                             for i in range((end_dt - start_dt).days + 1)]
            else:
                # Last 30 days
                date_range = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                             for i in range(30)]
            
            # Load data
            for date_str in date_range:
                daily_file = self.log_dir / f"queries_{date_str}.jsonl"
                if daily_file.exists():
                    with open(daily_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                queries.append(json.loads(line))
            
            if not queries:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(queries)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"query_export_{timestamp}.{format}"
            export_path = self.log_dir / "exports" / filename
            
            # Export based on format
            if format == 'csv':
                df.to_csv(export_path, index=False, encoding='utf-8')
            elif format == 'json':
                df.to_json(export_path, orient='records', indent=2, force_ascii=False)
            elif format == 'excel':
                df.to_excel(export_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Query data exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting query data: {e}")
            return None
    
    def cleanup_old_logs(self, days_to_keep: int = 90):
        """Clean up old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.log_dir.glob("queries_*.jsonl"):
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")

# Global logger instance
query_logger = QueryLogger()

def log_user_query(symptoms_text: str, 
                  image_path: Optional[str] = None,
                  predicted_disease: str = "Unknown",
                  confidence: float = 0.0,
                  model_availability: Dict[str, bool] = None,
                  prediction_source: str = "Unknown",
                  processing_time: float = 0.0,
                  language: str = "en") -> str:
    """
    Convenience function to log user queries.
    
    Args:
        symptoms_text: User's symptom description
        image_path: Path to uploaded image (optional)
        predicted_disease: Predicted disease
        confidence: Prediction confidence
        model_availability: Available models
        prediction_source: Source of prediction
        processing_time: Time taken for prediction
        language: Language of input
        
    Returns:
        Query ID
    """
    return query_logger.log_query(
        symptoms_text=symptoms_text,
        image_path=image_path,
        predicted_disease=predicted_disease,
        confidence=confidence,
        model_availability=model_availability,
        prediction_source=prediction_source,
        processing_time=processing_time,
        language=language
    )
