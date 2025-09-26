"""
Configuration settings for the AI Medical Assistant.
"""

import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "AI Medical Assistant"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Settings
    api_host: str = Field(default="localhost", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Model Settings
    model_cache_dir: str = Field(default=str(PROJECT_ROOT / "models"), env="MODEL_CACHE_DIR")
    max_model_size_mb: int = Field(default=1000, env="MAX_MODEL_SIZE_MB")
    
    # Data Settings
    data_dir: str = Field(default=str(PROJECT_ROOT / "data"), env="DATA_DIR")
    raw_data_dir: str = Field(default=str(PROJECT_ROOT / "data" / "raw"), env="RAW_DATA_DIR")
    processed_data_dir: str = Field(default=str(PROJECT_ROOT / "data" / "processed"), env="PROCESSED_DATA_DIR")
    
    # NLP Settings
    nlp_model_name: str = Field(default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", env="NLP_MODEL_NAME")
    max_text_length: int = Field(default=512, env="MAX_TEXT_LENGTH")
    
    # CV Settings
    cv_model_name: str = Field(default="microsoft/resnet-50", env="CV_MODEL_NAME")
    image_size: tuple = Field(default=(224, 224), env="IMAGE_SIZE")
    
    # HuggingFace Settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    hf_cache_dir: str = Field(default=str(PROJECT_ROOT / ".cache" / "huggingface"), env="HF_CACHE_DIR")
    
    # Database Settings (if needed)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default=str(PROJECT_ROOT / "logs" / "app.log"), env="LOG_FILE")
    
    class Config:
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.model_cache_dir, exist_ok=True)
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.raw_data_dir, exist_ok=True)
os.makedirs(settings.processed_data_dir, exist_ok=True)
os.makedirs(settings.hf_cache_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
