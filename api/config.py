"""
Application Configuration

Centralized configuration management using Pydantic Settings.
Supports environment variables and .env files.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Basic app configuration
    app_name: str = "Resume Relevance Check API"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Database configuration
    database_url: str = Field(
        default="sqlite:///./resume_checker.db",
        env="DATABASE_URL"
    )
    
    # Redis configuration for Celery
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    
    # Embedding configuration
    use_openai_embeddings: bool = Field(default=False, env="USE_OPENAI_EMBEDDINGS")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    
    # File upload limits
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Worker configuration
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
        env="CELERY_RESULT_BACKEND"
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_BROKER_URL"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Storage paths
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")
    temp_directory: str = Field(default="./tmp", env="TEMP_DIRECTORY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
