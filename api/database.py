"""
Database Configuration and Connection Management

This module handles database connectivity, session management, and provides
dependency injection for FastAPI endpoints.

Supports both PostgreSQL (production) and SQLite (development/testing).
"""

import os
from typing import Generator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import settings

# Database URL from environment or config
DATABASE_URL = settings.database_url

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for development/testing
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug  # Log SQL in debug mode
    )
else:
    # PostgreSQL configuration for production
    engine = create_engine(
        DATABASE_URL,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,  # Recycle connections every hour
        pool_pre_ping=True  # Validate connections before use
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models (imported in models.py)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI endpoints.
    
    Provides a database session that automatically commits on success
    and rolls back on exceptions.
    
    Usage in FastAPI endpoints:
        def endpoint(db: Session = Depends(get_db)):
            # Use db for queries
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    from . import models
    models.Base.metadata.create_all(bind=engine)

def drop_tables():
    """Drop all database tables (use with caution!)."""
    from . import models
    models.Base.metadata.drop_all(bind=engine)
