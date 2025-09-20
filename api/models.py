"""
SQLAlchemy Database Models

This module defines the database schema for the resume relevance check system:
- JobDescription: Job postings and requirements
- Resume: Candidate resumes and metadata  
- Evaluation: Scoring results and analysis
- User: Placement staff and API users

The models support JSON fields for complex data structures and maintain
relationships for efficient querying and data integrity.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class JobDescription(Base, TimestampMixin):
    """
    Job Description model for storing job postings and requirements.
    
    Supports both parsed (from uploaded files) and structured (from API) data.
    The parsed fields are populated by the async parser for uploaded files.
    """
    __tablename__ = "job_descriptions"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    jd_id = Column(String(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Basic information
    title = Column(String(200), nullable=False, index=True)
    company = Column(String(200), nullable=False, index=True)
    location = Column(String(100), index=True)
    status = Column(String(20), default="active", index=True)  # active, inactive, processing
    
    # Full text content
    description = Column(Text)  # Complete job description text
    
    # Structured requirements (JSON fields for flexibility)
    must_have_skills = Column(JSON)  # List[str]
    good_to_have_skills = Column(JSON)  # List[str]
    education_requirements = Column(JSON)  # Dict with level, field, etc.
    certifications_required = Column(JSON)  # List[str]
    experience_requirements = Column(JSON)  # Dict with years, type, etc.
    
    # File handling
    original_filename = Column(String(255))
    file_path = Column(String(500))
    
    # Metadata
    created_by = Column(String(36), ForeignKey("users.user_id"))
    parsing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    parsing_error = Column(Text)
    
    # Relationships
    creator = relationship("User", back_populates="job_descriptions")
    evaluations = relationship("Evaluation", back_populates="job_description")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_jd_company_location', 'company', 'location'),
        Index('idx_jd_status_created', 'status', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            'jd_id': self.jd_id,
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'description': self.description,
            'must_have_skills': self.must_have_skills or [],
            'good_to_have_skills': self.good_to_have_skills or [],
            'education_requirements': self.education_requirements or {},
            'certifications_required': self.certifications_required or [],
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class Resume(Base, TimestampMixin):
    """
    Resume model for storing candidate resumes and parsed data.
    
    Stores both the original file and the parsed structured data.
    Links to evaluations for scoring history.
    """
    __tablename__ = "resumes"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(String(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    
    # Candidate information
    candidate_name = Column(String(200), index=True)
    candidate_email = Column(String(200), index=True)
    candidate_phone = Column(String(50))
    
    # Processing status
    status = Column(String(20), default="uploaded", index=True)  # uploaded, processing, parsed, failed
    parsing_error = Column(Text)
    
    # Parsed resume data (JSON for flexibility)
    parsed_data = Column(JSON)  # Complete structured resume data
    skills = Column(JSON)  # List[str] - extracted skills
    experience_years = Column(Integer)  # Estimated total experience
    education_level = Column(String(50))  # Highest education level
    
    # Text content for search and similarity
    full_text = Column(Text)  # Complete resume text
    summary = Column(Text)  # Optional resume summary/objective
    
    # Metadata
    uploaded_by = Column(String(36), ForeignKey("users.user_id"))
    
    # Relationships
    uploader = relationship("User", back_populates="resumes")
    evaluations = relationship("Evaluation", back_populates="resume")
    
    # Indexes for search and filtering
    __table_args__ = (
        Index('idx_resume_candidate_name', 'candidate_name'),
        Index('idx_resume_skills', 'skills'),  # GIN index for PostgreSQL JSON search
        Index('idx_resume_status_created', 'status', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            'resume_id': self.resume_id,
            'filename': self.filename,
            'candidate_name': self.candidate_name,
            'candidate_email': self.candidate_email,
            'skills': self.skills or [],
            'experience_years': self.experience_years,
            'education_level': self.education_level,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class Evaluation(Base, TimestampMixin):
    """
    Evaluation model for storing resume-JD matching results.
    
    Contains all scoring components, feedback, and analysis results.
    Supports both completed evaluations and in-progress status tracking.
    """
    __tablename__ = "evaluations"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    resume_id = Column(String(36), ForeignKey("resumes.resume_id"), nullable=False, index=True)
    jd_id = Column(String(36), ForeignKey("job_descriptions.jd_id"), nullable=False, index=True)
    
    # Processing status
    status = Column(String(20), default="pending", index=True)  # pending, processing, completed, failed
    processing_stage = Column(String(50))  # parsing, hard_matching, semantic_matching, feedback_generation
    
    # Scoring results
    hard_score = Column(Float)  # Hard matching score (0-100)
    semantic_score = Column(Float)  # Semantic similarity score (0-100)
    final_score = Column(Integer)  # Combined final score (0-100)
    verdict = Column(String(10), index=True)  # high, medium, low
    
    # Detailed analysis (JSON fields for complex data)
    hard_score_breakdown = Column(JSON)  # Detailed hard matching analysis
    skill_matches = Column(JSON)  # Skill matching details
    missing_elements = Column(JSON)  # List of missing requirements
    top_semantic_matches = Column(JSON)  # Best semantic matches
    
    # AI-generated feedback
    feedback_suggestions = Column(JSON)  # List of improvement suggestions
    feedback_model_used = Column(String(50))  # LLM model used for feedback
    
    # Performance metrics
    processing_time_seconds = Column(Float)  # Total processing time
    embedding_model_used = Column(String(100))  # Embedding model used
    
    # Parsed data snapshots (for historical consistency)
    parsed_resume = Column(JSON)  # Resume data at time of evaluation
    parsed_jd = Column(JSON)  # JD data at time of evaluation
    
    # Error handling
    error_message = Column(Text)
    error_stage = Column(String(50))
    
    # Metadata
    evaluated_by = Column(String(36), ForeignKey("users.user_id"))
    
    # Relationships
    resume = relationship("Resume", back_populates="evaluations")
    job_description = relationship("JobDescription", back_populates="evaluations")
    evaluator = relationship("User", back_populates="evaluations")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_eval_resume_jd', 'resume_id', 'jd_id'),
        Index('idx_eval_score_verdict', 'final_score', 'verdict'),
        Index('idx_eval_status_created', 'status', 'created_at'),
        UniqueConstraint('resume_id', 'jd_id', name='uq_resume_jd_latest'),  # One latest eval per resume-JD pair
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return {
            'evaluation_id': self.evaluation_id,
            'resume_id': self.resume_id,
            'jd_id': self.jd_id,
            'status': self.status,
            'hard_score': self.hard_score,
            'semantic_score': self.semantic_score,
            'final_score': self.final_score,
            'verdict': self.verdict,
            'missing_elements': self.missing_elements,
            'feedback_suggestions': self.feedback_suggestions,
            'top_semantic_matches': self.top_semantic_matches,
            'processing_time_seconds': self.processing_time_seconds,
            'created_at': self.created_at,
            'error_message': self.error_message
        }

class User(Base, TimestampMixin):
    """
    User model for placement staff and API access management.
    
    Supports basic authentication and role-based access control.
    Tracks user activity and API usage.
    """
    __tablename__ = "users"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Authentication
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255))  # For future password auth
    api_token = Column(String(255), unique=True, index=True)  # Current token-based auth
    
    # User information
    full_name = Column(String(200))
    role = Column(String(50), default="staff", index=True)  # staff, admin, readonly
    department = Column(String(100))
    
    # Status and permissions
    is_active = Column(Boolean, default=True, index=True)
    is_admin = Column(Boolean, default=False)
    can_upload_jds = Column(Boolean, default=True)
    can_view_all_data = Column(Boolean, default=False)
    
    # Activity tracking
    last_login = Column(DateTime(timezone=True))
    login_count = Column(Integer, default=0)
    api_calls_count = Column(Integer, default=0)
    api_calls_today = Column(Integer, default=0)
    api_calls_reset_date = Column(DateTime(timezone=True))
    
    # Relationships
    job_descriptions = relationship("JobDescription", back_populates="creator")
    resumes = relationship("Resume", back_populates="uploader")
    evaluations = relationship("Evaluation", back_populates="evaluator")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_role_active', 'role', 'is_active'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses (exclude sensitive data)."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'department': self.department,
            'is_active': self.is_active,
            'last_login': self.last_login,
            'created_at': self.created_at
        }

# Additional models for future features

class EvaluationHistory(Base, TimestampMixin):
    """
    Track evaluation history and changes for audit purposes.
    Optional table for detailed audit logging.
    """
    __tablename__ = "evaluation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(36), ForeignKey("evaluations.evaluation_id"), nullable=False)
    
    # What changed
    field_name = Column(String(100), nullable=False)
    old_value = Column(JSON)
    new_value = Column(JSON)
    
    # Who made the change
    changed_by = Column(String(36), ForeignKey("users.user_id"))
    change_reason = Column(String(500))
    
    # Indexes
    __table_args__ = (
        Index('idx_eval_history_eval_id', 'evaluation_id'),
        Index('idx_eval_history_created', 'created_at'),
    )

class SystemConfig(Base, TimestampMixin):
    """
    System configuration and feature flags.
    Allows runtime configuration without code changes.
    """
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON)  # Flexible value storage
    description = Column(String(500))
    category = Column(String(50), index=True)  # scoring, features, limits, etc.
    
    is_active = Column(Boolean, default=True)
    requires_restart = Column(Boolean, default=False)
    
    # Access control
    editable_by_role = Column(String(50), default="admin")  # Which role can edit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'category': self.category,
            'is_active': self.is_active
        }
