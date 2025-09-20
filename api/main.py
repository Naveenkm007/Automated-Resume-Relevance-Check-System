"""
FastAPI Main Application

This module provides the main FastAPI application with endpoints for:
- Job description upload and management
- Resume upload and processing
- Results retrieval and search
- Async job processing integration

The API integrates the complete resume relevance checking pipeline:
1. Document parsing (PDF/DOCX -> structured data)
2. Hard matching (keyword-based scoring)
3. Semantic matching (embedding-based similarity)
4. LLM feedback generation
5. Results storage and retrieval
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

# Database and models
from .database import get_db, engine
from .models import JobDescription, Resume, Evaluation, User
from . import models

# Background task processing
from .tasks import evaluate_resume_task, parse_jd_task

# Configuration
from .config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Resume Relevance Check API",
    description="AI-powered resume analysis and job matching system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

# Pydantic models for request/response
class JobDescriptionCreate(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    description: str = Field(..., description="Full job description text")
    must_have_skills: List[str] = Field(default=[], description="Required skills")
    good_to_have_skills: List[str] = Field(default=[], description="Preferred skills")
    education_requirements: Optional[Dict[str, str]] = Field(None, description="Education requirements")
    certifications_required: List[str] = Field(default=[], description="Required certifications")

class JobDescriptionResponse(BaseModel):
    jd_id: str
    title: str
    company: str
    location: Optional[str]
    created_at: datetime
    status: str

class ResumeUploadResponse(BaseModel):
    resume_id: str
    filename: str
    jd_id: Optional[str]
    status: str
    message: str

class EvaluationResult(BaseModel):
    resume_id: str
    jd_id: Optional[str]
    status: str
    hard_score: Optional[float]
    semantic_score: Optional[float]
    final_score: Optional[int]
    verdict: Optional[str]
    missing_elements: Optional[List[Dict[str, Any]]]
    feedback_suggestions: Optional[List[Dict[str, str]]]
    top_semantic_matches: Optional[List[Dict[str, Any]]]
    parsed_resume: Optional[Dict[str, Any]]
    evaluation_time: Optional[datetime]
    error_message: Optional[str]

class SearchResult(BaseModel):
    resume_id: str
    candidate_name: Optional[str]
    filename: str
    final_score: Optional[int]
    verdict: Optional[str]
    jd_title: Optional[str]
    jd_company: Optional[str]
    created_at: datetime

# Authentication helper
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
                          db: Session = Depends(get_db)) -> Optional[User]:
    """
    Basic token-based authentication.
    In production, implement proper JWT validation.
    """
    if not credentials:
        return None
    
    # Simple token lookup - replace with proper JWT validation
    user = db.query(User).filter(User.api_token == credentials.credentials).first()
    return user

# File validation helpers
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file size and extension."""
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
    
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

# API Endpoints

@app.post("/upload-jd", response_model=JobDescriptionResponse)
async def upload_job_description(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    jd_data: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Upload and process a job description.
    
    Supports two input methods:
    1. File upload (PDF/DOCX/TXT) - will be parsed automatically
    2. JSON/form data - structured job description
    
    Returns:
        JobDescriptionResponse: Created JD with unique ID
    """
    try:
        jd_id = str(uuid.uuid4())
        
        # Create initial JD record
        db_jd = JobDescription(
            jd_id=jd_id,
            title=title or "Untitled Position",
            company=company or "Unknown Company",
            location=location,
            status="processing",
            created_by=current_user.user_id if current_user else None
        )
        
        if file:
            # File upload path
            validate_file(file)
            
            # Save file temporarily
            upload_dir = Path("uploads/jds")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"{jd_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            db_jd.file_path = str(file_path)
            db_jd.original_filename = file.filename
            
            # Queue parsing task
            background_tasks.add_task(parse_jd_task, jd_id, str(file_path))
            
        elif jd_data:
            # JSON data path
            import json
            try:
                parsed_data = json.loads(jd_data)
                jd_create = JobDescriptionCreate(**parsed_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON data: {e}")
            
            # Update JD with parsed data
            db_jd.title = jd_create.title
            db_jd.company = jd_create.company
            db_jd.location = jd_create.location
            db_jd.description = jd_create.description
            db_jd.must_have_skills = jd_create.must_have_skills
            db_jd.good_to_have_skills = jd_create.good_to_have_skills
            db_jd.education_requirements = jd_create.education_requirements
            db_jd.certifications_required = jd_create.certifications_required
            db_jd.status = "active"
            
        else:
            raise HTTPException(status_code=400, detail="Either file or jd_data must be provided")
        
        db.add(db_jd)
        db.commit()
        db.refresh(db_jd)
        
        return JobDescriptionResponse(
            jd_id=db_jd.jd_id,
            title=db_jd.title,
            company=db_jd.company,
            location=db_jd.location,
            created_at=db_jd.created_at,
            status=db_jd.status
        )
        
    except Exception as e:
        logger.error(f"Error uploading JD: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload-resume", response_model=ResumeUploadResponse)
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    jd_id: Optional[str] = Form(None),
    candidate_name: Optional[str] = Form(None),
    candidate_email: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Upload resume and trigger evaluation.
    
    Args:
        file: Resume file (PDF/DOCX)
        jd_id: Optional job description ID to evaluate against
        candidate_name: Optional candidate name
        candidate_email: Optional candidate email
        
    Returns:
        ResumeUploadResponse: Upload confirmation with resume_id
    """
    try:
        validate_file(file)
        
        resume_id = str(uuid.uuid4())
        
        # Verify JD exists if provided
        if jd_id:
            db_jd = db.query(JobDescription).filter(JobDescription.jd_id == jd_id).first()
            if not db_jd:
                raise HTTPException(status_code=404, detail="Job description not found")
        
        # Save file
        upload_dir = Path("uploads/resumes")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{resume_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create resume record
        db_resume = Resume(
            resume_id=resume_id,
            filename=file.filename,
            file_path=str(file_path),
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            status="processing",
            uploaded_by=current_user.user_id if current_user else None
        )
        
        db.add(db_resume)
        db.commit()
        db.refresh(db_resume)
        
        # Queue evaluation task
        if jd_id:
            background_tasks.add_task(evaluate_resume_task, resume_id, jd_id)
            message = "Resume uploaded and evaluation started"
        else:
            message = "Resume uploaded successfully. Use /evaluate endpoint to score against specific JD."
        
        return ResumeUploadResponse(
            resume_id=resume_id,
            filename=file.filename,
            jd_id=jd_id,
            status="processing",
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/evaluate/{resume_id}")
async def evaluate_resume_against_jd(
    resume_id: str,
    jd_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Trigger evaluation of existing resume against a job description.
    
    Args:
        resume_id: ID of uploaded resume
        jd_id: ID of job description to evaluate against
        
    Returns:
        Message confirming evaluation started
    """
    try:
        # Verify resume and JD exist
        db_resume = db.query(Resume).filter(Resume.resume_id == resume_id).first()
        if not db_resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        db_jd = db.query(JobDescription).filter(JobDescription.jd_id == jd_id).first()
        if not db_jd:
            raise HTTPException(status_code=404, detail="Job description not found")
        
        # Queue evaluation task
        background_tasks.add_task(evaluate_resume_task, resume_id, jd_id)
        
        return {"message": "Evaluation started", "resume_id": resume_id, "jd_id": jd_id}
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/results/{resume_id}", response_model=EvaluationResult)
async def get_evaluation_results(
    resume_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get comprehensive evaluation results for a resume.
    
    Returns:
        EvaluationResult: Complete evaluation including scores, feedback, and parsed data
    """
    try:
        # Get resume
        db_resume = db.query(Resume).filter(Resume.resume_id == resume_id).first()
        if not db_resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get latest evaluation
        db_evaluation = db.query(Evaluation).filter(
            Evaluation.resume_id == resume_id
        ).order_by(Evaluation.created_at.desc()).first()
        
        if not db_evaluation:
            return EvaluationResult(
                resume_id=resume_id,
                jd_id=None,
                status="not_evaluated",
                hard_score=None,
                semantic_score=None,
                final_score=None,
                verdict=None,
                missing_elements=None,
                feedback_suggestions=None,
                top_semantic_matches=None,
                parsed_resume=None,
                evaluation_time=None,
                error_message="No evaluation found"
            )
        
        return EvaluationResult(
            resume_id=resume_id,
            jd_id=db_evaluation.jd_id,
            status=db_evaluation.status,
            hard_score=db_evaluation.hard_score,
            semantic_score=db_evaluation.semantic_score,
            final_score=db_evaluation.final_score,
            verdict=db_evaluation.verdict,
            missing_elements=db_evaluation.missing_elements,
            feedback_suggestions=db_evaluation.feedback_suggestions,
            top_semantic_matches=db_evaluation.top_semantic_matches,
            parsed_resume=db_evaluation.parsed_resume,
            evaluation_time=db_evaluation.created_at,
            error_message=db_evaluation.error_message
        )
        
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search", response_model=List[SearchResult])
async def search_resumes(
    role: Optional[str] = Query(None, description="Filter by job role/title"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum final score"),
    location: Optional[str] = Query(None, description="Filter by job location"),
    verdict: Optional[str] = Query(None, regex="^(high|medium|low)$", description="Filter by verdict"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Search and filter resumes based on criteria.
    
    Args:
        role: Filter by job role/title
        min_score: Minimum final score threshold
        location: Filter by job location
        verdict: Filter by evaluation verdict (high/medium/low)
        limit: Maximum results to return
        offset: Pagination offset
        
    Returns:
        List[SearchResult]: Matching resumes with scores and metadata
    """
    try:
        # Build query
        query = db.query(Resume, Evaluation, JobDescription).join(
            Evaluation, Resume.resume_id == Evaluation.resume_id, isouter=True
        ).join(
            JobDescription, Evaluation.jd_id == JobDescription.jd_id, isouter=True
        )
        
        # Apply filters
        filters = []
        
        if role:
            filters.append(JobDescription.title.ilike(f"%{role}%"))
        
        if min_score is not None:
            filters.append(Evaluation.final_score >= min_score)
        
        if location:
            filters.append(JobDescription.location.ilike(f"%{location}%"))
        
        if verdict:
            filters.append(Evaluation.verdict == verdict)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Execute query with pagination
        results = query.order_by(
            Evaluation.final_score.desc().nullslast(),
            Resume.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        # Format results
        search_results = []
        for resume, evaluation, jd in results:
            search_results.append(SearchResult(
                resume_id=resume.resume_id,
                candidate_name=resume.candidate_name,
                filename=resume.filename,
                final_score=evaluation.final_score if evaluation else None,
                verdict=evaluation.verdict if evaluation else None,
                jd_title=jd.title if jd else None,
                jd_company=jd.company if jd else None,
                created_at=resume.created_at
            ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching resumes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get system statistics."""
    try:
        stats = {
            "total_resumes": db.query(Resume).count(),
            "total_jds": db.query(JobDescription).count(),
            "total_evaluations": db.query(Evaluation).count(),
            "recent_evaluations": db.query(Evaluation).filter(
                Evaluation.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()
        }
        
        # Verdict breakdown
        verdict_counts = db.query(Evaluation.verdict, db.func.count(Evaluation.id)).group_by(
            Evaluation.verdict
        ).all()
        
        stats["verdict_breakdown"] = {verdict: count for verdict, count in verdict_counts}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
