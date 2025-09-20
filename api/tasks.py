"""
Background Task Processing

This module implements async task processing using Celery for heavy operations:
- Resume parsing and analysis
- Job description processing  
- Semantic similarity computation
- LLM feedback generation

Tasks are processed asynchronously to keep API responses fast while handling
computationally expensive operations in the background.
"""

import os
import sys
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery import Celery
from sqlalchemy.orm import Session

# Import our modules
from resume_parser.extract import extract_text_from_file
from resume_parser.cleaner import normalize_text
from resume_parser.ner import extract_entities
from scoring.hard_match import compute_keyword_score
from semantic.combined_score import compute_combined_score
from semantic.feedback import generate_feedback

from .database import SessionLocal
from .models import JobDescription, Resume, Evaluation
from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "resume_checker_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["api.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

def get_db_session() -> Session:
    """Get database session for tasks."""
    return SessionLocal()

@celery_app.task(bind=True, name="parse_jd_task")
def parse_jd_task(self, jd_id: str, file_path: str) -> Dict[str, Any]:
    """
    Parse uploaded job description file and extract structured data.
    
    Args:
        jd_id: Job description ID
        file_path: Path to uploaded file
        
    Returns:
        Dict with parsing results and status
    """
    logger.info(f"Starting JD parsing task for {jd_id}")
    start_time = time.time()
    
    db = get_db_session()
    try:
        # Update status to processing
        db_jd = db.query(JobDescription).filter(JobDescription.jd_id == jd_id).first()
        if not db_jd:
            raise ValueError(f"Job description {jd_id} not found")
        
        db_jd.parsing_status = "processing"
        db.commit()
        
        # Extract text from file
        logger.info(f"Extracting text from {file_path}")
        raw_text = extract_text_from_file(file_path)
        
        if not raw_text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Clean and normalize text
        sections = normalize_text(raw_text)
        
        # Extract structured information using NER
        entities = extract_entities(sections.get('full_text', raw_text))
        
        # Update JD with parsed data
        db_jd.description = sections.get('full_text', raw_text)
        
        # Extract skills from parsed data or use simple extraction
        skills_text = sections.get('skills', '')
        if skills_text:
            # Simple skill extraction from skills section
            skills = [skill.strip() for skill in skills_text.replace(',', '\n').split('\n') if skill.strip()]
            db_jd.must_have_skills = skills[:10]  # First 10 as must-have
            db_jd.good_to_have_skills = skills[10:20]  # Next 10 as good-to-have
        
        # Extract education requirements
        education_text = sections.get('education', '')
        if education_text:
            # Simple heuristic for education requirements
            education_req = {}
            if 'bachelor' in education_text.lower():
                education_req['level'] = 'bachelor'
            elif 'master' in education_text.lower():
                education_req['level'] = 'master'
            elif 'phd' in education_text.lower() or 'doctorate' in education_text.lower():
                education_req['level'] = 'phd'
            
            if education_req:
                db_jd.education_requirements = education_req
        
        # Mark as completed
        db_jd.parsing_status = "completed"
        db_jd.status = "active"
        
        db.commit()
        
        processing_time = time.time() - start_time
        logger.info(f"JD parsing completed for {jd_id} in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "jd_id": jd_id,
            "processing_time": processing_time,
            "sections_found": list(sections.keys()),
            "text_length": len(raw_text)
        }
        
    except Exception as e:
        logger.error(f"JD parsing failed for {jd_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update status to failed
        if 'db_jd' in locals():
            db_jd.parsing_status = "failed"
            db_jd.parsing_error = str(e)
            db.commit()
        
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
    finally:
        db.close()

@celery_app.task(bind=True, name="evaluate_resume_task")  
def evaluate_resume_task(self, resume_id: str, jd_id: str) -> Dict[str, Any]:
    """
    Complete resume evaluation pipeline.
    
    Performs:
    1. Resume parsing and text extraction
    2. Hard matching (keyword-based scoring)
    3. Semantic similarity computation
    4. Combined scoring
    5. LLM feedback generation
    6. Results storage
    
    Args:
        resume_id: Resume ID to evaluate
        jd_id: Job description ID to evaluate against
        
    Returns:
        Dict with evaluation results
    """
    logger.info(f"Starting evaluation task for resume {resume_id} against JD {jd_id}")
    start_time = time.time()
    
    db = get_db_session()
    try:
        # Get resume and JD from database
        db_resume = db.query(Resume).filter(Resume.resume_id == resume_id).first()
        if not db_resume:
            raise ValueError(f"Resume {resume_id} not found")
        
        db_jd = db.query(JobDescription).filter(JobDescription.jd_id == jd_id).first()
        if not db_jd:
            raise ValueError(f"Job description {jd_id} not found")
        
        # Create or get evaluation record
        db_evaluation = db.query(Evaluation).filter(
            Evaluation.resume_id == resume_id,
            Evaluation.jd_id == jd_id
        ).first()
        
        if not db_evaluation:
            db_evaluation = Evaluation(
                resume_id=resume_id,
                jd_id=jd_id,
                status="processing"
            )
            db.add(db_evaluation)
        else:
            db_evaluation.status = "processing"
        
        db.commit()
        
        # STEP 1: Parse resume if not already done
        logger.info("Step 1: Parsing resume")
        db_evaluation.processing_stage = "parsing"
        db.commit()
        
        if not db_resume.parsed_data:
            resume_text = extract_text_from_file(db_resume.file_path)
            sections = normalize_text(resume_text)
            entities = extract_entities(sections.get('full_text', resume_text))
            
            # Update resume with parsed data
            db_resume.parsed_data = entities
            db_resume.full_text = sections.get('full_text', resume_text)
            db_resume.skills = entities.get('skills', [])
            db_resume.status = "parsed"
            
            # Estimate experience years
            experience = entities.get('experience', [])
            db_resume.experience_years = len(experience) if experience else 0
            
            # Extract education level
            education = entities.get('education', [])
            if education:
                degree = education[0].get('degree', '').lower()
                if 'master' in degree or 'mba' in degree:
                    db_resume.education_level = 'masters'
                elif 'bachelor' in degree:
                    db_resume.education_level = 'bachelors'
                elif 'phd' in degree or 'doctorate' in degree:
                    db_resume.education_level = 'phd'
                else:
                    db_resume.education_level = 'other'
        
        resume_struct = db_resume.parsed_data
        
        # STEP 2: Prepare JD structure
        logger.info("Step 2: Preparing JD structure")
        jd_struct = {
            'title': db_jd.title,
            'must_have_skills': db_jd.must_have_skills or [],
            'good_to_have_skills': db_jd.good_to_have_skills or [],
            'education_requirements': db_jd.education_requirements or {},
            'certifications_required': db_jd.certifications_required or [],
            'full_text': db_jd.description or ''
        }
        
        # STEP 3: Hard matching
        logger.info("Step 3: Computing hard match score")
        db_evaluation.processing_stage = "hard_matching"
        db.commit()
        
        hard_result = compute_keyword_score(resume_struct, jd_struct)
        db_evaluation.hard_score = hard_result['raw_score']
        db_evaluation.hard_score_breakdown = hard_result['breakdown']
        db_evaluation.skill_matches = hard_result['skill_matches']
        
        # STEP 4: Combined scoring (includes semantic)
        logger.info("Step 4: Computing combined score")
        db_evaluation.processing_stage = "semantic_matching"
        db.commit()
        
        combined_result = compute_combined_score(resume_struct, jd_struct, hard_result)
        
        db_evaluation.semantic_score = combined_result['semantic_score']
        db_evaluation.final_score = combined_result['final_score']
        db_evaluation.verdict = combined_result['verdict']
        db_evaluation.missing_elements = combined_result['missing_elements']
        db_evaluation.top_semantic_matches = combined_result['top_semantic_matches']
        
        # STEP 5: Generate feedback
        logger.info("Step 5: Generating LLM feedback")
        db_evaluation.processing_stage = "feedback_generation"
        db.commit()
        
        try:
            feedback = generate_feedback(
                resume_struct, 
                jd_struct, 
                combined_result['score_breakdown'],
                num_suggestions=3
            )
            db_evaluation.feedback_suggestions = feedback
            db_evaluation.feedback_model_used = settings.llm_model
        except Exception as e:
            logger.warning(f"Feedback generation failed: {e}")
            # Continue without feedback - not critical
            db_evaluation.feedback_suggestions = []
        
        # STEP 6: Store snapshots and finalize
        logger.info("Step 6: Finalizing evaluation")
        db_evaluation.parsed_resume = resume_struct
        db_evaluation.parsed_jd = jd_struct
        db_evaluation.processing_time_seconds = time.time() - start_time
        db_evaluation.embedding_model_used = settings.embedding_model
        db_evaluation.status = "completed"
        db_evaluation.processing_stage = "completed"
        
        db.commit()
        
        logger.info(f"Evaluation completed for {resume_id} in {time.time() - start_time:.2f}s")
        
        return {
            "status": "success",
            "resume_id": resume_id,
            "jd_id": jd_id,
            "final_score": db_evaluation.final_score,
            "verdict": db_evaluation.verdict,
            "processing_time": db_evaluation.processing_time_seconds
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed for {resume_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update evaluation with error
        if 'db_evaluation' in locals():
            db_evaluation.status = "failed"
            db_evaluation.error_message = str(e)
            db_evaluation.error_stage = getattr(db_evaluation, 'processing_stage', 'unknown')
            db.commit()
        
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
    finally:
        db.close()

@celery_app.task(name="cleanup_old_files")
def cleanup_old_files() -> Dict[str, Any]:
    """
    Periodic task to clean up old uploaded files.
    Removes files older than 30 days to save disk space.
    """
    logger.info("Starting file cleanup task")
    
    from datetime import timedelta
    import os
    import stat
    
    upload_dir = Path(settings.upload_directory)
    cutoff_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
    
    deleted_count = 0
    total_size_deleted = 0
    
    try:
        for file_path in upload_dir.rglob("*"):
            if file_path.is_file():
                file_stat = file_path.stat()
                if file_stat.st_mtime < cutoff_time:
                    try:
                        total_size_deleted += file_stat.st_size
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleanup completed: {deleted_count} files deleted, {total_size_deleted / 1024 / 1024:.2f} MB freed")
        
        return {
            "status": "success",
            "files_deleted": deleted_count,
            "bytes_freed": total_size_deleted
        }
        
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@celery_app.task(name="health_check")
def health_check() -> Dict[str, Any]:
    """Simple health check task for monitoring worker status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": os.getpid()
    }

# Task routing and periodic tasks
celery_app.conf.beat_schedule = {
    'cleanup-old-files': {
        'task': 'cleanup_old_files',
        'schedule': 24 * 60 * 60.0,  # Daily cleanup
    },
}

if __name__ == "__main__":
    # For running worker directly
    celery_app.start()
