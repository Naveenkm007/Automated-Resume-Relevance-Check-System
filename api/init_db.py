"""
Database Initialization Script

This script initializes the database with tables and sample data.
Run this script to set up a fresh database instance.

Usage:
    python -m api.init_db
"""

import sys
import os
from pathlib import Path
import uuid
from datetime import datetime
import secrets

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from api.database import engine, SessionLocal, create_tables
from api.models import User, JobDescription, SystemConfig
from api.config import settings

def create_admin_user(db: Session) -> User:
    """Create default admin user."""
    admin_user = User(
        user_id=str(uuid.uuid4()),
        username="admin",
        email="admin@company.com",
        full_name="System Administrator",
        role="admin",
        is_active=True,
        is_admin=True,
        can_upload_jds=True,
        can_view_all_data=True,
        api_token=secrets.token_urlsafe(32)
    )
    
    db.add(admin_user)
    return admin_user

def create_sample_staff_user(db: Session) -> User:
    """Create sample staff user."""
    staff_user = User(
        user_id=str(uuid.uuid4()),
        username="placement_staff",
        email="staff@company.com",
        full_name="Placement Staff Member",
        role="staff",
        department="Human Resources",
        is_active=True,
        is_admin=False,
        can_upload_jds=True,
        can_view_all_data=False,
        api_token=secrets.token_urlsafe(32)
    )
    
    db.add(staff_user)
    return staff_user

def create_sample_jd(db: Session, created_by: str) -> JobDescription:
    """Create sample job description."""
    sample_jd = JobDescription(
        jd_id=str(uuid.uuid4()),
        title="Senior Python Developer",
        company="TechCorp Inc",
        location="San Francisco, CA",
        description="""
We are seeking an experienced Senior Python Developer to join our dynamic team.
The ideal candidate will have strong experience in Python development, web frameworks,
and database technologies.

Key Responsibilities:
- Develop scalable web applications using Python and Django
- Design and implement REST APIs
- Work with PostgreSQL databases
- Collaborate with frontend developers
- Mentor junior developers

Required Skills:
- 5+ years of Python development experience
- Strong proficiency in Django or Flask
- Experience with PostgreSQL or MySQL
- Knowledge of REST API development
- Git version control experience

Preferred Skills:
- AWS cloud services experience
- Docker containerization
- React.js knowledge
- Redis caching experience
- Machine learning familiarity

Education:
Bachelor's degree in Computer Science or related field

Benefits:
- Competitive salary
- Health insurance
- Flexible work arrangements
- Professional development opportunities
        """.strip(),
        must_have_skills=["python", "django", "postgresql", "rest api", "git"],
        good_to_have_skills=["aws", "docker", "react", "redis", "machine learning"],
        education_requirements={
            "level": "bachelor",
            "field": "computer science"
        },
        certifications_required=[],
        status="active",
        created_by=created_by,
        parsing_status="completed"
    )
    
    db.add(sample_jd)
    return sample_jd

def create_system_config(db: Session):
    """Create default system configuration."""
    configs = [
        {
            "key": "hard_match_weight",
            "value": 0.6,
            "description": "Weight for hard matching in combined score",
            "category": "scoring"
        },
        {
            "key": "semantic_match_weight", 
            "value": 0.4,
            "description": "Weight for semantic matching in combined score",
            "category": "scoring"
        },
        {
            "key": "max_file_size_mb",
            "value": 10,
            "description": "Maximum file size for uploads in MB",
            "category": "limits"
        },
        {
            "key": "feedback_enabled",
            "value": True,
            "description": "Enable LLM feedback generation",
            "category": "features"
        },
        {
            "key": "embedding_cache_enabled",
            "value": True,
            "description": "Cache embeddings for performance",
            "category": "performance"
        }
    ]
    
    for config_data in configs:
        config = SystemConfig(
            key=config_data["key"],
            value=config_data["value"],
            description=config_data["description"],
            category=config_data["category"],
            is_active=True
        )
        db.add(config)

def init_database():
    """Initialize database with tables and sample data."""
    print("🚀 Initializing Resume Relevance Check Database...")
    
    # Create tables
    print("📊 Creating database tables...")
    create_tables()
    print("✅ Tables created successfully")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.username == "admin").first()
        if existing_admin:
            print("⚠️  Admin user already exists, skipping user creation")
        else:
            # Create users
            print("👤 Creating admin user...")
            admin_user = create_admin_user(db)
            
            print("👥 Creating sample staff user...")
            staff_user = create_sample_staff_user(db)
            
            # Create sample job description
            print("📋 Creating sample job description...")
            sample_jd = create_sample_jd(db, admin_user.user_id)
            
            # Create system configuration
            print("⚙️  Creating system configuration...")
            create_system_config(db)
            
            # Commit all changes
            db.commit()
            
            print("\n✅ Database initialization completed!")
            print(f"🔑 Admin API Token: {admin_user.api_token}")
            print(f"🔑 Staff API Token: {staff_user.api_token}")
            print(f"📋 Sample JD ID: {sample_jd.jd_id}")
            
            print("\n📚 Next Steps:")
            print("1. Start the API server: uvicorn api.main:app --reload")
            print("2. Start the worker: celery -A api.tasks worker --loglevel=info")
            print("3. Visit http://localhost:8000/docs for API documentation")
            print("4. Use the API tokens above for authentication")
    
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        db.rollback()
        raise
    
    finally:
        db.close()

def reset_database():
    """Reset database by dropping and recreating all tables."""
    print("⚠️  WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm database reset: ")
    
    if confirm.lower() != 'yes':
        print("Database reset cancelled")
        return
    
    print("🗑️  Dropping all tables...")
    from api.database import drop_tables
    drop_tables()
    
    print("🔄 Reinitializing database...")
    init_database()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database initialization script")
    parser.add_argument("--reset", action="store_true", help="Reset database (delete all data)")
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        init_database()
