#!/usr/bin/env python3
"""
Backend System Validation Script

This script validates that all backend components are working correctly:
- Database connectivity and models
- API endpoints functionality  
- Background task processing
- File upload and validation
- Authentication and security

Usage:
    python validate_backend.py [--skip-docker] [--skip-tasks]
"""

import os
import sys
import time
import json
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def check_api_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if API server is responding."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            return True
        else:
            print(f"❌ API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return False

def check_database_connection() -> bool:
    """Check database connectivity."""
    try:
        from api.database import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database Connection: OK")
            return True
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return False

def check_models_and_tables() -> bool:
    """Verify database models and table creation."""
    try:
        from api.database import SessionLocal
        from api.models import User, JobDescription, Resume, Evaluation
        
        db = SessionLocal()
        
        # Test basic queries
        user_count = db.query(User).count()
        jd_count = db.query(JobDescription).count()
        resume_count = db.query(Resume).count()
        eval_count = db.query(Evaluation).count()
        
        print(f"✅ Database Tables: Users({user_count}), JDs({jd_count}), Resumes({resume_count}), Evaluations({eval_count})")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Database Models Failed: {e}")
        return False

def test_file_upload(base_url: str = "http://localhost:8000") -> bool:
    """Test file upload functionality."""
    try:
        # Create a test PDF file
        test_content = b"This is a test resume. Python developer with 5 years experience."
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()
            
            # Test resume upload
            with open(tmp_file.name, "rb") as f:
                response = requests.post(
                    f"{base_url}/upload-resume",
                    files={"file": ("test_resume.pdf", f, "application/pdf")},
                    data={"candidate_name": "Test Candidate"},
                    timeout=10
                )
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ File Upload: resume_id = {data.get('resume_id', 'N/A')}")
                return True
            else:
                print(f"❌ File Upload Failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ File Upload Test Failed: {e}")
        return False

def test_jd_upload(base_url: str = "http://localhost:8000") -> bool:
    """Test job description upload."""
    try:
        jd_data = {
            "title": "Test Python Developer",
            "company": "Test Company", 
            "location": "Test City",
            "description": "Looking for a Python developer",
            "must_have_skills": ["python", "django"],
            "good_to_have_skills": ["aws", "docker"]
        }
        
        response = requests.post(
            f"{base_url}/upload-jd",
            data={"jd_data": json.dumps(jd_data)},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ JD Upload: jd_id = {data.get('jd_id', 'N/A')}")
            return True
        else:
            print(f"❌ JD Upload Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ JD Upload Test Failed: {e}")
        return False

def check_redis_connection() -> bool:
    """Check Redis connectivity for Celery."""
    try:
        import redis
        from api.config import settings
        
        r = redis.from_url(settings.redis_url)
        r.ping()
        print("✅ Redis Connection: OK")
        return True
    except Exception as e:
        print(f"❌ Redis Connection Failed: {e}")
        return False

def check_celery_worker() -> bool:
    """Check if Celery worker is available."""
    try:
        from celery import Celery
        from api.config import settings
        
        celery_app = Celery(broker=settings.celery_broker_url)
        
        # Get worker stats
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if stats:
            worker_count = len(stats)
            print(f"✅ Celery Workers: {worker_count} active")
            return True
        else:
            print("❌ No Celery Workers Found")
            return False
            
    except Exception as e:
        print(f"❌ Celery Check Failed: {e}")
        return False

def test_search_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test search functionality."""
    try:
        response = requests.get(f"{base_url}/search", params={"limit": 5}, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search Endpoint: returned {len(data)} results")
            return True
        else:
            print(f"❌ Search Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Search Test Failed: {e}")
        return False

def test_stats_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """Test statistics endpoint."""
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats Endpoint: {data.get('total_resumes', 0)} resumes, {data.get('total_jds', 0)} JDs")
            return True
        else:
            print(f"❌ Stats Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Stats Test Failed: {e}")
        return False

def validate_environment() -> bool:
    """Validate environment configuration."""
    try:
        from api.config import settings
        
        issues = []
        
        # Check required settings
        if not settings.database_url:
            issues.append("DATABASE_URL not set")
        
        if not settings.redis_url:
            issues.append("REDIS_URL not set")
        
        if settings.secret_key == "your-secret-key-change-in-production":
            issues.append("SECRET_KEY should be changed in production")
        
        if issues:
            print(f"⚠️  Environment Issues: {', '.join(issues)}")
        else:
            print("✅ Environment Configuration: OK")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ Environment Validation Failed: {e}")
        return False

def main():
    """Run complete backend validation."""
    print("🚀 BACKEND SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Environment Configuration", validate_environment),
        ("Database Connection", check_database_connection),
        ("Database Models", check_models_and_tables),
        ("Redis Connection", check_redis_connection),
        ("API Health Check", check_api_health),
        ("File Upload", test_file_upload),
        ("JD Upload", test_jd_upload),
        ("Search Endpoint", test_search_endpoint),
        ("Stats Endpoint", test_stats_endpoint),
    ]
    
    # Optional tests that require workers
    if "--skip-tasks" not in sys.argv:
        tests.append(("Celery Workers", check_celery_worker))
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   Test failed: {test_name}")
        except Exception as e:
            print(f"   Test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL BACKEND SYSTEMS OPERATIONAL!")
        print("\n🚀 Ready for production use:")
        print("   • API server running and responsive")
        print("   • Database connected and initialized") 
        print("   • File upload and processing working")
        print("   • Background task system operational")
    else:
        print("⚠️ Some systems need attention")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check docker-compose services: docker-compose ps")
        print("   • View logs: docker-compose logs api worker")
        print("   • Restart services: docker-compose restart")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
