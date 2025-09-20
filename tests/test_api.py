"""
FastAPI Integration Tests

This module tests the complete API endpoints including:
- File upload and validation
- Database integration 
- Background task processing (mocked)
- Authentication and error handling

Run tests with: pytest tests/test_api.py -v
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our application
import sys
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app, get_current_user
from api.database import get_db, Base
from api.models import User, JobDescription, Resume, Evaluation
from api.config import settings

# Test database URL (SQLite in-memory for fast testing)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# Create test engine and session
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database
Base.metadata.create_all(bind=engine)

def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

def override_get_current_user():
    """Override user dependency for testing."""
    return User(
        user_id="test-user-123",
        username="testuser",
        email="test@example.com",
        role="staff",
        is_active=True
    )

# Apply dependency overrides
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_background_tasks():
    """Mock background tasks to avoid actual processing during tests."""
    with patch('api.main.BackgroundTasks') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_jd_data():
    """Sample job description data for testing."""
    return {
        "title": "Test Python Developer",
        "company": "Test Company",
        "location": "Test City",
        "description": "Looking for a Python developer with Django experience",
        "must_have_skills": ["python", "django"],
        "good_to_have_skills": ["react", "aws"],
        "education_requirements": {"level": "bachelor"},
        "certifications_required": []
    }

@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing."""
    # Create a simple text file (simulating PDF for test purposes)
    content = b"This is a test resume content. Python developer with 5 years experience."
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file.flush()
        yield tmp_file.name
    
    # Clean up
    os.unlink(tmp_file.name)

class TestHealthEndpoints:
    """Test basic health and status endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_stats_endpoint(self):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_resumes" in data
        assert "total_jds" in data
        assert "total_evaluations" in data
        assert isinstance(data["total_resumes"], int)

class TestJobDescriptionUpload:
    """Test job description upload and processing."""
    
    def test_upload_jd_json(self, mock_background_tasks, sample_jd_data):
        """Test uploading JD via JSON data."""
        response = client.post(
            "/upload-jd",
            data={"jd_data": json.dumps(sample_jd_data)}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "jd_id" in data
        assert data["title"] == sample_jd_data["title"]
        assert data["company"] == sample_jd_data["company"]
        assert data["status"] == "active"

    def test_upload_jd_file(self, mock_background_tasks, sample_pdf_file):
        """Test uploading JD via file."""
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/upload-jd",
                files={"file": ("test_jd.pdf", f, "application/pdf")},
                data={
                    "title": "Test Position",
                    "company": "Test Company"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "jd_id" in data
        assert data["title"] == "Test Position"
        assert data["status"] == "processing"
        
        # Verify background task was queued
        mock_background_tasks.add_task.assert_called_once()

    def test_upload_jd_no_data(self):
        """Test error when no data provided."""
        response = client.post("/upload-jd")
        
        assert response.status_code == 400
        assert "Either file or jd_data must be provided" in response.json()["detail"]

    def test_upload_jd_invalid_json(self):
        """Test error with invalid JSON data."""
        response = client.post(
            "/upload-jd",
            data={"jd_data": "invalid json"}
        )
        
        assert response.status_code == 400
        assert "Invalid JSON data" in response.json()["detail"]

class TestResumeUpload:
    """Test resume upload and processing."""
    
    @pytest.fixture(autouse=True)
    def setup_jd(self):
        """Create a test job description for resume tests."""
        db = TestingSessionLocal()
        self.test_jd = JobDescription(
            jd_id="test-jd-123",
            title="Test Job",
            company="Test Company",
            status="active"
        )
        db.add(self.test_jd)
        db.commit()
        db.close()
        yield
        
        # Cleanup
        db = TestingSessionLocal()
        db.query(JobDescription).filter(JobDescription.jd_id == "test-jd-123").delete()
        db.commit()
        db.close()

    def test_upload_resume_success(self, mock_background_tasks, sample_pdf_file):
        """Test successful resume upload."""
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/upload-resume",
                files={"file": ("test_resume.pdf", f, "application/pdf")},
                data={
                    "jd_id": "test-jd-123",
                    "candidate_name": "John Doe",
                    "candidate_email": "john@example.com"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "resume_id" in data
        assert data["filename"] == "test_resume.pdf"
        assert data["jd_id"] == "test-jd-123"
        assert data["status"] == "processing"
        
        # Verify background task was queued
        mock_background_tasks.add_task.assert_called_once()

    def test_upload_resume_without_jd(self, mock_background_tasks, sample_pdf_file):
        """Test resume upload without JD assignment."""
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/upload-resume",
                files={"file": ("test_resume.pdf", f, "application/pdf")},
                data={"candidate_name": "Jane Doe"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["jd_id"] is None
        assert "uploaded successfully" in data["message"]

    def test_upload_resume_invalid_jd(self, sample_pdf_file):
        """Test error with non-existent JD ID."""
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/upload-resume", 
                files={"file": ("test_resume.pdf", f, "application/pdf")},
                data={"jd_id": "non-existent-jd"}
            )
        
        assert response.status_code == 404
        assert "Job description not found" in response.json()["detail"]

class TestFileValidation:
    """Test file upload validation."""
    
    def test_file_too_large(self):
        """Test error with oversized file."""
        # Create a large file content
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(large_content)
            tmp_file.flush()
            
            with open(tmp_file.name, "rb") as f:
                response = client.post(
                    "/upload-resume",
                    files={"file": ("large_resume.pdf", f, "application/pdf")}
                )
        
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_invalid_file_type(self):
        """Test error with invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix=".exe") as tmp_file:
            tmp_file.write(b"invalid content")
            tmp_file.flush()
            
            with open(tmp_file.name, "rb") as f:
                response = client.post(
                    "/upload-resume",
                    files={"file": ("malware.exe", f, "application/octet-stream")}
                )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

class TestResultsRetrieval:
    """Test evaluation results retrieval."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Create test resume and evaluation data."""
        db = TestingSessionLocal()
        
        # Create test resume
        self.test_resume = Resume(
            resume_id="test-resume-123",
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            status="parsed"
        )
        db.add(self.test_resume)
        
        # Create test evaluation
        self.test_evaluation = Evaluation(
            resume_id="test-resume-123",
            jd_id="test-jd-123",
            status="completed",
            hard_score=75.0,
            semantic_score=80.0,
            final_score=77,
            verdict="high",
            missing_elements=[{
                "type": "skills",
                "items": ["docker", "kubernetes"],
                "priority": "moderate"
            }],
            feedback_suggestions=[{
                "action": "Add Docker experience to your projects",
                "example": "Containerize an existing application using Docker",
                "priority": "medium",
                "category": "skill"
            }]
        )
        db.add(self.test_evaluation)
        db.commit()
        db.close()
        
        yield
        
        # Cleanup
        db = TestingSessionLocal()
        db.query(Evaluation).filter(Evaluation.resume_id == "test-resume-123").delete()
        db.query(Resume).filter(Resume.resume_id == "test-resume-123").delete()
        db.commit()
        db.close()

    def test_get_results_success(self):
        """Test successful results retrieval."""
        response = client.get("/results/test-resume-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["resume_id"] == "test-resume-123"
        assert data["status"] == "completed"
        assert data["hard_score"] == 75.0
        assert data["semantic_score"] == 80.0
        assert data["final_score"] == 77
        assert data["verdict"] == "high"
        assert len(data["missing_elements"]) > 0
        assert len(data["feedback_suggestions"]) > 0

    def test_get_results_not_found(self):
        """Test error for non-existent resume."""
        response = client.get("/results/non-existent-resume")
        
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]

    def test_get_results_not_evaluated(self):
        """Test response for resume without evaluation."""
        db = TestingSessionLocal()
        unevaluated_resume = Resume(
            resume_id="unevaluated-resume-123",
            filename="unevaluated.pdf", 
            file_path="/tmp/unevaluated.pdf",
            status="uploaded"
        )
        db.add(unevaluated_resume)
        db.commit()
        
        try:
            response = client.get("/results/unevaluated-resume-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_evaluated"
            assert data["final_score"] is None
            assert "No evaluation found" in data["error_message"]
        
        finally:
            db.query(Resume).filter(Resume.resume_id == "unevaluated-resume-123").delete()
            db.commit()
            db.close()

class TestSearchEndpoint:
    """Test resume search and filtering."""
    
    @pytest.fixture(autouse=True) 
    def setup_search_data(self):
        """Create test data for search tests."""
        db = TestingSessionLocal()
        
        # Create test JD
        test_jd = JobDescription(
            jd_id="search-jd-123",
            title="Senior Developer",
            company="SearchCorp",
            location="San Francisco",
            status="active"
        )
        db.add(test_jd)
        
        # Create test resumes with evaluations
        for i in range(3):
            resume = Resume(
                resume_id=f"search-resume-{i}",
                filename=f"resume_{i}.pdf",
                file_path=f"/tmp/resume_{i}.pdf",
                candidate_name=f"Candidate {i}",
                status="parsed"
            )
            db.add(resume)
            
            evaluation = Evaluation(
                resume_id=f"search-resume-{i}",
                jd_id="search-jd-123", 
                status="completed",
                final_score=70 + i * 10,  # Scores: 70, 80, 90
                verdict=["medium", "high", "high"][i]
            )
            db.add(evaluation)
        
        db.commit()
        db.close()
        
        yield
        
        # Cleanup
        db = TestingSessionLocal()
        for i in range(3):
            db.query(Evaluation).filter(Evaluation.resume_id == f"search-resume-{i}").delete()
            db.query(Resume).filter(Resume.resume_id == f"search-resume-{i}").delete()
        db.query(JobDescription).filter(JobDescription.jd_id == "search-jd-123").delete()
        db.commit()
        db.close()

    def test_search_all_resumes(self):
        """Test search without filters."""
        response = client.get("/search")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert all("resume_id" in item for item in data)
        assert all("final_score" in item for item in data)

    def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        response = client.get("/search?min_score=80")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # Only scores 80 and 90
        assert all(item["final_score"] >= 80 for item in data)

    def test_search_with_verdict(self):
        """Test search with verdict filter."""
        response = client.get("/search?verdict=high")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # Two high verdicts
        assert all(item["verdict"] == "high" for item in data)

    def test_search_with_role_filter(self):
        """Test search with role/title filter."""
        response = client.get("/search?role=developer")
        
        assert response.status_code == 200
        data = response.json()
        # Should find matches in JD title containing "Developer"
        assert len(data) >= 1

    def test_search_with_pagination(self):
        """Test search with pagination parameters."""
        response = client.get("/search?limit=2&offset=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2  # Respects limit

class TestEvaluationTrigger:
    """Test manual evaluation triggering."""
    
    def test_evaluate_resume_success(self, mock_background_tasks):
        """Test successful evaluation trigger."""
        # Create test data
        db = TestingSessionLocal()
        
        test_jd = JobDescription(
            jd_id="eval-jd-123",
            title="Test Job",
            company="Test Co",
            status="active"
        )
        db.add(test_jd)
        
        test_resume = Resume(
            resume_id="eval-resume-123",
            filename="eval_test.pdf",
            file_path="/tmp/eval_test.pdf",
            status="uploaded"
        )
        db.add(test_resume)
        db.commit()
        
        try:
            response = client.post(
                "/evaluate/eval-resume-123",
                data={"jd_id": "eval-jd-123"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["resume_id"] == "eval-resume-123"
            assert data["jd_id"] == "eval-jd-123" 
            assert "Evaluation started" in data["message"]
            
            # Verify background task queued
            mock_background_tasks.add_task.assert_called_once()
        
        finally:
            db.query(Resume).filter(Resume.resume_id == "eval-resume-123").delete()
            db.query(JobDescription).filter(JobDescription.jd_id == "eval-jd-123").delete()
            db.commit()
            db.close()

    def test_evaluate_resume_not_found(self):
        """Test error for non-existent resume."""
        response = client.post(
            "/evaluate/non-existent-resume",
            data={"jd_id": "any-jd"}
        )
        
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]

    def test_evaluate_jd_not_found(self):
        """Test error for non-existent JD."""
        db = TestingSessionLocal()
        test_resume = Resume(
            resume_id="test-resume-456",
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        db.add(test_resume)
        db.commit()
        
        try:
            response = client.post(
                "/evaluate/test-resume-456",
                data={"jd_id": "non-existent-jd"}
            )
            
            assert response.status_code == 404
            assert "Job description not found" in response.json()["detail"]
        
        finally:
            db.query(Resume).filter(Resume.resume_id == "test-resume-456").delete()
            db.commit()
            db.close()

@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for complete workflows."""
    
    @patch('api.tasks.evaluate_resume_task.delay')
    def test_complete_upload_and_evaluation_workflow(self, mock_task, sample_jd_data, sample_pdf_file):
        """Test complete workflow from JD upload to resume evaluation."""
        # Step 1: Upload JD
        jd_response = client.post(
            "/upload-jd",
            data={"jd_data": json.dumps(sample_jd_data)}
        )
        assert jd_response.status_code == 200
        jd_id = jd_response.json()["jd_id"]
        
        # Step 2: Upload resume
        with open(sample_pdf_file, "rb") as f:
            resume_response = client.post(
                "/upload-resume",
                files={"file": ("test_resume.pdf", f, "application/pdf")},
                data={"jd_id": jd_id}
            )
        
        assert resume_response.status_code == 200
        resume_id = resume_response.json()["resume_id"]
        
        # Step 3: Check task was queued
        assert mock_task.called
        
        # Step 4: Check results (would be "processing" in real scenario)
        results_response = client.get(f"/results/{resume_id}")
        assert results_response.status_code == 200
        
        # Step 5: Search should find the resume
        search_response = client.get("/search")
        assert search_response.status_code == 200
        resume_ids = [item["resume_id"] for item in search_response.json()]
        assert resume_id in resume_ids

if __name__ == "__main__":
    # Run tests when script executed directly
    pytest.main([__file__, "-v"])
