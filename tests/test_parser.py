"""
Unit Tests for Resume Parser

This test suite validates the functionality of the resume parser modules
using pytest. Tests cover text extraction, cleaning, normalization, and
entity extraction using sample text data.

Run tests with: pytest tests/test_parser.py -v
"""

import pytest
import json
from pathlib import Path
import tempfile

# Import the modules to test
from resume_parser.extract import extract_text_from_file
from resume_parser.cleaner import normalize_text, extract_section_content, get_section_lines
from resume_parser.ner import extract_entities
from resume_parser.utils import (
    extract_email, extract_phone, normalize_skill, extract_skills_from_text,
    parse_date_range, clean_company_name, is_technical_skill
)


class TestUtils:
    """Test utility functions."""
    
    def test_extract_email(self):
        """Test email extraction from text."""
        # Valid email cases
        assert extract_email("Contact me at john.doe@example.com") == "john.doe@example.com"
        assert extract_email("Email: jane_smith123@company.co.uk") == "jane_smith123@company.co.uk"
        assert extract_email("reach out to test+user@gmail.com for more info") == "test+user@gmail.com"
        
        # No email cases
        assert extract_email("No email here") is None
        assert extract_email("") is None
        assert extract_email("invalid.email@") is None
    
    def test_extract_phone(self):
        """Test phone number extraction from text."""
        # Various phone formats
        assert extract_phone("Call me at 123-456-7890") == "123-456-7890"
        assert extract_phone("Phone: (555) 123-4567") == "(555) 123-4567"
        assert extract_phone("Contact: 555.123.4567") == "555.123.4567"
        assert extract_phone("Mobile: +1 555 123 4567") == "+1 555 123 4567"
        
        # No phone cases
        assert extract_phone("No phone number here") is None
        assert extract_phone("") is None
        assert extract_phone("123") is None  # Too short
    
    def test_normalize_skill(self):
        """Test skill normalization."""
        # Common mappings
        assert normalize_skill("JS") == "javascript"
        assert normalize_skill("py") == "python"
        assert normalize_skill("ML") == "machine learning"
        assert normalize_skill("AWS") == "amazon web services"
        
        # Case insensitive
        assert normalize_skill("PYTHON") == "python"
        assert normalize_skill("React.js") == "react"
        
        # No mapping - return lowercase
        assert normalize_skill("CustomTool") == "customtool"
        assert normalize_skill("") == ""
    
    def test_extract_skills_from_text(self):
        """Test skill extraction from text."""
        text = """
        I have experience with Python, JavaScript, and AWS.
        Worked with Docker, Kubernetes, and PostgreSQL databases.
        """
        skills = extract_skills_from_text(text)
        
        # Should find several skills
        assert "python" in skills
        assert "javascript" in skills
        assert "amazon web services" in skills  # AWS mapped
        assert "docker" in skills
        assert "kubernetes" in skills
        assert "postgresql" in skills
    
    def test_parse_date_range(self):
        """Test date range parsing."""
        # Standard format
        result = parse_date_range("Jan 2020 - Mar 2022")
        assert result['start'] == "Jan 2020"
        assert result['end'] == "Mar 2022"
        
        # Present/current
        result = parse_date_range("June 2021 to Present")
        assert result['start'] == "June 2021"
        assert result['end'] == "Present"
        
        # Year only
        result = parse_date_range("2019-2021")
        assert result['start'] == "2019"
        assert result['end'] == "2021"
        
        # No valid date
        result = parse_date_range("Invalid date string")
        assert result['start'] is None
        assert result['end'] is None
    
    def test_clean_company_name(self):
        """Test company name cleaning."""
        assert clean_company_name("Microsoft Corp") == "Microsoft"
        assert clean_company_name("Google LLC") == "Google"
        assert clean_company_name("Apple Inc.") == "Apple"
        assert clean_company_name("Amazon Company") == "Amazon"
        assert clean_company_name("Simple Name") == "Simple Name"
    
    def test_is_technical_skill(self):
        """Test technical skill recognition."""
        assert is_technical_skill("python") == True
        assert is_technical_skill("JavaScript") == True
        assert is_technical_skill("AWS") == True  # Gets normalized
        assert is_technical_skill("Communication") == False
        assert is_technical_skill("") == False


class TestCleaner:
    """Test text cleaning and normalization functions."""
    
    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for testing."""
        return """
        John Doe
        Software Engineer
        Email: john.doe@email.com | Phone: (555) 123-4567
        
        EXPERIENCE
        Senior Software Developer at Tech Corp
        Jan 2020 - Present
        • Developed web applications using Python and React
        • Led a team of 5 developers
        • Improved system performance by 40%
        
        Junior Developer at StartupXYZ Inc
        Jun 2018 - Dec 2019
        • Built REST APIs using Django
        • Worked with PostgreSQL databases
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology, 2018
        
        SKILLS
        Python, JavaScript, React, Django, PostgreSQL, AWS, Docker
        Machine Learning, Data Analysis
        
        PROJECTS
        E-commerce Platform
        Built a full-stack e-commerce application using MERN stack
        
        Data Analysis Tool
        Created a Python-based tool for analyzing sales data
        """
    
    def test_normalize_text(self, sample_resume_text):
        """Test text normalization and section splitting."""
        sections = normalize_text(sample_resume_text)
        
        # Should identify main sections
        assert 'full_text' in sections
        assert 'experience' in sections
        assert 'education' in sections
        assert 'skills' in sections
        assert 'projects' in sections
        
        # Check section content
        assert 'Senior Software Developer' in sections['experience']
        assert 'Bachelor of Science' in sections['education']
        assert 'Python' in sections['skills']
        assert 'E-commerce Platform' in sections['projects']
    
    def test_extract_section_content(self, sample_resume_text):
        """Test extracting specific section content."""
        skills_section = extract_section_content(sample_resume_text, 'skills')
        assert 'Python' in skills_section
        assert 'JavaScript' in skills_section
        
        experience_section = extract_section_content(sample_resume_text, 'experience')
        assert 'Senior Software Developer' in experience_section
        assert 'Tech Corp' in experience_section
    
    def test_get_section_lines(self):
        """Test splitting section text into lines."""
        section_text = """
        Python, JavaScript, React
        Django, PostgreSQL
        AWS, Docker
        """
        lines = get_section_lines(section_text)
        
        assert len(lines) == 3
        assert 'Python, JavaScript, React' in lines
        assert 'Django, PostgreSQL' in lines
        assert 'AWS, Docker' in lines
    
    def test_normalize_empty_text(self):
        """Test handling of empty or None text."""
        result = normalize_text("")
        assert result == {'full_text': ''}
        
        result = normalize_text(None)
        assert result == {'full_text': ''}


class TestNER:
    """Test Named Entity Recognition functions."""
    
    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for NER testing."""
        return """
        Alice Johnson
        alice.johnson@email.com | (555) 987-6543
        
        EXPERIENCE
        Senior Data Scientist at DataCorp Inc
        March 2021 - Present
        • Developed machine learning models using Python and TensorFlow
        • Analyzed large datasets with Pandas and NumPy
        • Deployed models to AWS cloud infrastructure
        
        Data Analyst at Analytics LLC
        June 2019 - February 2021
        • Created dashboards using Tableau and Power BI
        • Performed statistical analysis with R and SQL
        
        EDUCATION
        Master of Science in Data Science
        Stanford University, 2019
        
        Bachelor of Engineering in Computer Science
        MIT, 2017
        
        SKILLS
        Python, R, SQL, TensorFlow, PyTorch, Pandas, NumPy
        Machine Learning, Deep Learning, Statistical Analysis
        Tableau, Power BI, AWS, Docker
        
        PROJECTS
        Predictive Analytics Platform
        Built an end-to-end ML pipeline for customer churn prediction
        
        Image Classification System
        Developed a CNN model for medical image classification using PyTorch
        """
    
    def test_extract_entities(self, sample_resume_text):
        """Test complete entity extraction."""
        entities = extract_entities(sample_resume_text)
        
        # Check structure
        assert isinstance(entities, dict)
        assert 'name' in entities
        assert 'email' in entities
        assert 'phone' in entities
        assert 'skills' in entities
        assert 'education' in entities
        assert 'experience' in entities
        assert 'projects' in entities
        
        # Check basic extraction (might not be perfect without actual spaCy)
        assert isinstance(entities['skills'], list)
        assert isinstance(entities['education'], list)
        assert isinstance(entities['experience'], list)
        assert isinstance(entities['projects'], list)
    
    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        entities = extract_entities("")
        
        # Should return empty structure
        assert entities['name'] is None
        assert entities['email'] is None
        assert entities['phone'] is None
        assert entities['skills'] == []
        assert entities['education'] == []
        assert entities['experience'] == []
        assert entities['projects'] == []


class TestIntegration:
    """Integration tests that test the full parsing pipeline."""
    
    def test_complete_parsing_pipeline(self):
        """Test the complete parsing workflow."""
        # Sample resume content
        resume_content = """
        Jane Smith
        jane.smith@example.com | +1-555-123-4567
        
        EXPERIENCE
        Software Engineer at Google LLC
        January 2020 - Present
        • Developed scalable web applications using Python and Django
        • Implemented REST APIs serving millions of requests daily
        • Collaborated with cross-functional teams using Agile methodology
        
        EDUCATION
        BS Computer Science, UC Berkeley, 2019
        
        SKILLS
        Python, Django, JavaScript, React, PostgreSQL, AWS, Docker
        """
        
        # Step 1: Normalize text
        sections = normalize_text(resume_content)
        assert 'full_text' in sections
        assert 'experience' in sections
        assert 'education' in sections
        assert 'skills' in sections
        
        # Step 2: Extract entities
        entities = extract_entities(sections['full_text'])
        
        # Verify extraction worked
        assert isinstance(entities, dict)
        assert len(entities['skills']) > 0  # Should find some skills
        
        # The exact results depend on spaCy availability, but structure should be correct
        for key in ['name', 'email', 'phone', 'skills', 'education', 'experience', 'projects']:
            assert key in entities
    
    def test_skills_extraction_comprehensive(self):
        """Test comprehensive skills extraction."""
        text_with_skills = """
        I have experience with Python, JS, and AWS.
        Worked on projects using React.js, Docker, and PostgreSQL.
        Familiar with ML techniques and data analysis.
        """
        
        skills = extract_skills_from_text(text_with_skills)
        
        # Should normalize and find multiple skills
        expected_skills = ['python', 'javascript', 'amazon web services', 'react', 'docker', 'postgresql']
        
        for expected in expected_skills:
            assert expected in skills or any(expected in skill for skill in skills)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_file_handling(self):
        """Test handling of invalid file paths."""
        with pytest.raises(FileNotFoundError):
            extract_text_from_file("nonexistent_file.pdf")
    
    def test_unsupported_file_format(self):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError):
            extract_text_from_file("test.txt")  # Unsupported format
    
    def test_malformed_text_handling(self):
        """Test handling of malformed or unusual text."""
        # Very short text
        result = normalize_text("Hi")
        assert 'full_text' in result
        
        # Text with lots of special characters
        weird_text = "!@#$%^&*()_+{}|:<>?[]\\;'\",./"
        result = normalize_text(weird_text)
        assert isinstance(result, dict)
        
        # Very long text (should not crash)
        long_text = "Test " * 10000
        result = normalize_text(long_text)
        assert 'full_text' in result


def test_sample_data_structure():
    """Test that the output structure matches the expected format."""
    expected_structure = {
        "name": "...",
        "email": "...",
        "phone": "...",
        "skills": ["python", "sql"],
        "education": [{"degree": "B.E", "stream": "CSE", "year": 2024, "college": "..."}],
        "experience": [{"title": "SDE Intern", "company": "X", "start": "2023-06", "end": "2023-08", "bullets": [...]}],
        "projects": [{"title": "...", "desc": "..."}]
    }
    
    # Test that our entity extraction returns the right structure
    sample_text = "John Doe\nSoftware Engineer\njohn@email.com"
    entities = extract_entities(sample_text)
    
    # Check all required keys exist
    for key in expected_structure.keys():
        assert key in entities
    
    # Check data types
    assert isinstance(entities['skills'], list)
    assert isinstance(entities['education'], list)
    assert isinstance(entities['experience'], list)
    assert isinstance(entities['projects'], list)


if __name__ == '__main__':
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])
