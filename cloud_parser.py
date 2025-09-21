#!/usr/bin/env python3
"""
Simplified Resume Parser for Streamlit Cloud Deployment
======================================================

A lightweight version of the resume parser optimized for cloud deployment
with minimal dependencies and error handling.
"""

import re
import os
from typing import Dict, List, Optional, Any
import json

# Basic skill patterns
TECHNICAL_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
    'html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab',
    'machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
    'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter',
    'api', 'rest', 'graphql', 'microservices', 'devops', 'ci/cd', 'agile', 'scrum'
}

def extract_text_simple(file_path: str) -> str:
    """
    Simple text extraction for cloud deployment.
    Falls back to basic file reading if specialized libraries fail.
    """
    try:
        # Try PDF extraction
        if file_path.endswith('.pdf'):
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    return text
                except ImportError:
                    return "PDF parsing not available. Please upload a TXT file."
        
        # Try DOCX extraction
        elif file_path.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                try:
                    import docx2txt
                    return docx2txt.process(file_path)
                except ImportError:
                    return "DOCX parsing not available. Please upload a TXT file."
        
        # Plain text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        return f"Error reading file: {str(e)}"

def normalize_text_simple(text: str) -> Dict[str, str]:
    """
    Simple text normalization for cloud deployment.
    """
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Simple section detection
    sections = {'full_text': text}
    
    # Look for common section headers
    if re.search(r'\b(experience|work experience|employment)\b', text, re.IGNORECASE):
        sections['has_experience'] = True
    if re.search(r'\b(education|qualifications|academic)\b', text, re.IGNORECASE):
        sections['has_education'] = True  
    if re.search(r'\b(skills|technical skills|competencies)\b', text, re.IGNORECASE):
        sections['has_skills'] = True
    
    return sections

def extract_entities_simple(text: str) -> Dict[str, Any]:
    """
    Simple entity extraction for cloud deployment.
    Uses regex patterns instead of spaCy for better cloud compatibility.
    """
    entities = {
        'name': None,
        'email': None,
        'phone': None,
        'skills': [],
        'education': [],
        'experience': [],
        'projects': []
    }
    
    text_lower = text.lower()
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        entities['email'] = email_match.group()
    
    # Extract phone
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        entities['phone'] = phone_match.group()
    
    # Extract name (simple heuristic)
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if len(line.split()) == 2 and line.replace(' ', '').isalpha():
            if '@' not in line and len(line) < 50:
                entities['name'] = line
                break
    
    # Extract skills
    found_skills = []
    for skill in TECHNICAL_SKILLS:
        if skill in text_lower:
            found_skills.append(skill)
    entities['skills'] = sorted(found_skills)
    
    # Extract education (simple pattern)
    education_patterns = [
        r'\b(bachelor|master|phd|doctorate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|mba)\b[^\n]*',
        r'\b(university|college|institute)\b[^\n]*\b\d{4}\b'
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities['education'].append({
                'degree': match.strip(),
                'institution': None,
                'year': None
            })
    
    # Extract experience (simple pattern)
    experience_patterns = [
        r'\b(developer|engineer|analyst|manager|lead|senior|junior)\b[^\n]*',
        r'\b(worked at|employed at|experience at)\b[^\n]*'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities['experience'].append({
                'title': match.strip(),
                'company': None,
                'duration': None
            })
    
    return entities

def analyze_resume_simple(text: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple resume analysis for cloud deployment.
    """
    # Extract entities
    entities = extract_entities_simple(text)
    
    # Get requirements
    must_have = set(skill.lower() for skill in job_requirements.get('must_have_skills', []))
    good_to_have = set(skill.lower() for skill in job_requirements.get('good_to_have_skills', []))
    
    # Calculate matches
    candidate_skills = set(entities['skills'])
    must_have_matches = candidate_skills.intersection(must_have)
    good_to_have_matches = candidate_skills.intersection(good_to_have)
    
    # Calculate scores
    must_have_score = (len(must_have_matches) / max(len(must_have), 1)) * 100
    good_to_have_score = (len(good_to_have_matches) / max(len(good_to_have), 1)) * 100
    
    # Overall score calculation
    overall_score = must_have_score * 0.6 + good_to_have_score * 0.4
    
    return {
        'entities': entities,
        'scores': {
            'must_have': must_have_score,
            'good_to_have': good_to_have_score,
            'overall': overall_score
        },
        'matches': {
            'must_have_matches': list(must_have_matches),
            'good_to_have_matches': list(good_to_have_matches),
            'missing_must_have': list(must_have - candidate_skills),
            'missing_good_to_have': list(good_to_have - candidate_skills)
        }
    }

# Export functions for backward compatibility
extract_text_from_file = extract_text_simple
normalize_text = normalize_text_simple
extract_entities = extract_entities_simple
