"""
Named Entity Recognition (NER) Module

This module uses spaCy for extracting structured information from resume text.
NER (Named Entity Recognition) is an NLP technique that identifies and classifies
named entities in text like person names, organizations, locations, etc.

We use spaCy's pre-trained models to extract:
- Personal information (name, email, phone)
- Skills (both from dedicated sections and inline mentions)
- Education details (degrees, institutions, years)
- Experience information (job titles, companies, dates)
- Projects and achievements

The en_core_web_sm model is used by default for speed, but en_core_web_trf
(transformer-based) can be used for higher accuracy if available.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import utility functions
from .utils import (
    extract_email, extract_phone, normalize_skill, extract_skills_from_text,
    parse_date_range, clean_company_name, is_technical_skill
)

# spaCy for NLP processing
try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
    
    # Try to load spaCy model (prefer en_core_web_sm for speed)
    try:
        nlp = spacy.load("en_core_web_sm")
        NLP_MODEL = "en_core_web_sm"
    except OSError:
        try:
            # Fallback to transformer model if available
            nlp = spacy.load("en_core_web_trf")
            NLP_MODEL = "en_core_web_trf"
            logging.info("Using transformer model en_core_web_trf for higher accuracy")
        except OSError:
            logging.error("No spaCy model found. Please install: python -m spacy download en_core_web_sm")
            nlp = None
            NLP_MODEL = None
            
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    NLP_MODEL = None
    logging.warning("spaCy not available")

# Common job titles for matching
JOB_TITLES = {
    'software engineer', 'software developer', 'full stack developer', 'frontend developer',
    'backend developer', 'web developer', 'mobile developer', 'devops engineer',
    'data scientist', 'data analyst', 'data engineer', 'machine learning engineer',
    'ai engineer', 'product manager', 'project manager', 'business analyst',
    'qa engineer', 'test engineer', 'sre', 'cloud engineer', 'security engineer',
    'intern', 'internship', 'associate', 'senior', 'lead', 'principal', 'staff',
    'consultant', 'architect', 'analyst', 'specialist', 'coordinator', 'manager'
}

# Education-related keywords
EDUCATION_KEYWORDS = {
    'bachelor', 'master', 'phd', 'doctorate', 'mba', 'bs', 'ba', 'ms', 'ma',
    'btech', 'mtech', 'be', 'me', 'bsc', 'msc', 'degree', 'diploma',
    'computer science', 'engineering', 'information technology', 'mathematics',
    'physics', 'chemistry', 'business', 'economics', 'finance', 'marketing'
}


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract structured entities from resume text using spaCy NLP.
    
    This is the main function that orchestrates entity extraction.
    It processes the full resume text and returns a structured dictionary
    with all extracted information.
    
    Args:
        text (str): Full resume text (should be normalized/cleaned)
        
    Returns:
        Dict[str, Any]: Structured dictionary containing:
            - name: Person's name
            - email: Email address
            - phone: Phone number
            - skills: List of technical skills
            - education: List of education entries
            - experience: List of work experience entries
            - projects: List of project entries
    """
    if not text or not SPACY_AVAILABLE or not nlp:
        return _get_empty_entity_dict()
    
    # Process text with spaCy
    doc = nlp(text[:1000000])  # Limit text length for performance
    
    # Extract different types of entities
    result = {
        'name': _extract_name(doc, text),
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': _extract_skills(doc, text),
        'education': _extract_education(doc, text),
        'experience': _extract_experience(doc, text),
        'projects': _extract_projects(doc, text)
    }
    
    return result


def _get_empty_entity_dict() -> Dict[str, Any]:
    """Return empty entity dictionary structure."""
    return {
        'name': None,
        'email': None,
        'phone': None,
        'skills': [],
        'education': [],
        'experience': [],
        'projects': []
    }


def _extract_name(doc, text: str) -> Optional[str]:
    """
    Extract person's name from resume text.
    
    Uses spaCy's PERSON entity recognition and heuristics to find
    the most likely candidate for the person's name (usually at the top).
    
    Args:
        doc: spaCy processed document
        text (str): Original text
        
    Returns:
        Optional[str]: Extracted name or None
    """
    # Look for PERSON entities in the first few lines (where name usually appears)
    first_lines = '\n'.join(text.split('\n')[:5])
    first_doc = nlp(first_lines) if nlp else None
    
    if first_doc:
        persons = [ent.text for ent in first_doc.ents if ent.label_ == "PERSON"]
        if persons:
            # Return the first person name found in the beginning
            return persons[0].strip()
    
    # Fallback: look for name patterns in first few lines
    lines = text.split('\n')[:3]
    for line in lines:
        line = line.strip()
        # Skip lines with common resume keywords
        if any(keyword in line.lower() for keyword in ['resume', 'cv', 'email', 'phone', 'address']):
            continue
        # Look for lines that might be names (2-4 words, proper capitalization)
        words = line.split()
        if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word.isalpha()):
            return line
    
    return None


def _extract_skills(doc, text: str) -> List[str]:
    """
    Extract technical skills from resume text.
    
    Combines multiple approaches:
    1. Look for dedicated "Skills" sections
    2. Extract inline skill mentions from experience bullets
    3. Use spaCy entities and custom matching patterns
    
    Args:
        doc: spaCy processed document
        text (str): Original text
        
    Returns:
        List[str]: List of normalized technical skills
    """
    skills = set()
    
    # Method 1: Extract from Skills section
    skills_section = _find_skills_section(text)
    if skills_section:
        skills.update(_parse_skills_from_section(skills_section))
    
    # Method 2: Extract inline skills from experience/project sections
    skills.update(_extract_inline_skills(text))
    
    # Method 3: Use general skill extraction from full text
    skills.update(extract_skills_from_text(text))
    
    # Normalize and filter technical skills
    normalized_skills = []
    for skill in skills:
        normalized = normalize_skill(skill)
        if normalized and is_technical_skill(normalized):
            normalized_skills.append(normalized)
    
    # Remove duplicates and sort
    return sorted(list(set(normalized_skills)))


def _find_skills_section(text: str) -> Optional[str]:
    """Find and extract the Skills section from resume text."""
    lines = text.split('\n')
    in_skills_section = False
    skills_content = []
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Check if this is a skills section header
        if any(header in line_lower for header in ['skills', 'technical skills', 'competencies']):
            in_skills_section = True
            continue
        
        # Check if we've moved to a new section
        if in_skills_section and line.strip() and any(
            header in line_lower for header in 
            ['experience', 'education', 'projects', 'achievements', 'certifications']
        ):
            break
        
        # Collect skills content
        if in_skills_section and line.strip():
            skills_content.append(line.strip())
    
    return '\n'.join(skills_content) if skills_content else None


def _parse_skills_from_section(skills_text: str) -> List[str]:
    """Parse individual skills from a skills section."""
    if not skills_text:
        return []
    
    skills = []
    
    # Split by common delimiters
    text = re.sub(r'[,;|•◦▪▫]', '\n', skills_text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # Further split each line by commas or semicolons
        items = re.split(r'[,;]+', line)
        for item in items:
            skill = item.strip()
            if skill and len(skill) > 1:
                skills.append(skill)
    
    return skills


def _extract_inline_skills(text: str) -> List[str]:
    """Extract skills mentioned inline in experience/project descriptions."""
    skills = []
    
    # Look for skill patterns in bullet points and descriptions
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        
        # Skip non-descriptive lines
        if len(line.strip()) < 10:
            continue
        
        # Look for technology/tool mentions
        skills.extend(_find_tech_mentions(line))
    
    return skills


def _find_tech_mentions(text: str) -> List[str]:
    """Find technology mentions in a line of text."""
    skills = []
    text_lower = text.lower()
    
    # Look for patterns like "using Python", "with React", "in Java"
    patterns = [
        r'using\s+([a-zA-Z0-9+#.]+)',
        r'with\s+([a-zA-Z0-9+#.]+)',
        r'in\s+([a-zA-Z0-9+#.]+)',
        r'built\s+with\s+([a-zA-Z0-9+#.]+)',
        r'developed\s+in\s+([a-zA-Z0-9+#.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) > 1 and match.isalpha():
                skills.append(match)
    
    return skills


def _extract_education(doc, text: str) -> List[Dict[str, Any]]:
    """
    Extract education information from resume text.
    
    Looks for degree information, institutions, graduation years,
    and relevant coursework.
    
    Args:
        doc: spaCy processed document
        text (str): Original text
        
    Returns:
        List[Dict]: List of education entries
    """
    education = []
    
    # Find education section
    education_section = _find_education_section(text)
    if not education_section:
        return education
    
    # Parse education entries
    entries = _split_education_entries(education_section)
    
    for entry in entries:
        edu_info = _parse_education_entry(entry)
        if edu_info:
            education.append(edu_info)
    
    return education


def _find_education_section(text: str) -> Optional[str]:
    """Find and extract the Education section from resume text."""
    lines = text.split('\n')
    in_education_section = False
    education_content = []
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Check if this is an education section header
        if any(header in line_lower for header in ['education', 'academic', 'qualifications']):
            in_education_section = True
            continue
        
        # Check if we've moved to a new section
        if in_education_section and line.strip() and any(
            header in line_lower for header in 
            ['experience', 'skills', 'projects', 'achievements', 'certifications']
        ):
            break
        
        # Collect education content
        if in_education_section and line.strip():
            education_content.append(line.strip())
    
    return '\n'.join(education_content) if education_content else None


def _split_education_entries(education_text: str) -> List[str]:
    """Split education section into individual entries."""
    if not education_text:
        return []
    
    # Simple approach: split by lines that start with degree or year
    lines = education_text.split('\n')
    entries = []
    current_entry = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this starts a new education entry
        if any(keyword in line.lower() for keyword in EDUCATION_KEYWORDS):
            if current_entry:
                entries.append('\n'.join(current_entry))
                current_entry = []
        
        current_entry.append(line)
    
    # Add the last entry
    if current_entry:
        entries.append('\n'.join(current_entry))
    
    return entries


def _parse_education_entry(entry_text: str) -> Optional[Dict[str, Any]]:
    """Parse individual education entry."""
    if not entry_text:
        return None
    
    # Extract degree, institution, year
    lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
    
    degree = None
    institution = None
    year = None
    
    for line in lines:
        # Look for degree information
        if not degree and any(keyword in line.lower() for keyword in EDUCATION_KEYWORDS):
            degree = line
        
        # Look for years
        year_match = re.search(r'\b(19|20)\d{2}\b', line)
        if year_match and not year:
            year = int(year_match.group(0))
        
        # Institution is usually the line without degree keywords or years
        if not institution and not any(keyword in line.lower() for keyword in EDUCATION_KEYWORDS):
            if not re.search(r'\b(19|20)\d{2}\b', line):
                institution = line
    
    if degree or institution:
        return {
            'degree': degree,
            'institution': institution,
            'year': year,
            'stream': None  # Could be enhanced to extract field of study
        }
    
    return None


def _extract_experience(doc, text: str) -> List[Dict[str, Any]]:
    """
    Extract work experience information from resume text.
    
    Looks for job titles, companies, employment dates, and
    responsibility bullets.
    
    Args:
        doc: spaCy processed document
        text (str): Original text
        
    Returns:
        List[Dict]: List of experience entries
    """
    experience = []
    
    # Find experience section
    experience_section = _find_experience_section(text)
    if not experience_section:
        return experience
    
    # Parse experience entries
    entries = _split_experience_entries(experience_section)
    
    for entry in entries:
        exp_info = _parse_experience_entry(entry)
        if exp_info:
            experience.append(exp_info)
    
    return experience


def _find_experience_section(text: str) -> Optional[str]:
    """Find and extract the Experience section from resume text."""
    lines = text.split('\n')
    in_experience_section = False
    experience_content = []
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Check if this is an experience section header
        if any(header in line_lower for header in ['experience', 'employment', 'work history']):
            in_experience_section = True
            continue
        
        # Check if we've moved to a new section
        if in_experience_section and line.strip() and any(
            header in line_lower for header in 
            ['education', 'skills', 'projects', 'achievements', 'certifications']
        ):
            break
        
        # Collect experience content
        if in_experience_section and line.strip():
            experience_content.append(line.strip())
    
    return '\n'.join(experience_content) if experience_content else None


def _split_experience_entries(experience_text: str) -> List[str]:
    """Split experience section into individual job entries."""
    if not experience_text:
        return []
    
    lines = experience_text.split('\n')
    entries = []
    current_entry = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this starts a new job entry (contains job title keywords)
        if any(title in line.lower() for title in JOB_TITLES) or re.search(r'\b(19|20)\d{2}\b', line):
            if current_entry and len(current_entry) > 1:  # Ensure substantial content
                entries.append('\n'.join(current_entry))
                current_entry = []
        
        current_entry.append(line)
    
    # Add the last entry
    if current_entry and len(current_entry) > 1:
        entries.append('\n'.join(current_entry))
    
    return entries


def _parse_experience_entry(entry_text: str) -> Optional[Dict[str, Any]]:
    """Parse individual experience entry."""
    if not entry_text:
        return None
    
    lines = [line.strip() for line in entry_text.split('\n') if line.strip()]
    
    title = None
    company = None
    start_date = None
    end_date = None
    bullets = []
    
    for i, line in enumerate(lines):
        # First line is usually title and/or company
        if i == 0:
            # Try to separate title and company
            if ' at ' in line:
                parts = line.split(' at ', 1)
                title = parts[0].strip()
                company = clean_company_name(parts[1].strip())
            elif any(title_word in line.lower() for title_word in JOB_TITLES):
                title = line
        
        # Look for date ranges
        date_match = re.search(r'(\w+\s+\d{4})\s*[-–—]\s*(\w+\s+\d{4}|present)', line.lower())
        if date_match and not start_date:
            start_date = date_match.group(1)
            end_date = date_match.group(2).title()
        
        # Collect bullet points (lines starting with bullets or describing work)
        if line.startswith(('•', '-', '*', '◦')) or (len(line) > 20 and i > 0):
            clean_bullet = re.sub(r'^[•\-*◦]\s*', '', line)
            if clean_bullet:
                bullets.append(clean_bullet)
    
    if title or company:
        return {
            'title': title,
            'company': company,
            'start': start_date,
            'end': end_date,
            'bullets': bullets
        }
    
    return None


def _extract_projects(doc, text: str) -> List[Dict[str, Any]]:
    """
    Extract project information from resume text.
    
    Args:
        doc: spaCy processed document
        text (str): Original text
        
    Returns:
        List[Dict]: List of project entries
    """
    projects = []
    
    # Find projects section
    projects_section = _find_projects_section(text)
    if not projects_section:
        return projects
    
    # Simple project parsing
    lines = [line.strip() for line in projects_section.split('\n') if line.strip()]
    
    current_project = None
    current_desc = []
    
    for line in lines:
        # Check if this is a project title (often starts with capital or bullet)
        if (line[0].isupper() or line.startswith(('•', '-', '*'))) and len(line) < 100:
            if current_project:
                projects.append({
                    'title': current_project,
                    'desc': ' '.join(current_desc)
                })
            current_project = re.sub(r'^[•\-*]\s*', '', line)
            current_desc = []
        else:
            current_desc.append(line)
    
    # Add last project
    if current_project:
        projects.append({
            'title': current_project,
            'desc': ' '.join(current_desc)
        })
    
    return projects


def _find_projects_section(text: str) -> Optional[str]:
    """Find and extract the Projects section from resume text."""
    lines = text.split('\n')
    in_projects_section = False
    projects_content = []
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Check if this is a projects section header
        if any(header in line_lower for header in ['projects', 'personal projects', 'key projects']):
            in_projects_section = True
            continue
        
        # Check if we've moved to a new section
        if in_projects_section and line.strip() and any(
            header in line_lower for header in 
            ['experience', 'education', 'skills', 'achievements', 'certifications']
        ):
            break
        
        # Collect projects content
        if in_projects_section and line.strip():
            projects_content.append(line.strip())
    
    return '\n'.join(projects_content) if projects_content else None
