"""
Utility Functions Module

This module contains helper functions for resume parsing, including:
- Email and phone number extraction using regex patterns
- Skill normalization and mapping (handling synonyms and variations)
- Common utility functions for text processing

These utilities support the main parsing pipeline with reusable components.
"""

import re
from typing import Dict, List, Optional, Set
from datetime import datetime


# Regex patterns for contact information extraction
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    re.IGNORECASE
)

# Phone patterns - handles various formats
PHONE_PATTERNS = [
    re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # 123-456-7890, 123.456.7890, 1234567890
    re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),   # (123) 456-7890
    re.compile(r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'),  # International
]

# Skill normalization mapping - maps variations to standard forms
SKILL_MAPPINGS = {
    # Programming Languages
    'py': 'python',
    'js': 'javascript', 
    'ts': 'typescript',
    'c++': 'cpp',
    'c#': 'csharp',
    'node': 'nodejs',
    'react.js': 'react',
    'vue.js': 'vue',
    'angular.js': 'angular',
    
    # Databases
    'mysql': 'mysql',
    'postgresql': 'postgresql',
    'postgres': 'postgresql',
    'ms sql': 'sql server',
    'mssql': 'sql server',
    'sql server': 'sql server',
    'sqlite': 'sqlite',
    'mongo': 'mongodb',
    'mongodb': 'mongodb',
    'nosql': 'nosql',
    
    # Cloud & DevOps
    'aws': 'amazon web services',
    'amazon web services': 'amazon web services',
    'gcp': 'google cloud platform',
    'google cloud': 'google cloud platform',
    'azure': 'microsoft azure',
    'docker': 'docker',
    'kubernetes': 'kubernetes',
    'k8s': 'kubernetes',
    'jenkins': 'jenkins',
    'git': 'git',
    'github': 'github',
    'gitlab': 'gitlab',
    
    # Web Technologies
    'html': 'html',
    'css': 'css',
    'html5': 'html',
    'css3': 'css',
    'bootstrap': 'bootstrap',
    'jquery': 'jquery',
    'ajax': 'ajax',
    'rest': 'rest api',
    'restful': 'rest api',
    'api': 'api development',
    
    # Frameworks
    'django': 'django',
    'flask': 'flask',
    'fastapi': 'fastapi',
    'spring': 'spring framework',
    'spring boot': 'spring boot',
    'express': 'express.js',
    'express.js': 'express.js',
    
    # Data Science & ML
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'cv': 'computer vision',
    'dl': 'deep learning',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'scikit-learn',
    'sklearn': 'scikit-learn',
    'tensorflow': 'tensorflow',
    'pytorch': 'pytorch',
    'keras': 'keras',
    
    # Tools & Platforms
    'vs code': 'visual studio code',
    'vscode': 'visual studio code',
    'intellij': 'intellij idea',
    'pycharm': 'pycharm',
    'jupyter': 'jupyter notebook',
    'colab': 'google colab',
    'tableau': 'tableau',
    'power bi': 'power bi',
    'excel': 'microsoft excel',
    'powerpoint': 'microsoft powerpoint',
    'word': 'microsoft word',
}

# Common technical skills to recognize (lowercase)
TECHNICAL_SKILLS = {
    # Programming languages
    'python', 'java', 'javascript', 'typescript', 'c', 'cpp', 'csharp', 'php',
    'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    
    # Web technologies
    'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express.js', 'django',
    'flask', 'fastapi', 'spring framework', 'spring boot', 'bootstrap', 'jquery',
    
    # Databases
    'mysql', 'postgresql', 'sql server', 'sqlite', 'mongodb', 'redis', 'elasticsearch',
    'oracle', 'cassandra', 'dynamodb',
    
    # Cloud & DevOps
    'amazon web services', 'google cloud platform', 'microsoft azure', 'docker',
    'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'terraform', 'ansible',
    
    # Data Science & ML
    'machine learning', 'deep learning', 'artificial intelligence', 'data science',
    'natural language processing', 'computer vision', 'pandas', 'numpy', 'scikit-learn',
    'tensorflow', 'pytorch', 'keras', 'matplotlib', 'seaborn', 'tableau', 'power bi',
    
    # Other tools
    'linux', 'windows', 'macos', 'bash', 'powershell', 'visual studio code',
    'intellij idea', 'pycharm', 'jupyter notebook', 'microsoft excel', 'jira', 'confluence'
}


def extract_email(text: str) -> Optional[str]:
    """
    Extract email address from text using regex pattern.
    
    Finds the first valid email address in the text. Email validation
    follows standard RFC patterns but is not overly strict to handle
    edge cases in resume formatting.
    
    Args:
        text (str): Text to search for email
        
    Returns:
        Optional[str]: First email found, or None if no email found
    """
    if not text:
        return None
    
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """
    Extract phone number from text using multiple regex patterns.
    
    Handles various phone number formats commonly found in resumes:
    - 123-456-7890
    - (123) 456-7890  
    - 123.456.7890
    - +1 123 456 7890 (international)
    
    Args:
        text (str): Text to search for phone number
        
    Returns:
        Optional[str]: First phone number found, or None if no phone found
    """
    if not text:
        return None
    
    for pattern in PHONE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    
    return None


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill name to its standard form.
    
    Handles common variations and synonyms (e.g., "JS" -> "javascript",
    "ML" -> "machine learning"). Makes skills consistent for matching
    against job requirements.
    
    Args:
        skill (str): Raw skill name from resume
        
    Returns:
        str: Normalized skill name
    """
    if not skill:
        return ""
    
    # Clean and lowercase
    skill_clean = skill.strip().lower()
    
    # Remove common prefixes/suffixes
    skill_clean = re.sub(r'^(proficient in|experience with|skilled in)\s+', '', skill_clean)
    skill_clean = re.sub(r'\s+(programming|language|framework|library|tool|platform)$', '', skill_clean)
    
    # Apply mapping if exists
    if skill_clean in SKILL_MAPPINGS:
        return SKILL_MAPPINGS[skill_clean]
    
    return skill_clean


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract technical skills from text using keyword matching.
    
    Looks for known technical skills in the text and returns them
    in normalized form. This is used as a fallback when structured
    skill sections aren't clearly marked.
    
    Args:
        text (str): Text to search for skills
        
    Returns:
        List[str]: List of normalized skills found in text
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_skills = set()
    
    # Look for exact skill matches
    for skill in TECHNICAL_SKILLS:
        if skill in text_lower:
            found_skills.add(skill)
    
    # Look for mapped skill variations
    for variation, standard_skill in SKILL_MAPPINGS.items():
        if variation in text_lower:
            found_skills.add(standard_skill)
    
    return sorted(list(found_skills))


def extract_years_of_experience(text: str) -> Optional[int]:
    """
    Extract years of experience from text.
    
    Looks for patterns like "3 years of experience", "5+ years", etc.
    Useful for extracting experience level from job descriptions or
    resume summaries.
    
    Args:
        text (str): Text to search
        
    Returns:
        Optional[int]: Years of experience if found, None otherwise
    """
    if not text:
        return None
    
    # Patterns for years of experience
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+in\s+',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    
    return None


def parse_date_range(date_text: str) -> Dict[str, Optional[str]]:
    """
    Parse date ranges from experience/education entries.
    
    Handles formats like:
    - "Jan 2020 - Present"
    - "2018-2022"
    - "June 2019 to March 2021"
    
    Args:
        date_text (str): Text containing date range
        
    Returns:
        Dict[str, Optional[str]]: Dictionary with 'start' and 'end' dates
    """
    result = {'start': None, 'end': None}
    
    if not date_text:
        return result
    
    text = date_text.strip().lower()
    
    # Replace common separators
    text = re.sub(r'\s*(?:to|-|–|—)\s*', ' TO ', text)
    text = re.sub(r'\bpresent\b|\bcurrent\b|\bnow\b', 'PRESENT', text)
    
    # Simple pattern matching for common formats
    date_patterns = [
        r'(\w+\s+\d{4})\s+TO\s+(\w+\s+\d{4}|PRESENT)',  # Jan 2020 TO Mar 2022
        r'(\d{4})\s+TO\s+(\d{4}|PRESENT)',  # 2020 TO 2022
        r'(\w+\s+\d{2,4})\s+TO\s+(\w+\s+\d{2,4}|PRESENT)',  # Jan 20 TO Mar 22
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            result['start'] = match.group(1).title()
            end_date = match.group(2)
            result['end'] = 'Present' if end_date == 'PRESENT' else end_date.title()
            break
    
    return result


def is_technical_skill(skill: str) -> bool:
    """
    Check if a skill is considered technical/programming-related.
    
    Args:
        skill (str): Skill to check
        
    Returns:
        bool: True if skill is technical, False otherwise
    """
    if not skill:
        return False
    
    skill_normalized = normalize_skill(skill)
    return skill_normalized in TECHNICAL_SKILLS


def clean_company_name(company: str) -> str:
    """
    Clean and standardize company names.
    
    Removes common suffixes like Inc., LLC, etc. and standardizes
    formatting for better matching.
    
    Args:
        company (str): Raw company name
        
    Returns:
        str: Cleaned company name
    """
    if not company:
        return ""
    
    # Remove common company suffixes
    suffixes = [
        r'\s*,?\s*inc\.?$',
        r'\s*,?\s*llc\.?$',
        r'\s*,?\s*ltd\.?$',
        r'\s*,?\s*corp\.?$',
        r'\s*,?\s*corporation\.?$',
        r'\s*,?\s*co\.?$',
        r'\s*,?\s*company\.?$',
    ]
    
    cleaned = company.strip()
    for suffix in suffixes:
        cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()
