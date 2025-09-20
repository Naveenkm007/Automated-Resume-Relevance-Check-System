"""
Text Cleaning and Normalization Module

This module handles cleaning and normalizing extracted text from resumes.
It removes headers/footers, normalizes whitespace, removes non-ASCII noise,
and splits content into logical sections like Experience, Education, Skills, etc.

The cleaning process is crucial for accurate NLP processing downstream.
"""

import re
from typing import Dict, List, Optional, Tuple


# Common resume section headers (case-insensitive matching)
SECTION_HEADERS = {
    'experience': [
        'experience', 'work experience', 'professional experience', 'employment',
        'work history', 'career history', 'professional history', 'employment history'
    ],
    'education': [
        'education', 'educational background', 'academic background', 'qualifications',
        'academic qualifications', 'educational qualifications', 'academics'
    ],
    'skills': [
        'skills', 'technical skills', 'core competencies', 'competencies',
        'technologies', 'technical competencies', 'areas of expertise',
        'expertise', 'proficiencies', 'technical proficiencies'
    ],
    'projects': [
        'projects', 'personal projects', 'academic projects', 'key projects',
        'notable projects', 'project experience', 'project work'
    ],
    'certifications': [
        'certifications', 'certificates', 'professional certifications',
        'licenses', 'credentials'
    ],
    'achievements': [
        'achievements', 'accomplishments', 'awards', 'honors',
        'recognition', 'distinctions'
    ]
}

# Patterns for common resume noise/headers/footers
NOISE_PATTERNS = [
    r'page\s+\d+\s+of\s+\d+',  # Page numbers
    r'confidential',  # Confidential headers
    r'resume\s+of\s+',  # "Resume of" headers
    r'curriculum\s+vitae',  # CV headers
    r'contact\s+information',  # Contact headers
    r'\btel\b|\bphone\b|\bmobile\b|\bcell\b',  # Phone labels (when standalone)
]


def normalize_text(text: str) -> Dict[str, str]:
    """
    Clean and normalize resume text, then split into sections.
    
    This function performs several cleaning steps:
    1. Remove headers/footers and noise patterns
    2. Normalize whitespace (multiple spaces, tabs, newlines)
    3. Remove non-ASCII characters that might interfere with NLP
    4. Split text into logical sections (Experience, Education, etc.)
    
    Args:
        text (str): Raw text extracted from resume
        
    Returns:
        Dict[str, str]: Dictionary with sections as keys and cleaned text as values
                       Keys include: 'full_text', 'experience', 'education', 'skills', etc.
    """
    if not text or not text.strip():
        return {'full_text': ''}
    
    # Step 1: Remove common noise patterns
    cleaned_text = _remove_noise_patterns(text)
    
    # Step 2: Normalize whitespace
    cleaned_text = _normalize_whitespace(cleaned_text)
    
    # Step 3: Remove non-ASCII noise (but preserve common symbols)
    cleaned_text = _remove_non_ascii_noise(cleaned_text)
    
    # Step 4: Split into sections
    sections = _split_into_sections(cleaned_text)
    
    # Add the full cleaned text
    sections['full_text'] = cleaned_text
    
    return sections


def _remove_noise_patterns(text: str) -> str:
    """
    Remove common header/footer patterns and noise from resume text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with noise patterns removed
    """
    # Remove noise patterns (case-insensitive)
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove lines that are mostly symbols/formatting
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that are mostly dashes, equals, or other formatting
        if re.match(r'^[\s\-=_*#]+$', line):
            continue
        # Skip very short lines that are likely formatting artifacts
        if len(line.strip()) <= 2:
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    - Convert multiple spaces/tabs to single space
    - Remove excessive newlines (more than 2 consecutive)
    - Trim whitespace from line ends
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized whitespace
    """
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace more than 2 consecutive newlines with exactly 2
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Trim whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)


def _remove_non_ascii_noise(text: str) -> str:
    """
    Remove problematic non-ASCII characters while preserving useful ones.
    
    Keeps common symbols like bullets (•), dashes (–, —), quotes, etc.
    but removes control characters and other noise.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with non-ASCII noise removed
    """
    # Define characters to keep (in addition to ASCII)
    keep_chars = set([
        '•', '◦', '▪', '▫',  # Bullet points
        '–', '—', ''', ''', '"', '"',  # Typography
        '°', '©', '®', '™',  # Common symbols
    ])
    
    cleaned_chars = []
    for char in text:
        # Keep ASCII characters
        if ord(char) < 128:
            cleaned_chars.append(char)
        # Keep explicitly allowed non-ASCII characters
        elif char in keep_chars:
            cleaned_chars.append(char)
        # Replace other non-ASCII with space (to avoid word joining)
        else:
            cleaned_chars.append(' ')
    
    return ''.join(cleaned_chars)


def _split_into_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into logical sections based on headers.
    
    Identifies section headers like "Experience", "Education", "Skills"
    and splits the text accordingly. Uses fuzzy matching to handle
    variations in section naming.
    
    Args:
        text (str): Cleaned resume text
        
    Returns:
        Dict[str, str]: Dictionary with section names as keys
    """
    sections = {
        'experience': '',
        'education': '',
        'skills': '',
        'projects': '',
        'certifications': '',
        'achievements': '',
        'other': ''
    }
    
    lines = text.split('\n')
    current_section = 'other'
    current_content = []
    
    for line in lines:
        # Check if this line is a section header
        detected_section = _detect_section_header(line)
        
        if detected_section:
            # Save previous section content
            if current_content:
                sections[current_section] += '\n'.join(current_content) + '\n'
                current_content = []
            
            # Switch to new section
            current_section = detected_section
        else:
            # Add line to current section
            current_content.append(line)
    
    # Save final section content
    if current_content:
        sections[current_section] += '\n'.join(current_content)
    
    # Clean up sections (remove empty ones)
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def _detect_section_header(line: str) -> Optional[str]:
    """
    Detect if a line is a section header and return the section type.
    
    Uses fuzzy matching against known section headers. Looks for:
    - Lines that are mostly the header text
    - Lines that start with the header text followed by colon
    - Headers in all caps or title case
    
    Args:
        line (str): Text line to check
        
    Returns:
        Optional[str]: Section type if detected, None otherwise
    """
    line_clean = line.strip().lower()
    
    # Skip very short lines
    if len(line_clean) < 3:
        return None
    
    # Check against each section type
    for section_type, headers in SECTION_HEADERS.items():
        for header in headers:
            # Exact match
            if line_clean == header:
                return section_type
            
            # Header with colon
            if line_clean == header + ':' or line_clean.startswith(header + ':'):
                return section_type
            
            # Header at start of line (with some tolerance for extra words)
            if line_clean.startswith(header) and len(line_clean) <= len(header) + 10:
                return section_type
    
    return None


def extract_section_content(text: str, section_name: str) -> str:
    """
    Extract content from a specific section of resume text.
    
    Convenience function to get just one section from the normalized text.
    
    Args:
        text (str): Raw resume text
        section_name (str): Section to extract ('experience', 'education', etc.)
        
    Returns:
        str: Content from the specified section
    """
    sections = normalize_text(text)
    return sections.get(section_name, '')


def get_section_lines(section_text: str) -> List[str]:
    """
    Split section text into meaningful lines, removing empty ones.
    
    Useful for processing bullet points in experience sections
    or course lists in education sections.
    
    Args:
        section_text (str): Text from a specific section
        
    Returns:
        List[str]: Non-empty lines from the section
    """
    if not section_text:
        return []
    
    lines = [line.strip() for line in section_text.split('\n')]
    return [line for line in lines if line]
