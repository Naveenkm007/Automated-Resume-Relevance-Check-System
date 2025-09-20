"""
Resume Parser Package

A comprehensive resume parsing system that extracts structured information
from PDF and DOCX resume files using NLP techniques.

Author: Cascade AI
"""

from .extract import extract_text_from_pdf, extract_text_from_docx
from .cleaner import normalize_text
from .ner import extract_entities
from .utils import normalize_skill, extract_email, extract_phone

__version__ = "1.0.0"
__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx", 
    "normalize_text",
    "extract_entities",
    "normalize_skill",
    "extract_email",
    "extract_phone"
]
