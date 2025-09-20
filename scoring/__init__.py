"""
Scoring Module

This package contains various scoring algorithms for matching resumes to job descriptions.
Includes hard matching (keyword-based), soft matching (semantic), and hybrid approaches.

Author: Cascade AI
"""

from .hard_match import compute_keyword_score, tfidf_similarity, fuzzy_match_score

__version__ = "1.0.0"
__all__ = [
    "compute_keyword_score",
    "tfidf_similarity", 
    "fuzzy_match_score"
]
