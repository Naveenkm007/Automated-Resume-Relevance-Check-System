"""
Semantic Matching Module

This package provides semantic similarity scoring and LLM-based feedback
for resume-job description matching using embedding models and language models.

Components:
- embeddings: Text embedding generation and vector storage
- similarity: Semantic similarity computation using cosine similarity
- feedback: LLM-powered personalized improvement suggestions
- combined_score: Integration of hard matching and semantic scores

Author: Cascade AI
"""

from .embeddings import get_embedding, embed_and_index
from .similarity import compute_semantic_score
from .feedback import generate_feedback
from .combined_score import compute_combined_score

__version__ = "1.0.0"
__all__ = [
    "get_embedding",
    "embed_and_index", 
    "compute_semantic_score",
    "generate_feedback",
    "compute_combined_score"
]
