"""
Semantic Similarity Scoring Module

This module computes semantic similarity between resume and job description texts
using embedding vectors and cosine similarity. Unlike keyword matching, semantic
similarity captures meaning and context, allowing matches between semantically
similar but textually different content.

How semantic similarity works:
1. Convert texts to embedding vectors (dense numerical representations)
2. Compute cosine similarity between vectors (measures angle between them)
3. Map similarity score to 0-100 range for consistency with hard matching

Why cosine similarity:
- Measures similarity regardless of vector magnitude (text length)
- Works well with high-dimensional embedding spaces
- Intuitive interpretation: 1=identical, 0=orthogonal, -1=opposite
- Robust to scaling and normalization differences

Cosine similarity formula:
similarity = (A · B) / (|A| × |B|)
where A and B are embedding vectors, · is dot product, |·| is vector magnitude
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re

from .embeddings import get_embedding, compute_similarity


def compute_semantic_score(resume_text: str, jd_text: str) -> float:
    """
    Compute semantic similarity score between resume and job description.
    
    This function measures how semantically similar the resume is to the job
    description using embedding vectors and cosine similarity. The result
    is mapped to a 0-100 scale for consistency with hard matching scores.
    
    Semantic similarity captures:
    - Conceptual relationships (Python ↔ programming, ML ↔ artificial intelligence)
    - Contextual meaning (experience in vs proficiency with vs expertise in)
    - Domain knowledge overlap (technical concepts, industry terminology)
    - Communication style compatibility (formal vs casual, technical vs business)
    
    What semantic similarity cannot capture:
    - Specific requirements (must have exactly 5 years experience)
    - Quantitative criteria (salary expectations, location preferences)
    - Compliance requirements (specific certifications, security clearances)
    - Cultural fit indicators (team dynamics, company values)
    
    The 0-100 mapping helps with:
    - Consistent scoring across different similarity measures
    - Easy interpretation for non-technical stakeholders  
    - Integration with other scoring components
    - Threshold-based filtering and ranking
    
    Args:
        resume_text (str): Complete resume text content
        jd_text (str): Complete job description text content
        
    Returns:
        float: Semantic similarity score from 0-100
               0 = completely different semantic content
               50 = moderate semantic overlap
               100 = very similar semantic content
               
    Example:
        resume = "Software engineer with Python and machine learning experience"
        jd = "Looking for ML engineer proficient in Python programming"
        score = compute_semantic_score(resume, jd)
        # Expected: ~85-95 (high semantic similarity despite different wording)
    """
    if not resume_text or not jd_text:
        return 0.0
    
    try:
        # Preprocess texts to improve embedding quality
        clean_resume = _preprocess_text_for_embedding(resume_text)
        clean_jd = _preprocess_text_for_embedding(jd_text)
        
        # Compute cosine similarity using embeddings
        cosine_sim = compute_similarity(clean_resume, clean_jd)
        
        # Map cosine similarity [-1, 1] to score [0, 100]
        semantic_score = _cosine_to_score(cosine_sim)
        
        return semantic_score
        
    except Exception as e:
        logging.error(f"Semantic similarity computation failed: {e}")
        return 0.0


def _preprocess_text_for_embedding(text: str) -> str:
    """
    Preprocess text to improve embedding quality.
    
    Preprocessing can significantly improve embedding quality by:
    - Removing noise that doesn't contribute to semantic meaning
    - Normalizing formatting inconsistencies
    - Preserving important semantic content
    - Optimizing for the embedding model's training data
    
    Common preprocessing steps:
    - Remove excessive whitespace and formatting artifacts
    - Normalize bullet points and list formatting
    - Remove email addresses and phone numbers (personal info, not semantic)
    - Keep technical terms and domain-specific vocabulary
    - Preserve sentence structure for context
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned text optimized for embedding generation
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize bullet points and list markers
    text = re.sub(r'[•◦▪▫\-\*]\s*', '• ', text)
    
    # Remove email addresses (they don't add semantic value)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '', text)
    
    # Remove URLs (usually not semantically relevant for job matching)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\bpage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def _cosine_to_score(cosine_similarity: float) -> float:
    """
    Map cosine similarity [-1, 1] to intuitive score [0, 100].
    
    Cosine similarity interpretation:
    - 1.0: Identical semantic content (rare in practice)
    - 0.8-0.99: Very similar content (strong match)
    - 0.6-0.8: Moderately similar content (good match)
    - 0.4-0.6: Some similarity (weak match)
    - 0.2-0.4: Little similarity (poor match)
    - 0.0-0.2: Minimal similarity (very poor match)
    - < 0.0: Dissimilar content (extremely rare)
    
    Mapping strategy:
    We use a nonlinear mapping to make scores more interpretable:
    - High cosine similarities (>0.8) map to high scores (80-100)
    - Medium similarities (0.4-0.8) map to medium scores (40-80)
    - Low similarities (<0.4) map to low scores (0-40)
    
    This creates more separation in the useful range and makes
    the scores align with human intuition about similarity.
    
    Args:
        cosine_similarity (float): Cosine similarity between -1 and 1
        
    Returns:
        float: Mapped score between 0 and 100
    """
    # Clamp input to valid range
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    
    # Handle negative similarities (very rare with good embeddings)
    if cosine_similarity < 0:
        return 0.0
    
    # Nonlinear mapping for better score distribution
    # This makes the scores more intuitive and spreads them better
    if cosine_similarity >= 0.9:
        # Excellent similarity -> 90-100 score
        score = 90 + (cosine_similarity - 0.9) * 100
    elif cosine_similarity >= 0.7:
        # Good similarity -> 70-90 score  
        score = 70 + (cosine_similarity - 0.7) * 100
    elif cosine_similarity >= 0.5:
        # Moderate similarity -> 50-70 score
        score = 50 + (cosine_similarity - 0.5) * 100
    elif cosine_similarity >= 0.3:
        # Weak similarity -> 30-50 score
        score = 30 + (cosine_similarity - 0.3) * 100
    else:
        # Poor similarity -> 0-30 score
        score = cosine_similarity * 100
    
    # Ensure score is in valid range
    return max(0.0, min(100.0, score))


def compute_section_similarities(resume_sections: Dict[str, str], 
                                jd_sections: Dict[str, str]) -> Dict[str, float]:
    """
    Compute semantic similarity for corresponding sections.
    
    Section-wise comparison provides more granular insights:
    - Skills section similarity: How well technical abilities align
    - Experience similarity: Relevance of past work to requirements
    - Education similarity: Academic background alignment
    - Project similarity: Demonstration of relevant capabilities
    
    This helps identify specific strengths and gaps, enabling
    more targeted feedback and decision-making.
    
    Args:
        resume_sections (Dict[str, str]): Resume sections (from cleaner.py)
        jd_sections (Dict[str, str]): JD sections  
        
    Returns:
        Dict[str, float]: Section-wise similarity scores (0-100)
    """
    section_scores = {}
    
    # Common sections to compare
    sections_to_compare = [
        'skills', 'experience', 'education', 'projects',
        'responsibilities', 'requirements', 'qualifications'
    ]
    
    for section in sections_to_compare:
        resume_content = resume_sections.get(section, '')
        jd_content = jd_sections.get(section, '')
        
        if resume_content and jd_content:
            similarity = compute_semantic_score(resume_content, jd_content)
            section_scores[section] = similarity
    
    return section_scores


def find_semantic_matches(resume_text: str, jd_sentences: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find resume sentences that semantically match JD requirements.
    
    This function helps identify which parts of the resume are most
    relevant to specific job requirements, enabling:
    - Highlighting relevant experience in applications
    - Identifying gaps where additional detail would help
    - Understanding which resume sections drive the similarity score
    - Providing evidence for high/low semantic scores
    
    Method:
    1. Split resume into sentences or logical units
    2. Compute semantic similarity between each resume unit and JD sentences
    3. Return top-k matches with scores and context
    
    Args:
        resume_text (str): Complete resume text
        jd_sentences (List[str]): List of JD sentences/requirements to match against
        top_k (int): Number of top matches to return
        
    Returns:
        List[Dict]: Top semantic matches with scores and text
    """
    if not resume_text or not jd_sentences:
        return []
    
    try:
        # Split resume into meaningful units (sentences/bullet points)
        resume_sentences = _extract_resume_sentences(resume_text)
        
        matches = []
        
        # Compare each resume sentence to each JD sentence
        for resume_sent in resume_sentences:
            best_score = 0.0
            best_jd_match = ""
            
            for jd_sent in jd_sentences:
                score = compute_semantic_score(resume_sent, jd_sent)
                if score > best_score:
                    best_score = score
                    best_jd_match = jd_sent
            
            if best_score > 30:  # Only include reasonable matches
                matches.append({
                    'resume_text': resume_sent,
                    'jd_match': best_jd_match,
                    'score': best_score
                })
        
        # Sort by score and return top-k
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k]
        
    except Exception as e:
        logging.error(f"Semantic match finding failed: {e}")
        return []


def _extract_resume_sentences(resume_text: str) -> List[str]:
    """
    Extract meaningful sentences or bullet points from resume text.
    
    Resume text often contains bullet points, short phrases, and
    various formatting that needs special handling compared to
    regular sentence tokenization.
    
    Args:
        resume_text (str): Resume text to split
        
    Returns:
        List[str]: List of meaningful text units
    """
    if not resume_text:
        return []
    
    # Split on bullet points and newlines
    sentences = []
    
    # Split by newlines first
    lines = resume_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:  # Skip very short lines
            continue
        
        # Split bullet points into separate items
        if '•' in line:
            bullet_items = line.split('•')
            for item in bullet_items:
                item = item.strip()
                if len(item) > 10:
                    sentences.append(item)
        else:
            # Regular sentence splitting
            import re
            sent_splits = re.split(r'[.!?]+', line)
            for sent in sent_splits:
                sent = sent.strip()
                if len(sent) > 10:
                    sentences.append(sent)
    
    return sentences


def get_similarity_explanation(similarity_score: float) -> str:
    """
    Provide human-readable explanation of similarity score.
    
    Args:
        similarity_score (float): Similarity score (0-100)
        
    Returns:
        str: Human-readable explanation
    """
    if similarity_score >= 90:
        return "Excellent semantic match - very strong alignment with job requirements"
    elif similarity_score >= 80:
        return "Very good semantic match - strong alignment with most requirements"
    elif similarity_score >= 70:
        return "Good semantic match - solid alignment with key requirements"
    elif similarity_score >= 60:
        return "Moderate semantic match - some alignment but room for improvement"
    elif similarity_score >= 40:
        return "Fair semantic match - limited alignment with requirements"
    elif similarity_score >= 20:
        return "Poor semantic match - minimal alignment with requirements"
    else:
        return "Very poor semantic match - little to no alignment with requirements"


# Batch processing utilities
def compute_batch_similarities(resume_texts: List[str], jd_text: str) -> List[float]:
    """
    Compute semantic similarities for multiple resumes against one JD.
    
    Useful for:
    - Ranking candidates by semantic fit
    - Batch processing applications
    - A/B testing different embedding models
    - Performance optimization with caching
    
    Args:
        resume_texts (List[str]): List of resume texts
        jd_text (str): Job description text
        
    Returns:
        List[float]: Similarity scores for each resume (0-100)
    """
    if not jd_text:
        return [0.0] * len(resume_texts)
    
    scores = []
    for resume_text in resume_texts:
        score = compute_semantic_score(resume_text, jd_text)
        scores.append(score)
    
    return scores


def analyze_semantic_coverage(resume_text: str, jd_requirements: List[str]) -> Dict[str, Any]:
    """
    Analyze how well resume covers JD requirements semantically.
    
    Args:
        resume_text (str): Resume text
        jd_requirements (List[str]): List of JD requirement sentences
        
    Returns:
        Dict: Analysis including coverage scores, gaps, and strengths
    """
    if not resume_text or not jd_requirements:
        return {'coverage_score': 0.0, 'covered_requirements': [], 'gaps': []}
    
    covered = []
    gaps = []
    total_score = 0.0
    
    for requirement in jd_requirements:
        score = compute_semantic_score(resume_text, requirement)
        total_score += score
        
        if score >= 60:  # Good coverage threshold
            covered.append({'requirement': requirement, 'score': score})
        else:
            gaps.append({'requirement': requirement, 'score': score})
    
    avg_coverage = total_score / len(jd_requirements) if jd_requirements else 0.0
    
    return {
        'coverage_score': avg_coverage,
        'covered_requirements': sorted(covered, key=lambda x: x['score'], reverse=True),
        'gaps': sorted(gaps, key=lambda x: x['score']),
        'total_requirements': len(jd_requirements),
        'well_covered': len(covered),
        'poorly_covered': len(gaps)
    }
