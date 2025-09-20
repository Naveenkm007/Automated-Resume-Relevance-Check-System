"""
Hard Match Scoring Module

This module implements deterministic, keyword-based scoring between parsed job descriptions
and parsed resumes. It uses exact keyword matching, fuzzy string matching, and TF-IDF
similarity to compute compatibility scores.

Hard matching is useful for:
- Initial filtering of candidates
- Checking mandatory requirements
- Compliance with specific skill requirements
- Fast, interpretable scoring

However, it has limitations:
- Cannot understand context or synonyms well
- May miss qualified candidates with different terminology
- Vulnerable to keyword stuffing
- Doesn't capture soft skills or cultural fit
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter
import math

# Required libraries for TF-IDF and fuzzy matching
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - TF-IDF similarity disabled")

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning("RapidFuzz not available - fuzzy matching disabled")

# Default scoring weights - can be customized
DEFAULT_WEIGHTS = {
    'must_have_skills': 0.50,    # 50% - Must-have skills are critical
    'good_to_have_skills': 0.20, # 20% - Nice-to-have skills add value
    'education_match': 0.10,     # 10% - Education requirements
    'certifications_match': 0.05, # 5% - Professional certifications
    'tfidf_similarity': 0.15     # 15% - Overall text similarity
}


def compute_keyword_score(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                         weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Compute deterministic keyword-based score between resume and job description.
    
    This function performs exact and fuzzy matching of skills, education, and certifications
    between a parsed resume and job description. It's designed for initial filtering
    and compliance checking where specific requirements must be met.
    
    The scoring approach:
    1. Must-have skills: Critical requirements that significantly impact score
    2. Good-to-have skills: Desirable but not mandatory skills
    3. Education matching: Degree level and field requirements
    4. Certification matching: Professional credentials
    5. TF-IDF similarity: Overall textual similarity between documents
    
    Args:
        resume_struct (Dict): Parsed resume structure with keys:
            - 'skills': List of candidate skills
            - 'education': List of education entries
            - 'certifications': List of certifications (optional)
            - 'full_text': Complete resume text (for TF-IDF)
        jd_struct (Dict): Parsed job description structure with keys:
            - 'must_have_skills': List of required skills
            - 'good_to_have_skills': List of preferred skills  
            - 'education_requirements': Education criteria
            - 'certifications_required': Required certifications (optional)
            - 'full_text': Complete JD text (for TF-IDF)
            
    Returns:
        Dict[str, Any]: Scoring results containing:
            - skill_matches: Detailed skill matching results
            - education_match: Boolean education compatibility
            - certifications_match: List of matched certifications
            - tfidf_score: Text similarity score (0-1)
            - raw_score: Final normalized score (0-100)
            - breakdown: Score component breakdown for transparency
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Handle None inputs
    if resume_struct is None:
        resume_struct = {}
    if jd_struct is None:
        jd_struct = {}
    
    # Initialize result structure
    result = {
        'skill_matches': {
            'must_have': 0,
            'good_to_have': 0, 
            'missing_must': [],
            'missing_good': [],
            'matched_must': [],
            'matched_good': []
        },
        'education_match': False,
        'certifications_match': [],
        'tfidf_score': 0.0,
        'raw_score': 0.0,
        'breakdown': {}
    }
    
    # Extract data with safe defaults and type checking
    resume_skills_raw = resume_struct.get('skills', [])
    if not isinstance(resume_skills_raw, list):
        resume_skills_raw = []
    resume_skills = set(skill.lower().strip() for skill in resume_skills_raw if isinstance(skill, str))
    
    jd_must_have_raw = jd_struct.get('must_have_skills', [])
    if not isinstance(jd_must_have_raw, list):
        jd_must_have_raw = []
    jd_must_have = set(skill.lower().strip() for skill in jd_must_have_raw if isinstance(skill, str))
    
    jd_good_to_have_raw = jd_struct.get('good_to_have_skills', [])
    if not isinstance(jd_good_to_have_raw, list):
        jd_good_to_have_raw = []
    jd_good_to_have = set(skill.lower().strip() for skill in jd_good_to_have_raw if isinstance(skill, str))
    
    # 1. SKILL MATCHING - Core component of hard matching
    skill_score, skill_details = _compute_skill_matches(
        resume_skills, jd_must_have, jd_good_to_have, weights
    )
    result['skill_matches'].update(skill_details)
    
    # 2. EDUCATION MATCHING - Check degree and field requirements
    education_score = _compute_education_match(
        resume_struct.get('education', []), 
        jd_struct.get('education_requirements', {}),
        weights['education_match']
    )
    result['education_match'] = education_score > 0
    
    # 3. CERTIFICATION MATCHING - Professional credentials
    cert_score, matched_certs = _compute_certification_match(
        resume_struct.get('certifications', []),
        jd_struct.get('certifications_required', []),
        weights['certifications_match']
    )
    result['certifications_match'] = matched_certs
    
    # 4. TF-IDF SIMILARITY - Overall document similarity
    tfidf_score = tfidf_similarity(
        resume_struct.get('full_text', ''),
        jd_struct.get('full_text', '')
    )
    result['tfidf_score'] = tfidf_score
    tfidf_weighted = tfidf_score * weights['tfidf_similarity'] * 100
    
    # 5. COMBINE SCORES
    total_score = skill_score + education_score + cert_score + tfidf_weighted
    result['raw_score'] = min(100.0, max(0.0, total_score))  # Clamp to 0-100
    
    # 6. BREAKDOWN FOR TRANSPARENCY
    result['breakdown'] = {
        'skill_score': skill_score,
        'education_score': education_score, 
        'certification_score': cert_score,
        'tfidf_score': tfidf_weighted,
        'total': result['raw_score'],
        'weights_used': weights.copy()
    }
    
    return result


def _compute_skill_matches(resume_skills: Set[str], must_have: Set[str], 
                          good_to_have: Set[str], weights: Dict[str, float]) -> Tuple[float, Dict]:
    """
    Compute skill matching scores using exact and fuzzy matching.
    
    This function performs both exact string matching and fuzzy matching to handle
    variations in skill naming (e.g., "JS" vs "JavaScript", "ML" vs "Machine Learning").
    
    Fuzzy matching helps with:
    - Abbreviations and synonyms
    - Typos and formatting differences
    - Similar skill names with minor variations
    
    But it can fail with:
    - Completely different terms for same concept
    - Context-dependent meanings
    - Skills that sound similar but are different
    
    Args:
        resume_skills: Set of normalized candidate skills
        must_have: Set of required skills from JD
        good_to_have: Set of preferred skills from JD
        weights: Scoring weight configuration
        
    Returns:
        Tuple[float, Dict]: (total_skill_score, detailed_matches)
    """
    details = {
        'must_have': 0,
        'good_to_have': 0,
        'missing_must': [],
        'missing_good': [],
        'matched_must': [],
        'matched_good': []
    }
    
    # MUST-HAVE SKILLS MATCHING
    must_score = 0.0
    if must_have:
        # Exact matches first
        exact_must_matches = resume_skills.intersection(must_have)
        
        # Fuzzy matching for remaining skills
        remaining_must = must_have - exact_must_matches
        fuzzy_must_matches = set()
        
        if remaining_must and RAPIDFUZZ_AVAILABLE:
            for required_skill in remaining_must:
                best_match = fuzzy_match_skill(required_skill, resume_skills)
                if best_match:
                    fuzzy_must_matches.add(best_match)
        
        # Calculate must-have score
        total_must_matches = len(exact_must_matches) + len(fuzzy_must_matches)
        must_have_ratio = total_must_matches / len(must_have)
        must_score = must_have_ratio * weights['must_have_skills'] * 100
        
        # Update details
        details['must_have'] = total_must_matches
        details['matched_must'] = list(exact_must_matches) + list(fuzzy_must_matches)
        details['missing_must'] = list(must_have - exact_must_matches - fuzzy_must_matches)
    
    # GOOD-TO-HAVE SKILLS MATCHING
    good_score = 0.0
    if good_to_have:
        # Exact matches first
        exact_good_matches = resume_skills.intersection(good_to_have)
        
        # Fuzzy matching for remaining skills
        remaining_good = good_to_have - exact_good_matches
        fuzzy_good_matches = set()
        
        if remaining_good and RAPIDFUZZ_AVAILABLE:
            for preferred_skill in remaining_good:
                best_match = fuzzy_match_skill(preferred_skill, resume_skills)
                if best_match:
                    fuzzy_good_matches.add(best_match)
        
        # Calculate good-to-have score
        total_good_matches = len(exact_good_matches) + len(fuzzy_good_matches)
        good_to_have_ratio = total_good_matches / len(good_to_have)
        good_score = good_to_have_ratio * weights['good_to_have_skills'] * 100
        
        # Update details
        details['good_to_have'] = total_good_matches
        details['matched_good'] = list(exact_good_matches) + list(fuzzy_good_matches)
        details['missing_good'] = list(good_to_have - exact_good_matches - fuzzy_good_matches)
    
    return must_score + good_score, details


def fuzzy_match_skill(target_skill: str, skill_pool: Set[str], threshold: int = 80) -> Optional[str]:
    """
    Find the best fuzzy match for a skill in a pool of skills.
    
    Uses RapidFuzz for efficient fuzzy string matching. This helps match skills
    that are semantically similar but textually different.
    
    Common cases where fuzzy matching helps:
    - "javascript" vs "js" 
    - "machine learning" vs "ml"
    - "postgresql" vs "postgres"
    - Minor typos or formatting differences
    
    Limitations:
    - May match unrelated skills that sound similar
    - Threshold tuning is domain-specific
    - Cannot understand semantic relationships
    
    Args:
        target_skill: Skill to find matches for
        skill_pool: Set of available skills to match against
        threshold: Minimum similarity score (0-100)
        
    Returns:
        Best matching skill from pool, or None if no good match
    """
    if not RAPIDFUZZ_AVAILABLE or not skill_pool:
        return None
    
    # Find best match using ratio (balanced speed/accuracy)
    result = process.extractOne(
        target_skill, 
        skill_pool, 
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )
    
    return result[0] if result else None


def _compute_education_match(resume_education: List[Dict], education_req: Dict, weight: float) -> float:
    """
    Check if resume meets education requirements.
    
    Education matching is often binary (meets requirement or not) but can be
    more nuanced based on:
    - Degree level (Bachelor's, Master's, PhD)
    - Field of study relevance
    - Institution prestige (if specified)
    - Graduation recency
    
    This implementation does basic level and field matching. More sophisticated
    matching could include:
    - Related field mapping (CS -> Software Engineering)
    - Equivalent experience substitution
    - International degree recognition
    
    Args:
        resume_education: List of education entries from resume
        education_req: Education requirements from JD
        weight: Weight for education component
        
    Returns:
        Weighted education score
    """
    if not education_req or not resume_education:
        return 0.0
    
    required_level = education_req.get('level', '').lower()
    required_field = education_req.get('field', '').lower()
    
    # Define degree level hierarchy
    degree_levels = {
        'high school': 1, 'diploma': 1,
        'associate': 2, 'associates': 2,
        'bachelor': 3, 'bachelors': 3, 'bs': 3, 'ba': 3, 'be': 3, 'btech': 3,
        'master': 4, 'masters': 4, 'ms': 4, 'ma': 4, 'mtech': 4, 'mba': 4,
        'phd': 5, 'doctorate': 5, 'doctoral': 5
    }
    
    required_level_num = degree_levels.get(required_level, 0)
    
    # Check each education entry
    for edu in resume_education:
        degree = edu.get('degree', '').lower()
        field = edu.get('stream', '') or edu.get('field', '')
        
        # Extract degree level from degree string
        candidate_level_num = 0
        for level_name, level_num in degree_levels.items():
            if level_name in degree:
                candidate_level_num = max(candidate_level_num, level_num)
        
        # Check if degree meets level requirement
        level_match = candidate_level_num >= required_level_num
        
        # Check field match (if specified)
        field_match = True
        if required_field:
            field_match = (
                required_field in field.lower() or
                _is_related_field(required_field, field.lower())
            )
        
        # If both level and field match, return full score
        if level_match and field_match:
            return weight * 100
    
    return 0.0


def _is_related_field(required_field: str, candidate_field: str) -> bool:
    """
    Check if candidate field is related to required field.
    
    This is a simplified implementation. A production system might use:
    - Ontology mapping of academic fields
    - Industry-specific field relationships
    - Machine learning-based field similarity
    """
    field_mappings = {
        'computer science': ['cs', 'cse', 'computer engineering', 'software engineering', 'it', 'information technology'],
        'engineering': ['computer science', 'electrical', 'mechanical', 'civil', 'chemical'],
        'business': ['mba', 'management', 'economics', 'finance', 'marketing'],
        'data science': ['statistics', 'mathematics', 'computer science', 'analytics'],
    }
    
    related_fields = field_mappings.get(required_field, [])
    return any(related in candidate_field for related in related_fields)


def _compute_certification_match(resume_certs: List[str], required_certs: List[str], weight: float) -> Tuple[float, List[str]]:
    """
    Match certifications between resume and job requirements.
    
    Certification matching is typically more exact than skill matching since
    certification names are standardized. However, we still use fuzzy matching
    to handle:
    - Different formatting ("AWS Certified" vs "AWS-Certified")
    - Abbreviations vs full names
    - Version differences ("CISSP 2021" vs "CISSP")
    
    Args:
        resume_certs: List of certifications from resume
        required_certs: List of required certifications from JD
        weight: Weight for certification component
        
    Returns:
        Tuple of (weighted_score, matched_certifications)
    """
    if not required_certs:
        return 0.0, []
    
    resume_certs_lower = [cert.lower().strip() for cert in resume_certs]
    required_certs_lower = [cert.lower().strip() for cert in required_certs]
    
    matched_certs = []
    
    for required_cert in required_certs_lower:
        # Check for exact match first
        if required_cert in resume_certs_lower:
            matched_certs.append(required_cert)
        # Check for fuzzy match
        elif RAPIDFUZZ_AVAILABLE:
            best_match = fuzzy_match_skill(required_cert, set(resume_certs_lower), threshold=85)
            if best_match:
                matched_certs.append(best_match)
    
    # Calculate score
    if required_certs:
        match_ratio = len(matched_certs) / len(required_certs)
        score = match_ratio * weight * 100
    else:
        score = 0.0
    
    return score, matched_certs


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute TF-IDF cosine similarity between resume and job description texts.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) measures how important
    words are in documents relative to a collection of documents. Cosine similarity
    then measures the angle between document vectors, giving us semantic similarity.
    
    Why TF-IDF is useful:
    - Captures overall document similarity beyond exact keyword matching
    - Reduces impact of common words (the, and, for, etc.)
    - Considers word frequency and rarity
    - Works well for technical documents with domain-specific vocabulary
    
    When TF-IDF fails:
    - Documents with completely different vocabulary for same concepts
    - Very short documents (limited context)
    - Documents with different languages or writing styles
    - Doesn't understand semantics (cat vs feline won't match)
    
    Mitigation strategies:
    - Combine with semantic similarity (word embeddings)
    - Use domain-specific preprocessing
    - Adjust n-gram range for better context capture
    - Apply lemmatization/stemming for word normalization
    
    Args:
        resume_text (str): Full text content of resume
        jd_text (str): Full text content of job description
        
    Returns:
        float: Cosine similarity score between 0 and 1
               0 = completely different, 1 = identical
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("scikit-learn not available - returning 0 for TF-IDF similarity")
        return 0.0
    
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    
    try:
        # Create TF-IDF vectorizer with optimized parameters for resumes/JDs
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',  # Remove common English words
            ngram_range=(1, 2),    # Include single words and bigrams
            max_features=5000,     # Limit vocabulary size for efficiency
            min_df=1,              # Keep rare terms (important for technical skills)
            max_df=0.95            # Remove very common terms
        )
        
        # Fit and transform both documents
        documents = [resume_text, jd_text]
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0]
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, float(similarity_score)))
        
    except Exception as e:
        logging.error(f"TF-IDF similarity computation failed: {e}")
        return 0.0


def fuzzy_match_score(list_a: List[str], list_b: List[str], threshold: int = 80) -> float:
    """
    Compute fuzzy matching score between two lists of strings.
    
    This function is useful for matching job titles, skill variations, or any
    string lists where exact matching might miss valid matches due to formatting
    differences, abbreviations, or minor typos.
    
    Uses RapidFuzz for efficient fuzzy string matching with multiple algorithms:
    - Ratio: Balanced speed/accuracy (default)
    - Partial ratio: Good for substring matching
    - Token sort: Handles word order differences
    - Token set: Handles duplicate words and order
    
    When fuzzy matching helps:
    - Job titles: "Software Engineer" vs "Software Developer"
    - Skills: "JavaScript" vs "JS", "PostgreSQL" vs "Postgres"
    - Company names: "Google Inc." vs "Google LLC"
    - Location names: "New York" vs "NYC"
    
    When fuzzy matching fails:
    - Semantically similar but textually different terms
    - Different languages for same concept
    - Context-dependent terms with multiple meanings
    - Very short strings (high chance of false positives)
    
    Args:
        list_a (List[str]): First list of strings
        list_b (List[str]): Second list of strings  
        threshold (int): Minimum similarity score (0-100) to consider a match
        
    Returns:
        float: Average fuzzy match score (0-1 range)
               0 = no matches, 1 = all perfect matches
    """
    if not RAPIDFUZZ_AVAILABLE:
        logging.warning("RapidFuzz not available - using exact matching only")
        return _exact_match_score(list_a, list_b)
    
    if not list_a or not list_b:
        return 0.0
    
    total_score = 0.0
    match_count = 0
    
    # For each item in list_a, find best match in list_b
    for item_a in list_a:
        if not item_a.strip():
            continue
            
        best_match = process.extractOne(
            item_a.strip(),
            [item.strip() for item in list_b if item.strip()],
            scorer=fuzz.ratio,
            score_cutoff=threshold
        )
        
        if best_match:
            # Convert RapidFuzz score (0-100) to 0-1 range
            normalized_score = best_match[1] / 100.0
            total_score += normalized_score
            match_count += 1
    
    # Return average match score
    if match_count > 0:
        return total_score / len(list_a)
    else:
        return 0.0


def _exact_match_score(list_a: List[str], list_b: List[str]) -> float:
    """Fallback exact matching when RapidFuzz is not available."""
    if not list_a or not list_b:
        return 0.0
    
    set_a = set(item.lower().strip() for item in list_a if item.strip())
    set_b = set(item.lower().strip() for item in list_b if item.strip())
    
    if not set_a:
        return 0.0
    
    matches = len(set_a.intersection(set_b))
    return matches / len(set_a)


# FAILURE MODES AND MITIGATION STRATEGIES
"""
Common failure modes of hard matching and mitigation strategies:

1. SYNONYM/ABBREVIATION MISSES
   Problem: "JS" vs "JavaScript", "ML" vs "Machine Learning"
   Mitigation: 
   - Expand skill normalization dictionary in utils.py
   - Use fuzzy matching with appropriate thresholds
   - Implement semantic similarity as fallback

2. CONTEXT INSENSITIVITY  
   Problem: "Python" (programming) vs "Python" (snake)
   Mitigation:
   - Use context-aware matching (check surrounding words)
   - Maintain domain-specific skill vocabularies
   - Consider document sections when extracting skills

3. KEYWORD STUFFING VULNERABILITY
   Problem: Candidates may add irrelevant keywords to game the system
   Mitigation:
   - Combine with semantic similarity scores
   - Check for keyword context and usage patterns
   - Use experience validation and depth assessment

4. SKILL LEVEL GRANULARITY
   Problem: "Beginner Python" vs "Expert Python" both match "Python"
   Mitigation:
   - Extract and consider proficiency levels
   - Weight skills by experience duration
   - Use project complexity as skill level indicator

5. RAPIDLY EVOLVING TERMINOLOGY
   Problem: New frameworks, tools, and methodologies not in mappings
   Mitigation:
   - Regular updates to skill normalization dictionaries
   - Community-driven skill ontology maintenance
   - Machine learning-based skill relationship discovery

6. CULTURAL/REGIONAL VARIATIONS
   Problem: Different regions use different terms for same roles/skills
   Mitigation:
   - Maintain regional skill/title mappings
   - Use location-aware normalization
   - Include cultural context in matching algorithms
"""
