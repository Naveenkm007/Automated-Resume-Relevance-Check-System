"""
Combined Scoring Module

This module combines hard matching scores (keyword-based) with semantic scores
(embedding-based) to provide a comprehensive resume-job match assessment.

Why combine hard and semantic scoring:
- Hard matching catches specific requirements (exact skills, certifications)
- Semantic matching captures contextual fit and transferable skills
- Together they provide both precision and recall in candidate evaluation
- Reduces false negatives from keyword-only matching
- Adds interpretability through multiple scoring dimensions

Scoring formula:
final_score = hard_score * 0.6 + semantic_score * 0.4

The 60/40 weighting reflects that:
- Specific requirements (hard) are often non-negotiable (higher weight)
- Semantic fit (soft) indicates potential and cultural alignment (lower weight)
- This ratio can be customized based on role requirements and company priorities

Score interpretation:
- High (80-100): Excellent match, strong candidate
- Medium (50-79): Good match, worth interviewing
- Low (0-49): Poor match, likely not suitable
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .similarity import compute_semantic_score, find_semantic_matches, analyze_semantic_coverage


# Default weights for combining scores
DEFAULT_HARD_WEIGHT = 0.6
DEFAULT_SEMANTIC_WEIGHT = 0.4

# Score thresholds for categorization
SCORE_THRESHOLDS = {
    'high': 80,
    'medium': 50,
    'low': 0
}


def compute_combined_score(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                          hard_score_result: Dict[str, Any], 
                          hard_weight: float = DEFAULT_HARD_WEIGHT,
                          semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT) -> Dict[str, Any]:
    """
    Compute combined score from hard matching and semantic similarity.
    
    This function integrates multiple scoring approaches to provide a comprehensive
    assessment that balances specific requirements with overall fit.
    
    The combination approach:
    1. Takes proven hard matching score (keyword-based)
    2. Computes semantic similarity score (embedding-based)  
    3. Combines using weighted average
    4. Provides verdict and detailed breakdown
    5. Identifies top semantic matches for explanation
    
    Benefits of combined scoring:
    - Reduces keyword stuffing vulnerability
    - Captures transferable skills and potential
    - Provides multiple evidence sources for decisions
    - Enables fine-tuned weighting for different roles
    - Improves candidate experience through better matching
    
    Args:
        resume_struct (Dict): Parsed resume structure
        jd_struct (Dict): Parsed job description structure
        hard_score_result (Dict): Result from hard matching scoring
        hard_weight (float): Weight for hard matching score (0-1)
        semantic_weight (float): Weight for semantic score (0-1)
        
    Returns:
        Dict[str, Any]: Combined scoring result with:
            - final_score: Combined score (0-100)
            - verdict: 'high'/'medium'/'low' match quality
            - hard_score: Original hard matching score
            - semantic_score: Semantic similarity score
            - missing_elements: Critical gaps identified
            - top_semantic_matches: Best matching resume content
            - score_breakdown: Detailed component analysis
            - recommendations: Next steps based on score
    """
    # Validate weights
    if abs(hard_weight + semantic_weight - 1.0) > 0.001:
        logging.warning(f"Weights don't sum to 1.0: {hard_weight} + {semantic_weight} = {hard_weight + semantic_weight}")
        # Normalize weights
        total_weight = hard_weight + semantic_weight
        hard_weight = hard_weight / total_weight
        semantic_weight = semantic_weight / total_weight
    
    try:
        # Extract hard score
        hard_score = hard_score_result.get('raw_score', 0.0)
        
        # Compute semantic score
        resume_text = resume_struct.get('full_text', '')
        jd_text = jd_struct.get('full_text', '')
        semantic_score = compute_semantic_score(resume_text, jd_text)
        
        # Combine scores
        final_score = round(hard_score * hard_weight + semantic_score * semantic_weight)
        
        # Determine verdict
        verdict = _get_score_verdict(final_score)
        
        # Identify missing elements (from hard matching)
        missing_elements = _extract_missing_elements(hard_score_result)
        
        # Find top semantic matches for explanation
        top_matches = _get_top_semantic_matches(resume_struct, jd_struct)
        
        # Create detailed breakdown
        breakdown = _create_score_breakdown(
            hard_score, semantic_score, final_score, 
            hard_weight, semantic_weight, hard_score_result
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(verdict, missing_elements, breakdown)
        
        result = {
            'final_score': final_score,
            'verdict': verdict,
            'hard_score': hard_score,
            'semantic_score': semantic_score,
            'missing_elements': missing_elements,
            'top_semantic_matches': top_matches,
            'score_breakdown': breakdown,
            'recommendations': recommendations,
            'weights_used': {
                'hard_weight': hard_weight,
                'semantic_weight': semantic_weight
            }
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Combined scoring failed: {e}")
        return _get_fallback_result(hard_score_result)


def _get_score_verdict(score: float) -> str:
    """
    Convert numerical score to categorical verdict.
    
    Score ranges:
    - High (80-100): Strong match, recommend for interview
    - Medium (50-79): Moderate match, consider for interview  
    - Low (0-49): Weak match, likely not suitable
    
    These thresholds can be adjusted based on:
    - Market conditions (tight vs loose talent market)
    - Role criticality (senior vs junior positions)
    - Company standards (startup vs enterprise hiring bars)
    - Diversity and inclusion goals
    """
    if score >= SCORE_THRESHOLDS['high']:
        return 'high'
    elif score >= SCORE_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'low'


def _extract_missing_elements(hard_score_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract critical missing elements from hard matching results.
    
    Missing elements help prioritize what candidates should focus on
    and what hiring managers should probe during interviews.
    
    Returns:
        List of missing elements with type, items, and priority
    """
    missing = []
    
    skill_matches = hard_score_result.get('skill_matches', {})
    
    # Missing must-have skills (highest priority)
    missing_must = skill_matches.get('missing_must', [])
    if missing_must:
        missing.append({
            'type': 'must_have_skills',
            'items': missing_must,
            'priority': 'critical',
            'impact': 'High - these are required skills for the role'
        })
    
    # Missing good-to-have skills (medium priority)
    missing_good = skill_matches.get('missing_good', [])
    if missing_good:
        missing.append({
            'type': 'good_to_have_skills',
            'items': missing_good[:5],  # Limit to top 5
            'priority': 'moderate',
            'impact': 'Medium - these would strengthen the candidacy'
        })
    
    # Education mismatch (if applicable)
    if not hard_score_result.get('education_match', True):
        missing.append({
            'type': 'education_requirement',
            'items': ['Education requirement not met'],
            'priority': 'moderate',
            'impact': 'Medium - may need additional verification or consideration'
        })
    
    # Missing certifications
    required_certs = len(hard_score_result.get('certifications_match', []))
    if required_certs == 0:  # Assuming there were requirements but no matches
        missing.append({
            'type': 'certifications',
            'items': ['Required certifications missing'],
            'priority': 'moderate',
            'impact': 'Medium - professional credentials may be needed'
        })
    
    return missing


def _get_top_semantic_matches(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                             top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find top semantic matches between resume and job description.
    
    These matches help explain why the semantic score is high/low and
    provide evidence for hiring decisions or candidate feedback.
    
    Args:
        resume_struct: Parsed resume
        jd_struct: Parsed job description
        top_k: Number of top matches to return
        
    Returns:
        List of top semantic matches with context
    """
    try:
        resume_text = resume_struct.get('full_text', '')
        jd_text = jd_struct.get('full_text', '')
        
        if not resume_text or not jd_text:
            return []
        
        # Split JD into requirements/sentences for matching
        jd_sentences = _extract_jd_requirements(jd_text)
        
        if not jd_sentences:
            return [{'resume_text': 'Full resume', 'jd_match': 'Full job description', 
                    'score': compute_semantic_score(resume_text, jd_text)}]
        
        # Find semantic matches
        matches = find_semantic_matches(resume_text, jd_sentences, top_k)
        
        return matches
        
    except Exception as e:
        logging.error(f"Failed to find semantic matches: {e}")
        return []


def _extract_jd_requirements(jd_text: str) -> List[str]:
    """
    Extract key requirements/sentences from job description.
    
    Focuses on requirements, responsibilities, and qualifications
    rather than company background or benefits.
    """
    if not jd_text:
        return []
    
    # Split into sentences and filter for requirements
    sentences = []
    lines = jd_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 20:  # Skip very short lines
            continue
        
        # Look for requirement indicators
        requirement_indicators = [
            'experience', 'skill', 'knowledge', 'proficiency', 'familiar',
            'required', 'must have', 'should have', 'preferred', 'ideal',
            'responsible', 'duties', 'develop', 'maintain', 'implement'
        ]
        
        if any(indicator in line.lower() for indicator in requirement_indicators):
            sentences.append(line)
    
    # If no specific requirements found, use all substantial sentences
    if not sentences:
        for line in lines:
            line = line.strip()
            if len(line) >= 30:  # Longer sentences likely contain requirements
                sentences.append(line)
    
    return sentences[:20]  # Limit to avoid excessive processing


def _create_score_breakdown(hard_score: float, semantic_score: float, final_score: float,
                           hard_weight: float, semantic_weight: float, 
                           hard_score_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create detailed score breakdown for transparency and debugging.
    
    Provides insights into:
    - How the final score was calculated
    - Contribution of each component
    - Performance in different areas
    - Potential areas for improvement
    """
    # Hard score component breakdown
    hard_breakdown = hard_score_result.get('breakdown', {})
    
    breakdown = {
        'final_calculation': {
            'hard_contribution': round(hard_score * hard_weight, 1),
            'semantic_contribution': round(semantic_score * semantic_weight, 1),
            'final_score': final_score,
            'formula': f"{hard_score} * {hard_weight} + {semantic_score} * {semantic_weight}"
        },
        'component_scores': {
            'hard_matching': {
                'total': hard_score,
                'skills': hard_breakdown.get('skill_score', 0),
                'education': hard_breakdown.get('education_score', 0),
                'certification': hard_breakdown.get('certification_score', 0),
                'tfidf': hard_breakdown.get('tfidf_score', 0)
            },
            'semantic_matching': {
                'total': semantic_score,
                'explanation': _get_semantic_score_explanation(semantic_score)
            }
        },
        'performance_analysis': {
            'strengths': _identify_strengths(hard_breakdown, semantic_score),
            'weaknesses': _identify_weaknesses(hard_breakdown, semantic_score),
            'balance': _analyze_score_balance(hard_score, semantic_score)
        }
    }
    
    return breakdown


def _get_semantic_score_explanation(semantic_score: float) -> str:
    """Provide explanation for semantic score level."""
    if semantic_score >= 90:
        return "Excellent semantic alignment - very strong conceptual match"
    elif semantic_score >= 80:
        return "Very good semantic alignment - strong conceptual overlap"
    elif semantic_score >= 70:
        return "Good semantic alignment - solid conceptual match"
    elif semantic_score >= 60:
        return "Moderate semantic alignment - some conceptual overlap"
    elif semantic_score >= 40:
        return "Fair semantic alignment - limited conceptual match"
    elif semantic_score >= 20:
        return "Poor semantic alignment - minimal conceptual overlap"
    else:
        return "Very poor semantic alignment - little conceptual similarity"


def _identify_strengths(hard_breakdown: Dict, semantic_score: float) -> List[str]:
    """Identify candidate's main strengths based on scoring."""
    strengths = []
    
    # Check hard matching components
    if hard_breakdown.get('skill_score', 0) >= 70:
        strengths.append("Strong skill match with job requirements")
    
    if hard_breakdown.get('education_score', 0) >= 80:
        strengths.append("Education background aligns well with requirements")
    
    if hard_breakdown.get('certification_score', 0) >= 80:
        strengths.append("Has relevant professional certifications")
    
    # Check semantic alignment
    if semantic_score >= 80:
        strengths.append("Excellent semantic fit - demonstrates deep domain understanding")
    elif semantic_score >= 70:
        strengths.append("Good semantic fit - shows relevant experience and context")
    
    # General strengths
    if not strengths:
        strengths.append("Shows potential for role based on available information")
    
    return strengths


def _identify_weaknesses(hard_breakdown: Dict, semantic_score: float) -> List[str]:
    """Identify areas for improvement based on scoring."""
    weaknesses = []
    
    # Check hard matching components
    if hard_breakdown.get('skill_score', 0) < 50:
        weaknesses.append("Missing several key technical skills")
    
    if hard_breakdown.get('education_score', 0) < 50:
        weaknesses.append("Education background may not fully align with requirements")
    
    if hard_breakdown.get('certification_score', 0) < 30:
        weaknesses.append("Lacks relevant professional certifications")
    
    # Check semantic alignment
    if semantic_score < 50:
        weaknesses.append("Limited semantic alignment - may need more relevant experience")
    elif semantic_score < 70:
        weaknesses.append("Moderate semantic alignment - could strengthen domain expertise")
    
    return weaknesses


def _analyze_score_balance(hard_score: float, semantic_score: float) -> str:
    """Analyze the balance between hard and semantic scores."""
    diff = abs(hard_score - semantic_score)
    
    if diff < 10:
        return "Well-balanced profile - consistent performance across skill and semantic matching"
    elif hard_score > semantic_score + 15:
        return "Skills-heavy profile - has required skills but may lack domain depth"
    elif semantic_score > hard_score + 15:
        return "Context-heavy profile - strong domain understanding but missing specific skills"
    else:
        return "Moderately balanced profile - some variation between skill and semantic scores"


def _generate_recommendations(verdict: str, missing_elements: List[Dict], 
                            breakdown: Dict) -> List[str]:
    """
    Generate actionable recommendations based on combined scoring.
    
    Recommendations guide next steps for different stakeholders:
    - Hiring managers: Interview focus areas, additional screening
    - Candidates: Improvement areas, skill development
    - Recruiters: Sourcing priorities, candidate coaching
    """
    recommendations = []
    
    if verdict == 'high':
        recommendations.extend([
            "Strong candidate - recommend for technical interview",
            "Focus interview on cultural fit and advanced technical concepts",
            "Verify depth of experience in key skill areas"
        ])
    
    elif verdict == 'medium':
        recommendations.extend([
            "Moderate candidate - consider for phone screening",
            "Assess transferable skills and learning potential",
            "Evaluate specific experience depth in missing skill areas"
        ])
        
        # Add specific recommendations based on missing elements
        for missing in missing_elements:
            if missing['priority'] == 'critical':
                recommendations.append(f"Address critical gap: {', '.join(missing['items'][:3])}")
    
    else:  # low
        recommendations.extend([
            "Weak candidate - likely not suitable for current role",
            "Consider for junior positions or different role types",
            "Significant skill development needed before consideration"
        ])
    
    # Add balance-specific recommendations
    balance_analysis = breakdown.get('performance_analysis', {}).get('balance', '')
    if 'skills-heavy' in balance_analysis.lower():
        recommendations.append("Focus interview on practical application and domain knowledge")
    elif 'context-heavy' in balance_analysis.lower():
        recommendations.append("Assess technical skill depth and hands-on experience")
    
    return recommendations[:5]  # Limit to top 5 recommendations


def _get_fallback_result(hard_score_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return fallback result when combined scoring fails."""
    return {
        'final_score': hard_score_result.get('raw_score', 0),
        'verdict': _get_score_verdict(hard_score_result.get('raw_score', 0)),
        'hard_score': hard_score_result.get('raw_score', 0),
        'semantic_score': 0.0,
        'missing_elements': _extract_missing_elements(hard_score_result),
        'top_semantic_matches': [],
        'score_breakdown': {'error': 'Semantic scoring unavailable'},
        'recommendations': ['Review based on hard matching results only'],
        'weights_used': {'hard_weight': 1.0, 'semantic_weight': 0.0}
    }


# Utility functions for custom weighting
def get_role_specific_weights(role_type: str) -> Tuple[float, float]:
    """
    Get recommended weights for different role types.
    
    Args:
        role_type: Type of role ('technical', 'creative', 'sales', 'management')
        
    Returns:
        Tuple of (hard_weight, semantic_weight)
    """
    weight_configs = {
        'technical': (0.7, 0.3),      # Skills matter more
        'creative': (0.4, 0.6),       # Context and portfolio matter more
        'sales': (0.5, 0.5),          # Balance of skills and communication
        'management': (0.4, 0.6),     # Leadership and experience context
        'entry_level': (0.6, 0.4),    # Skills and education focus
        'senior': (0.5, 0.5),         # Balance of skills and experience depth
        'default': (0.6, 0.4)         # Standard weighting
    }
    
    return weight_configs.get(role_type.lower(), weight_configs['default'])


def analyze_score_sensitivity(resume_struct: Dict, jd_struct: Dict, 
                             hard_score_result: Dict) -> Dict[str, float]:
    """
    Analyze how sensitive the final score is to weight changes.
    
    Helps understand if the scoring is robust or highly dependent
    on the specific weighting scheme used.
    
    Returns:
        Dict mapping weight configurations to final scores
    """
    weight_configs = [
        (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
        (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)
    ]
    
    sensitivity_results = {}
    
    for hard_weight, semantic_weight in weight_configs:
        try:
            result = compute_combined_score(
                resume_struct, jd_struct, hard_score_result,
                hard_weight, semantic_weight
            )
            config_name = f"H{int(hard_weight*100)}_S{int(semantic_weight*100)}"
            sensitivity_results[config_name] = result['final_score']
        except:
            continue
    
    return sensitivity_results
