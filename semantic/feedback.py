"""
LLM-Powered Feedback Generation Module

This module uses Large Language Models (LLMs) to generate personalized,
actionable improvement suggestions for resume-job matching. Unlike rule-based
feedback, LLMs can understand context, provide nuanced advice, and adapt
to different industries and roles.

How LLM feedback works:
1. Analyze resume structure, content, and scoring breakdown
2. Compare against job requirements and identify specific gaps
3. Generate contextual, actionable suggestions with examples
4. Format advice in a constructive, professional manner

Why use LLMs for feedback:
- Contextual understanding: Can interpret complex requirements and situations
- Personalization: Tailors advice to specific candidate profile and role
- Creativity: Generates diverse, non-repetitive suggestions
- Expertise simulation: Provides advice similar to experienced recruiters
- Natural language: Communicates clearly with candidates

LLM limitations to consider:
- Cost: API calls can add up with high usage
- Latency: 1-3 seconds per request vs instant rule-based feedback
- Consistency: May provide slightly different advice for same input
- Hallucination: Might suggest non-existent tools or companies
- Bias: Could reflect biases present in training data
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json
import time

# Environment configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')  # Default to GPT-3.5 for cost efficiency
MAX_FEEDBACK_REQUESTS_PER_MINUTE = int(os.getenv('MAX_FEEDBACK_REQUESTS_PER_MINUTE', '20'))

# LLM client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available - install with: pip install openai")

# Rate limiting tracking
_last_request_times = []

# Global client instance
_openai_client = None


def generate_feedback(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                     score_breakdown: Dict[str, Any], num_suggestions: int = 3) -> List[Dict[str, str]]:
    """
    Generate personalized improvement suggestions using LLM analysis.
    
    This function takes structured resume and job description data along with
    scoring analysis to produce actionable, specific advice for candidates.
    
    The LLM analyzes:
    - Skill gaps between resume and requirements
    - Experience relevance and presentation
    - Education alignment with role expectations  
    - Missing certifications or credentials
    - Resume structure and content quality
    - Industry-specific best practices
    
    Output format:
    Each suggestion contains:
    - action: One-line actionable advice (what to do)
    - example: One-line concrete example (how to do it)
    - priority: high/medium/low based on impact potential
    - category: skill/experience/education/format for organization
    
    Args:
        resume_struct (Dict): Parsed resume structure from resume parser
        jd_struct (Dict): Parsed job description structure  
        score_breakdown (Dict): Detailed scoring from hard/semantic matching
        num_suggestions (int): Number of suggestions to generate (1-5)
        
    Returns:
        List[Dict]: List of improvement suggestions with actions and examples
        
    Example output:
        [
            {
                "action": "Add cloud computing projects to demonstrate AWS skills",
                "example": "Create a web app deployed on AWS EC2 with RDS database",
                "priority": "high",
                "category": "skill"
            }
        ]
    """
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        logging.warning("OpenAI not available - returning template feedback")
        return _generate_template_feedback(resume_struct, jd_struct, score_breakdown, num_suggestions)
    
    try:
        # Apply rate limiting
        _apply_rate_limiting()
        
        # Prepare context for LLM
        context = _prepare_feedback_context(resume_struct, jd_struct, score_breakdown)
        
        # Generate feedback using LLM
        suggestions = _call_llm_for_feedback(context, num_suggestions)
        
        # Post-process and validate suggestions
        validated_suggestions = _validate_and_format_suggestions(suggestions, num_suggestions)
        
        return validated_suggestions
        
    except Exception as e:
        logging.error(f"LLM feedback generation failed: {e}")
        # Fallback to template-based feedback
        return _generate_template_feedback(resume_struct, jd_struct, score_breakdown, num_suggestions)


def _prepare_feedback_context(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                             score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare structured context for LLM analysis.
    
    This function extracts and organizes the most relevant information
    for generating feedback, while keeping the prompt concise to:
    - Reduce API costs (fewer tokens)
    - Improve response quality (focused context)
    - Ensure consistent output format
    - Handle edge cases gracefully
    """
    # Extract key resume information
    candidate_profile = {
        'name': resume_struct.get('name', 'Candidate'),
        'skills': resume_struct.get('skills', [])[:15],  # Limit to top 15 skills
        'experience_years': _estimate_experience_years(resume_struct.get('experience', [])),
        'education_level': _extract_education_level(resume_struct.get('education', [])),
        'has_relevant_projects': len(resume_struct.get('projects', [])) > 0,
        'certifications': resume_struct.get('certifications', [])
    }
    
    # Extract job requirements
    job_requirements = {
        'title': jd_struct.get('title', 'Position'),
        'must_have_skills': jd_struct.get('must_have_skills', [])[:10],
        'good_to_have_skills': jd_struct.get('good_to_have_skills', [])[:10],
        'education_required': jd_struct.get('education_requirements', {}),
        'certifications_required': jd_struct.get('certifications_required', []),
        'experience_level': jd_struct.get('experience_level', 'not specified')
    }
    
    # Extract scoring insights
    scoring_insights = {
        'overall_score': score_breakdown.get('raw_score', 0),
        'hard_score': score_breakdown.get('hard_score', 0),
        'semantic_score': score_breakdown.get('semantic_score', 0),
        'missing_must_have': score_breakdown.get('skill_matches', {}).get('missing_must', []),
        'missing_good_to_have': score_breakdown.get('skill_matches', {}).get('missing_good', []),
        'education_match': score_breakdown.get('education_match', False),
        'main_weakness': _identify_main_weakness(score_breakdown)
    }
    
    return {
        'candidate': candidate_profile,
        'job': job_requirements,
        'scoring': scoring_insights
    }


def _call_llm_for_feedback(context: Dict[str, Any], num_suggestions: int) -> List[Dict[str, str]]:
    """
    Call LLM API to generate improvement suggestions.
    
    This function crafts a carefully designed prompt that:
    - Provides clear context about the candidate and role
    - Specifies the desired output format
    - Includes examples to guide the model
    - Sets constraints to ensure useful advice
    """
    global _openai_client
    
    # Initialize client if needed
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Craft the prompt
    prompt = _create_feedback_prompt(context, num_suggestions)
    
    try:
        response = _openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert career coach and resume reviewer. Provide actionable, specific advice to help candidates improve their job application success."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.7,  # Some creativity but not too random
            max_tokens=800,   # Enough for detailed suggestions
            top_p=0.9        # Focus on most likely completions
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        suggestions = _parse_llm_response(response_text)
        
        return suggestions
        
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        raise


def _create_feedback_prompt(context: Dict[str, Any], num_suggestions: int) -> str:
    """
    Create a structured prompt for the LLM.
    
    Prompt engineering best practices:
    - Clear instructions with examples
    - Structured input data
    - Specific output format requirements
    - Constraints to prevent hallucination
    - Context about the domain (resume matching)
    """
    candidate = context['candidate']
    job = context['job']
    scoring = context['scoring']
    
    prompt = f"""
You are analyzing a resume for the position: {job['title']}

CANDIDATE PROFILE:
- Skills: {', '.join(candidate['skills'][:10])}
- Experience: ~{candidate['experience_years']} years
- Education: {candidate['education_level']}
- Has Projects: {candidate['has_relevant_projects']}
- Certifications: {', '.join(candidate['certifications']) if candidate['certifications'] else 'None'}

JOB REQUIREMENTS:
- Must-have skills: {', '.join(job['must_have_skills'])}
- Good-to-have skills: {', '.join(job['good_to_have_skills'])}
- Education required: {job['education_required'].get('level', 'Not specified')}
- Required certifications: {', '.join(job['certifications_required']) if job['certifications_required'] else 'None'}

SCORING ANALYSIS:
- Overall match score: {scoring['overall_score']:.1f}/100
- Missing must-have skills: {', '.join(scoring['missing_must_have']) if scoring['missing_must_have'] else 'None'}
- Missing good-to-have skills: {', '.join(scoring['missing_good_to_have'][:5]) if scoring['missing_good_to_have'] else 'None'}
- Education requirement met: {scoring['education_match']}
- Main weakness: {scoring['main_weakness']}

Generate {num_suggestions} specific, actionable improvement suggestions. For each suggestion, provide:
1. A one-line action the candidate should take
2. A one-line concrete example of how to implement it
3. Priority level (high/medium/low)
4. Category (skill/experience/education/format)

Focus on the most impactful improvements based on the missing requirements and scoring analysis.

EXAMPLE FORMAT:
{{
  "action": "Add React.js projects to demonstrate frontend skills",
  "example": "Build a portfolio website using React with interactive components",
  "priority": "high",
  "category": "skill"
}}

Provide {num_suggestions} suggestions in valid JSON array format:
"""
    
    return prompt


def _parse_llm_response(response_text: str) -> List[Dict[str, str]]:
    """
    Parse LLM response into structured suggestions.
    
    LLM responses can be inconsistent, so this function:
    - Handles various JSON formatting issues
    - Extracts suggestions even from malformed responses
    - Validates required fields
    - Provides fallbacks for parsing errors
    """
    suggestions = []
    
    try:
        # Try to parse as JSON first
        if response_text.strip().startswith('['):
            suggestions = json.loads(response_text)
        elif response_text.strip().startswith('{'):
            # Single suggestion wrapped in object
            suggestion = json.loads(response_text)
            suggestions = [suggestion]
        else:
            # Try to extract JSON from text
            import re
            json_matches = re.findall(r'\{[^{}]*\}', response_text)
            for match in json_matches:
                try:
                    suggestion = json.loads(match)
                    suggestions.append(suggestion)
                except:
                    continue
    
    except json.JSONDecodeError:
        # Fallback: parse structured text manually
        suggestions = _parse_text_response(response_text)
    
    return suggestions


def _parse_text_response(response_text: str) -> List[Dict[str, str]]:
    """
    Fallback parser for non-JSON responses.
    
    Sometimes LLMs return well-structured text that isn't valid JSON.
    This parser extracts suggestions from such responses.
    """
    suggestions = []
    lines = response_text.split('\n')
    
    current_suggestion = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for action patterns
        if line.lower().startswith('action:') or 'action' in line.lower():
            if current_suggestion and 'action' in current_suggestion:
                suggestions.append(current_suggestion)
                current_suggestion = {}
            current_suggestion['action'] = line.split(':', 1)[-1].strip()
        
        elif line.lower().startswith('example:') or 'example' in line.lower():
            current_suggestion['example'] = line.split(':', 1)[-1].strip()
        
        elif 'priority' in line.lower():
            priority = 'medium'
            if 'high' in line.lower():
                priority = 'high'
            elif 'low' in line.lower():
                priority = 'low'
            current_suggestion['priority'] = priority
        
        elif 'category' in line.lower():
            category = 'skill'
            if 'experience' in line.lower():
                category = 'experience'
            elif 'education' in line.lower():
                category = 'education'
            elif 'format' in line.lower():
                category = 'format'
            current_suggestion['category'] = category
    
    # Add final suggestion
    if current_suggestion and 'action' in current_suggestion:
        suggestions.append(current_suggestion)
    
    return suggestions


def _validate_and_format_suggestions(suggestions: List[Dict[str, str]], 
                                   num_requested: int) -> List[Dict[str, str]]:
    """
    Validate and format LLM-generated suggestions.
    
    Ensures suggestions meet quality standards:
    - Have required fields (action, example)
    - Are actionable and specific
    - Don't contain inappropriate content
    - Are properly formatted and trimmed
    """
    validated = []
    
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        
        # Check required fields
        if 'action' not in suggestion or 'example' not in suggestion:
            continue
        
        # Clean and validate content
        action = str(suggestion['action']).strip()
        example = str(suggestion['example']).strip()
        
        if len(action) < 10 or len(example) < 10:
            continue  # Too short to be useful
        
        if len(action) > 200 or len(example) > 200:
            action = action[:200] + '...'
            example = example[:200] + '...'
        
        # Set defaults for optional fields
        priority = suggestion.get('priority', 'medium').lower()
        if priority not in ['high', 'medium', 'low']:
            priority = 'medium'
        
        category = suggestion.get('category', 'skill').lower()
        if category not in ['skill', 'experience', 'education', 'format']:
            category = 'skill'
        
        validated_suggestion = {
            'action': action,
            'example': example,
            'priority': priority,
            'category': category
        }
        
        validated.append(validated_suggestion)
        
        if len(validated) >= num_requested:
            break
    
    # If we don't have enough suggestions, pad with templates
    while len(validated) < num_requested:
        template_suggestions = _get_template_suggestions()
        for template in template_suggestions:
            if len(validated) >= num_requested:
                break
            if template not in validated:  # Avoid duplicates
                validated.append(template)
    
    return validated[:num_requested]


def _generate_template_feedback(resume_struct: Dict[str, Any], jd_struct: Dict[str, Any], 
                               score_breakdown: Dict[str, Any], num_suggestions: int) -> List[Dict[str, str]]:
    """
    Generate template-based feedback when LLM is unavailable.
    
    This fallback system provides basic but useful feedback based on:
    - Rule-based analysis of skill gaps
    - Common resume improvement patterns
    - Industry best practices
    - Scoring breakdown insights
    """
    suggestions = []
    
    # Analyze missing skills
    missing_must = score_breakdown.get('skill_matches', {}).get('missing_must', [])
    missing_good = score_breakdown.get('skill_matches', {}).get('missing_good', [])
    
    # Generate skill-based suggestions
    if missing_must:
        top_missing = missing_must[0]
        suggestions.append({
            'action': f"Develop and showcase {top_missing} skills through projects or coursework",
            'example': f"Create a portfolio project demonstrating {top_missing} proficiency",
            'priority': 'high',
            'category': 'skill'
        })
    
    if missing_good:
        top_good = missing_good[0]
        suggestions.append({
            'action': f"Consider learning {top_good} to stand out from other candidates",
            'example': f"Complete an online course or tutorial in {top_good}",
            'priority': 'medium',
            'category': 'skill'
        })
    
    # Education-based suggestions
    if not score_breakdown.get('education_match', True):
        suggestions.append({
            'action': "Highlight relevant coursework or self-study to bridge education gap",
            'example': "Add a 'Relevant Coursework' section listing applicable subjects",
            'priority': 'medium',
            'category': 'education'
        })
    
    # Add generic improvement suggestions
    templates = _get_template_suggestions()
    suggestions.extend(templates)
    
    return suggestions[:num_suggestions]


def _get_template_suggestions() -> List[Dict[str, str]]:
    """Return common template suggestions for fallback use."""
    return [
        {
            'action': 'Quantify achievements with specific metrics and numbers',
            'example': 'Replace "improved performance" with "increased efficiency by 25%"',
            'priority': 'high',
            'category': 'format'
        },
        {
            'action': 'Add more technical projects to demonstrate hands-on experience',
            'example': 'Include a GitHub portfolio with 2-3 relevant projects',
            'priority': 'medium',
            'category': 'experience'
        },
        {
            'action': 'Customize resume keywords to match job description terminology',
            'example': 'Use "machine learning" instead of "ML" if JD uses full term',
            'priority': 'medium',
            'category': 'format'
        }
    ]


# Utility functions
def _estimate_experience_years(experience_list: List[Dict]) -> int:
    """Estimate total years of experience from experience entries."""
    if not experience_list:
        return 0
    
    total_months = 0
    for exp in experience_list:
        # Simple heuristic - assume 12 months if no dates
        total_months += 12
    
    return max(1, total_months // 12)


def _extract_education_level(education_list: List[Dict]) -> str:
    """Extract highest education level."""
    if not education_list:
        return "Not specified"
    
    levels = ['diploma', 'associate', 'bachelor', 'master', 'phd', 'doctorate']
    highest = "High school"
    
    for edu in education_list:
        degree = edu.get('degree', '').lower()
        for level in levels:
            if level in degree:
                if levels.index(level) > levels.index(highest.lower()):
                    highest = level.title()
    
    return highest


def _identify_main_weakness(score_breakdown: Dict) -> str:
    """Identify the main weakness based on scoring breakdown."""
    scores = {
        'skills': score_breakdown.get('skill_score', 0),
        'education': score_breakdown.get('education_score', 0),
        'experience': score_breakdown.get('experience_score', 0),
        'overall_content': score_breakdown.get('semantic_score', 0)
    }
    
    lowest_score = min(scores.values())
    for category, score in scores.items():
        if score == lowest_score:
            return category
    
    return 'skills'


def _apply_rate_limiting():
    """Apply simple rate limiting to avoid API limits."""
    global _last_request_times
    
    current_time = time.time()
    
    # Remove requests older than 1 minute
    _last_request_times = [t for t in _last_request_times if current_time - t < 60]
    
    # Check if we're at the limit
    if len(_last_request_times) >= MAX_FEEDBACK_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (current_time - _last_request_times[0])
        if sleep_time > 0:
            logging.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
    
    # Record this request
    _last_request_times.append(current_time)


def get_feedback_cost_estimate(num_requests: int, avg_resume_length: int = 2000) -> Dict[str, float]:
    """
    Estimate cost for generating feedback.
    
    Args:
        num_requests: Number of feedback requests
        avg_resume_length: Average resume length in characters
        
    Returns:
        Dict: Cost estimates for different models
    """
    # Rough token estimates (1 token ≈ 4 characters)
    prompt_tokens = (avg_resume_length + 1000) // 4  # Resume + prompt overhead
    completion_tokens = 300  # Typical feedback response
    
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03}
    }
    
    estimates = {}
    for model, costs in pricing.items():
        input_cost = (prompt_tokens / 1000) * costs['input'] * num_requests
        output_cost = (completion_tokens / 1000) * costs['output'] * num_requests
        total_cost = input_cost + output_cost
        
        estimates[model] = {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
    
    return estimates
