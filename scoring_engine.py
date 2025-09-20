#!/usr/bin/env python3
"""
Resume Relevance Scoring Engine
Generates 0-100 relevance scores for resume-JD matching
"""

import re
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

@dataclass
class JobDescription:
    """Structured job description data."""
    title: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    experience_required: str  # e.g., "2-4 years"
    education_required: List[str]  # e.g., ["Bachelor's", "Computer Science"]
    certifications: List[str]
    description: str

@dataclass
class ScoringResult:
    """Complete scoring result."""
    relevance_score: float  # 0-100
    verdict: str  # "High", "Medium", "Low"
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    semantic_match_score: float
    missing_must_have_skills: List[str]
    missing_good_to_have_skills: List[str]
    suggestions: List[str]
    score_breakdown: Dict[str, float]

class RelevanceScorer:
    """Main scoring engine for resume-JD matching."""
    
    def __init__(self):
        self.skill_synonyms = {
            'javascript': ['js', 'node.js', 'nodejs', 'react', 'vue', 'angular'],
            'python': ['django', 'flask', 'fastapi', 'py'],
            'java': ['spring', 'hibernate', 'jsp', 'servlet'],
            'aws': ['amazon web services', 'ec2', 's3', 'lambda', 'cloudformation'],
            'docker': ['containerization', 'containers'],
            'kubernetes': ['k8s', 'orchestration'],
            'sql': ['mysql', 'postgresql', 'sqlite', 'database'],
            'machine learning': ['ml', 'ai', 'artificial intelligence', 'deep learning'],
            'react': ['reactjs', 'react.js'],
            'angular': ['angularjs', 'angular.js']
        }
        
        # Scoring weights
        self.weights = {
            'must_have_skills': 0.40,  # 40% weight
            'good_to_have_skills': 0.15,  # 15% weight
            'experience': 0.20,  # 20% weight
            'education': 0.10,  # 10% weight
            'semantic_match': 0.15  # 15% weight
        }
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill name for better matching."""
        skill = skill.lower().strip()
        skill = re.sub(r'[^\w\s]', '', skill)
        skill = re.sub(r'\s+', ' ', skill)
        return skill
    
    def find_skill_matches(self, resume_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str]]:
        """Find matched and missing skills."""
        resume_skills_normalized = [self.normalize_skill(s) for s in resume_skills]
        required_skills_normalized = [self.normalize_skill(s) for s in required_skills]
        
        matched = []
        missing = []
        
        for req_skill in required_skills_normalized:
            found = False
            
            # Exact match
            if req_skill in resume_skills_normalized:
                matched.append(req_skill)
                found = True
            else:
                # Fuzzy match and synonym check
                for resume_skill in resume_skills_normalized:
                    if self.similarity_ratio(req_skill, resume_skill) > 0.8:
                        matched.append(req_skill)
                        found = True
                        break
                    
                    # Check synonyms
                    if self.check_skill_synonyms(req_skill, resume_skill):
                        matched.append(req_skill)
                        found = True
                        break
            
            if not found:
                missing.append(req_skill)
        
        return matched, missing
    
    def similarity_ratio(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def check_skill_synonyms(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are synonyms."""
        for base_skill, synonyms in self.skill_synonyms.items():
            if (skill1 == base_skill and skill2 in synonyms) or \
               (skill2 == base_skill and skill1 in synonyms) or \
               (skill1 in synonyms and skill2 in synonyms):
                return True
        return False
    
    def score_skills(self, resume_skills: List[str], must_have: List[str], good_to_have: List[str]) -> Dict[str, Any]:
        """Score skill matching."""
        must_matched, must_missing = self.find_skill_matches(resume_skills, must_have)
        good_matched, good_missing = self.find_skill_matches(resume_skills, good_to_have)
        
        # Calculate scores
        must_have_score = (len(must_matched) / len(must_have)) * 100 if must_have else 100
        good_to_have_score = (len(good_matched) / len(good_to_have)) * 100 if good_to_have else 100
        
        # Combined skill score
        skill_score = (must_have_score * 0.7) + (good_to_have_score * 0.3)
        
        return {
            'score': skill_score,
            'must_have_score': must_have_score,
            'good_to_have_score': good_to_have_score,
            'must_matched': must_matched,
            'must_missing': must_missing,
            'good_matched': good_matched,
            'good_missing': good_missing
        }
    
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience from text."""
        patterns = [
            r'(\d+)[\+\-\s]*(?:to|-)[\s]*(\d+)[\s]*years?',
            r'(\d+)[\s]*years?',
            r'(\d+)[\s]*yrs?',
            r'(\d+)[\+][\s]*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    return float(matches[0][1])  # Take the upper bound
                else:
                    return float(matches[0])
        
        return 0.0
    
    def score_experience(self, resume_experience: List[Dict], required_experience: str) -> float:
        """Score experience matching."""
        if not required_experience:
            return 100.0
        
        # Extract required years
        required_years = self.extract_experience_years(required_experience)
        
        # Calculate total experience from resume
        total_years = 0.0
        for exp in resume_experience:
            exp_text = f"{exp.get('title', '')} {exp.get('company', '')}"
            exp_years = self.extract_experience_years(exp_text)
            total_years += exp_years
        
        if total_years == 0:
            # Try to extract from bullets
            for exp in resume_experience:
                for bullet in exp.get('bullets', []):
                    exp_years = self.extract_experience_years(bullet)
                    total_years += exp_years
        
        # Score based on experience match
        if total_years >= required_years:
            return 100.0
        elif total_years >= required_years * 0.7:  # 70% of required
            return 80.0
        elif total_years >= required_years * 0.5:  # 50% of required
            return 60.0
        else:
            return (total_years / required_years) * 50.0 if required_years > 0 else 0.0
    
    def score_education(self, resume_education: List[Dict], required_education: List[str]) -> float:
        """Score education matching."""
        if not required_education:
            return 100.0
        
        if not resume_education:
            return 0.0
        
        education_text = " ".join([
            f"{edu.get('degree', '')} {edu.get('institution', '')}"
            for edu in resume_education
        ]).lower()
        
        matches = 0
        for req_ed in required_education:
            if req_ed.lower() in education_text:
                matches += 1
            elif any(word in education_text for word in req_ed.lower().split()):
                matches += 0.5
        
        return min(100.0, (matches / len(required_education)) * 100)
    
    def semantic_similarity_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity between resume and JD."""
        if not nlp:
            # Fallback: simple word overlap
            resume_words = set(resume_text.lower().split())
            jd_words = set(jd_text.lower().split())
            
            if not jd_words:
                return 0.0
            
            overlap = len(resume_words.intersection(jd_words))
            return min(100.0, (overlap / len(jd_words)) * 100)
        
        # Use spaCy for better semantic analysis
        resume_doc = nlp(resume_text[:1000000])  # Limit text length
        jd_doc = nlp(jd_text[:1000000])
        
        similarity = resume_doc.similarity(jd_doc)
        return similarity * 100
    
    def generate_verdict(self, score: float) -> str:
        """Generate verdict based on score."""
        if score >= 75:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"
    
    def generate_suggestions(self, missing_must: List[str], missing_good: List[str], 
                           experience_score: float, education_score: float) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if missing_must:
            suggestions.append(f"Learn these critical skills: {', '.join(missing_must[:3])}")
        
        if missing_good and len(missing_good) > 2:
            suggestions.append(f"Consider adding these skills: {', '.join(missing_good[:2])}")
        
        if experience_score < 60:
            suggestions.append("Gain more relevant work experience or highlight existing experience better")
        
        if education_score < 50:
            suggestions.append("Consider relevant certifications or additional education")
        
        if len(suggestions) == 0:
            suggestions.append("Great profile! Focus on keeping skills updated")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def calculate_relevance_score(self, resume_data: Dict, job_description: JobDescription) -> ScoringResult:
        """Calculate complete relevance score for a resume against a job description."""
        
        # Extract resume data
        resume_skills = resume_data.get('skills', [])
        resume_experience = resume_data.get('experience', [])
        resume_education = resume_data.get('education', [])
        resume_text = " ".join([
            resume_data.get('name', ''),
            " ".join(resume_skills),
            " ".join([exp.get('title', '') + ' ' + exp.get('company', '') for exp in resume_experience])
        ])
        
        # Score individual components
        skill_result = self.score_skills(resume_skills, job_description.must_have_skills, 
                                       job_description.good_to_have_skills)
        
        experience_score = self.score_experience(resume_experience, job_description.experience_required)
        education_score = self.score_education(resume_education, job_description.education_required)
        semantic_score = self.semantic_similarity_score(resume_text, job_description.description)
        
        # Calculate weighted final score
        final_score = (
            skill_result['score'] * self.weights['must_have_skills'] +
            skill_result['good_to_have_score'] * self.weights['good_to_have_skills'] +
            experience_score * self.weights['experience'] +
            education_score * self.weights['education'] +
            semantic_score * self.weights['semantic_match']
        )
        
        # Generate verdict and suggestions
        verdict = self.generate_verdict(final_score)
        suggestions = self.generate_suggestions(
            skill_result['must_missing'], 
            skill_result['good_missing'],
            experience_score, 
            education_score
        )
        
        return ScoringResult(
            relevance_score=round(final_score, 1),
            verdict=verdict,
            skill_match_score=skill_result['score'],
            experience_match_score=experience_score,
            education_match_score=education_score,
            semantic_match_score=semantic_score,
            missing_must_have_skills=skill_result['must_missing'],
            missing_good_to_have_skills=skill_result['good_missing'],
            suggestions=suggestions,
            score_breakdown={
                'Skills (Must Have)': skill_result['must_have_score'],
                'Skills (Good to Have)': skill_result['good_to_have_score'],
                'Experience': experience_score,
                'Education': education_score,
                'Semantic Match': semantic_score,
                'Final Score': final_score
            }
        )

def demo_scoring():
    """Demo the scoring system."""
    # Sample job description
    jd = JobDescription(
        title="Senior Python Developer",
        must_have_skills=["Python", "Django", "PostgreSQL", "REST API"],
        good_to_have_skills=["Docker", "AWS", "React", "Redis"],
        experience_required="3-5 years",
        education_required=["Bachelor's", "Computer Science"],
        certifications=["AWS Certified Developer"],
        description="We are looking for an experienced Python developer to build scalable web applications."
    )
    
    # Sample resume data
    resume = {
        'name': 'John Smith',
        'skills': ['Python', 'Django', 'JavaScript', 'Docker', 'PostgreSQL'],
        'experience': [
            {
                'title': 'Software Developer',
                'company': 'Tech Corp',
                'bullets': ['Developed Python applications for 3 years']
            }
        ],
        'education': [
            {
                'degree': 'Bachelor of Science in Computer Science',
                'institution': 'MIT'
            }
        ]
    }
    
    # Calculate score
    scorer = RelevanceScorer()
    result = scorer.calculate_relevance_score(resume, jd)
    
    print("=== RESUME RELEVANCE SCORING DEMO ===")
    print(f"Final Relevance Score: {result.relevance_score}/100")
    print(f"Verdict: {result.verdict}")
    print(f"\nScore Breakdown:")
    for component, score in result.score_breakdown.items():
        print(f"  {component}: {score:.1f}")
    
    print(f"\nMissing Must-Have Skills: {result.missing_must_have_skills}")
    print(f"Missing Good-to-Have Skills: {result.missing_good_to_have_skills}")
    
    print(f"\nSuggestions:")
    for i, suggestion in enumerate(result.suggestions, 1):
        print(f"  {i}. {suggestion}")

if __name__ == "__main__":
    demo_scoring()
