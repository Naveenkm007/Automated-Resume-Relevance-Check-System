#!/usr/bin/env python3
"""
Enhanced Scoring System with Email Automation
============================================

Integrates email automation with the resume relevance scoring system.
Automatically sends emails when candidates score 90%+.

Built for Innomatics Research Labs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from email_automation import EmailAutomation, EmailConfig, CandidateData, load_email_config
import re
from typing import Dict, List, Any, Optional
import logging

# Import your existing modules
try:
    from resume_parser.extract import extract_text_from_file
    from resume_parser.cleaner import normalize_text
    from resume_parser.ner import extract_entities
    from scoring.hard_match import compute_keyword_score
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ Some modules not available - using simplified versions")

logger = logging.getLogger(__name__)

class EnhancedResumeAnalyzer:
    """Enhanced resume analyzer with automatic email notifications."""
    
    def __init__(self, email_config: Optional[EmailConfig] = None):
        self.email_system = EmailAutomation(email_config or load_email_config())
        self.analysis_history = []
        
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text."""
        contact = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact['email'] = email_match.group()
        
        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group()
        
        # Name extraction (improved heuristic)
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            words = line.split()
            
            # Look for likely name patterns
            if (len(words) >= 2 and len(words) <= 4 and 
                all(word.replace('.', '').isalpha() for word in words) and
                not any(keyword in line.lower() for keyword in ['email', 'phone', 'address', 'resume', 'cv']) and
                len(line) < 50):
                contact['name'] = line
                break
        
        return contact
    
    def analyze_resume_with_email(self, 
                                file_path: str, 
                                job_requirements: Dict[str, Any],
                                send_emails: bool = True,
                                score_threshold: float = 90.0) -> Dict[str, Any]:
        """
        Analyze resume and automatically send emails if score exceeds threshold.
        
        Args:
            file_path: Path to resume file
            job_requirements: Job requirements dictionary
            send_emails: Whether to send automated emails
            score_threshold: Score threshold for email automation
            
        Returns:
            Complete analysis results with email status
        """
        
        try:
            # Step 1: Extract text from file
            logger.info(f"📄 Extracting text from {file_path}")
            if MODULES_AVAILABLE:
                text = extract_text_from_file(file_path)
                sections = normalize_text(text)
                entities = extract_entities(text)
            else:
                # Fallback for demonstration
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                entities = {'skills': [], 'experience': [], 'education': []}
                sections = {'summary': text}
            
            # Step 2: Extract contact information
            contact_info = self.extract_contact_info(text)
            
            # Step 3: Perform scoring analysis
            analysis_results = self.perform_detailed_scoring(entities, job_requirements, text)
            
            # Step 4: Create candidate data object
            candidate = CandidateData(
                name=contact_info.get('name', 'Unknown Candidate'),
                email=contact_info.get('email', ''),
                phone=contact_info.get('phone', ''),
                overall_score=analysis_results['overall_score'],
                job_title=job_requirements.get('job_title', 'Unknown Position'),
                matched_skills=analysis_results['matched_skills'],
                missing_skills=analysis_results['missing_skills'],
                resume_text=text[:500] + "..." if len(text) > 500 else text
            )
            
            # Step 5: Process email automation if enabled
            email_results = {"threshold_met": False, "candidate_email_sent": False, "hr_notification_sent": False}
            
            if send_emails and candidate.overall_score >= score_threshold:
                logger.info(f"🎯 Score {candidate.overall_score:.1f}% exceeds threshold {score_threshold}% - Triggering emails!")
                email_results = self.email_system.process_candidate_score(candidate, score_threshold)
            
            # Step 6: Compile final results
            final_results = {
                'candidate_info': {
                    'name': candidate.name,
                    'email': candidate.email,
                    'phone': candidate.phone
                },
                'job_info': {
                    'title': job_requirements.get('job_title', 'Unknown'),
                    'required_skills': job_requirements.get('must_have_skills', []),
                    'preferred_skills': job_requirements.get('good_to_have_skills', [])
                },
                'scoring_results': analysis_results,
                'email_automation': email_results,
                'extracted_entities': entities,
                'contact_extracted': contact_info,
                'processing_timestamp': str(datetime.now())
            }
            
            # Log the analysis
            self.analysis_history.append(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing resume {file_path}: {e}")
            raise
    
    def perform_detailed_scoring(self, entities: Dict, job_requirements: Dict, text: str) -> Dict[str, Any]:
        """Perform detailed scoring analysis."""
        
        # Extract skills from entities or text
        candidate_skills = entities.get('skills', [])
        if not candidate_skills:
            # Fallback: extract skills from text using keywords
            candidate_skills = self.extract_skills_from_text(text)
        
        # Job requirements
        required_skills = [skill.lower().strip() for skill in job_requirements.get('must_have_skills', [])]
        preferred_skills = [skill.lower().strip() for skill in job_requirements.get('good_to_have_skills', [])]
        
        # Skill matching
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        matched_required = [skill for skill in required_skills if skill in candidate_skills_lower]
        matched_preferred = [skill for skill in preferred_skills if skill in candidate_skills_lower]
        missing_required = [skill for skill in required_skills if skill not in candidate_skills_lower]
        
        # Calculate scores
        required_score = (len(matched_required) / max(len(required_skills), 1)) * 100
        preferred_score = (len(matched_preferred) / max(len(preferred_skills), 1)) * 100
        
        # Experience scoring
        experience_entries = entities.get('experience', [])
        experience_score = min(len(experience_entries) * 20, 100) if experience_entries else 50
        
        # Education scoring  
        education_entries = entities.get('education', [])
        education_score = 80 if education_entries else 60
        
        # Overall weighted score
        overall_score = (
            required_score * 0.5 +      # 50% weight on required skills
            preferred_score * 0.2 +     # 20% weight on preferred skills  
            experience_score * 0.2 +    # 20% weight on experience
            education_score * 0.1       # 10% weight on education
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'required_skills_score': round(required_score, 1),
            'preferred_skills_score': round(preferred_score, 1),
            'experience_score': round(experience_score, 1),
            'education_score': round(education_score, 1),
            'matched_skills': matched_required + matched_preferred,
            'missing_skills': missing_required,
            'total_skills_found': len(candidate_skills),
            'verdict': self.get_verdict(overall_score)
        }
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching."""
        common_skills = [
            'python', 'java', 'javascript', 'typescript', 'html', 'css', 'react', 'angular', 'vue',
            'django', 'flask', 'spring', 'node', 'express', 'sql', 'mysql', 'postgresql', 'mongodb',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'api', 'rest', 'microservices'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        return found_skills
    
    def get_verdict(self, score: float) -> str:
        """Get verdict based on score."""
        if score >= 90:
            return "Exceptional Match - Immediate Contact Recommended"
        elif score >= 80:
            return "Excellent Match"
        elif score >= 70:
            return "Good Match"
        elif score >= 60:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        
        report = f"""
        
🎯 RESUME ANALYSIS REPORT
========================

👤 Candidate: {results['candidate_info']['name']}
📧 Email: {results['candidate_info']['email']}
📱 Phone: {results['candidate_info']['phone']}

💼 Position: {results['job_info']['title']}

📊 SCORING RESULTS:
- Overall Score: {results['scoring_results']['overall_score']}% ({results['scoring_results']['verdict']})
- Required Skills: {results['scoring_results']['required_skills_score']}%
- Preferred Skills: {results['scoring_results']['preferred_skills_score']}%  
- Experience: {results['scoring_results']['experience_score']}%
- Education: {results['scoring_results']['education_score']}%

✅ MATCHED SKILLS ({len(results['scoring_results']['matched_skills'])}):
{', '.join(results['scoring_results']['matched_skills'])}

❌ MISSING SKILLS ({len(results['scoring_results']['missing_skills'])}):
{', '.join(results['scoring_results']['missing_skills'])}

📧 EMAIL AUTOMATION:
- Threshold Met: {'✅ YES' if results['email_automation']['threshold_met'] else '❌ NO'}
- Candidate Email: {'✅ SENT' if results['email_automation']['candidate_email_sent'] else '❌ NOT SENT'}
- HR Notification: {'✅ SENT' if results['email_automation']['hr_notification_sent'] else '❌ NOT SENT'}

⏰ Processed: {results['processing_timestamp']}
        """
        
        return report

# Import datetime for timestamp
from datetime import datetime

def demo_email_automation():
    """Demonstrate the email automation feature."""
    
    print("🚀 Email Automation Demo - Innomatics Research Labs")
    print("=" * 60)
    
    # Create analyzer
    analyzer = EnhancedResumeAnalyzer()
    
    # Sample job requirements
    job_requirements = {
        'job_title': 'Senior Python Developer',
        'must_have_skills': ['python', 'django', 'postgresql', 'rest api'],
        'good_to_have_skills': ['aws', 'docker', 'react', 'kubernetes'],
        'experience_required': 3
    }
    
    # Sample resume text (simulated high-scoring candidate)
    sample_resume = """
    John Doe
    john.doe@example.com
    +91-9876543210
    
    Senior Python Developer with 5 years of experience in Django, PostgreSQL, REST API development.
    Skilled in Python, Django, PostgreSQL, REST API, AWS, Docker, React, Kubernetes, Git, Linux.
    
    Experience:
    - Senior Developer at Tech Corp (2020-2024)
    - Python Developer at StartupXYZ (2019-2020)
    
    Education:
    - B.Tech Computer Science, XYZ University (2019)
    """
    
    # Save sample resume to file
    sample_file = "sample_resume.txt"
    with open(sample_file, 'w') as f:
        f.write(sample_resume)
    
    try:
        # Analyze resume (with email automation disabled for demo)
        results = analyzer.analyze_resume_with_email(
            sample_file, 
            job_requirements, 
            send_emails=False,  # Set to True to actually send emails
            score_threshold=90.0
        )
        
        # Generate and display report
        report = analyzer.generate_analysis_report(results)
        print(report)
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("💡 To enable actual email sending:")
        print("   1. Configure environment variables (SENDER_EMAIL, SENDER_PASSWORD, HR_EMAIL)")
        print("   2. Set send_emails=True in the function call")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        import os
        if os.path.exists(sample_file):
            os.remove(sample_file)

if __name__ == "__main__":
    demo_email_automation()
