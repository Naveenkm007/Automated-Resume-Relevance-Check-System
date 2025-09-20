#!/usr/bin/env python3
"""
Hard Match Scoring Example

This script demonstrates the hard match scoring system by comparing
sample job descriptions with resume structures. It shows how the
deterministic scoring algorithm works and what factors contribute
to the final score.

Usage:
    python examples/hard_match_example.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scoring.hard_match import compute_keyword_score, tfidf_similarity, fuzzy_match_score


def create_sample_jd():
    """Create a sample job description structure."""
    return {
        'title': 'Senior Software Engineer',
        'must_have_skills': [
            'python', 'sql', 'javascript', 'git'
        ],
        'good_to_have_skills': [
            'aws', 'docker', 'kubernetes', 'spark', 'react'
        ],
        'education_requirements': {
            'level': 'bachelor',
            'field': 'computer science'
        },
        'certifications_required': [
            'aws certified developer'
        ],
        'full_text': """
        Senior Software Engineer Position
        
        We are seeking an experienced Senior Software Engineer to join our dynamic team.
        The ideal candidate will have strong experience in Python development, SQL databases,
        and modern JavaScript frameworks.
        
        Required Skills:
        - Proficiency in Python programming
        - Strong SQL database experience
        - JavaScript and modern web development
        - Version control with Git
        
        Preferred Skills:
        - AWS cloud services experience
        - Containerization with Docker
        - Kubernetes orchestration
        - Apache Spark for big data processing
        - React.js for frontend development
        
        Education: Bachelor's degree in Computer Science or related field
        Certifications: AWS Certified Developer preferred
        
        You will be responsible for designing, developing, and maintaining scalable
        software solutions. Experience with cloud technologies and DevOps practices
        is highly valued.
        """
    }


def create_sample_resume_strong():
    """Create a strong candidate resume structure."""
    return {
        'name': 'Alice Johnson',
        'email': 'alice.johnson@email.com',
        'phone': '(555) 123-4567',
        'skills': [
            'python', 'sql', 'javascript', 'git', 'aws', 'docker', 
            'react', 'postgresql', 'django', 'flask'
        ],
        'education': [
            {
                'degree': 'Bachelor of Science in Computer Science',
                'institution': 'MIT',
                'year': 2019,
                'stream': 'Computer Science'
            }
        ],
        'certifications': [
            'AWS Certified Developer Associate',
            'Python Institute PCEP'
        ],
        'experience': [
            {
                'title': 'Software Engineer',
                'company': 'TechCorp',
                'start': '2019-06',
                'end': 'Present',
                'bullets': [
                    'Developed web applications using Python and Django',
                    'Managed PostgreSQL databases and optimized SQL queries',
                    'Built responsive frontends with JavaScript and React',
                    'Deployed applications on AWS using Docker containers'
                ]
            }
        ],
        'projects': [
            {
                'title': 'E-commerce Platform',
                'desc': 'Built a scalable e-commerce platform using Python, Django, PostgreSQL, and AWS'
            }
        ],
        'full_text': """
        Alice Johnson
        Software Engineer
        alice.johnson@email.com | (555) 123-4567
        
        EXPERIENCE
        Software Engineer at TechCorp (2019-Present)
        • Developed web applications using Python and Django framework
        • Managed PostgreSQL databases and optimized complex SQL queries
        • Built responsive user interfaces with JavaScript and React.js
        • Deployed and maintained applications on AWS cloud infrastructure
        • Used Docker for containerization and Git for version control
        • Collaborated with team using Agile development methodologies
        
        EDUCATION
        Bachelor of Science in Computer Science
        Massachusetts Institute of Technology, 2019
        
        SKILLS
        Programming: Python, JavaScript, SQL
        Frameworks: Django, Flask, React.js
        Databases: PostgreSQL, MySQL
        Cloud: AWS (EC2, S3, RDS, Lambda)
        Tools: Docker, Git, Jenkins
        
        CERTIFICATIONS
        • AWS Certified Developer Associate
        • Python Institute PCEP Certified Entry-Level Python Programmer
        
        PROJECTS
        E-commerce Platform
        Built a scalable e-commerce platform using Python, Django, PostgreSQL, and AWS.
        Implemented user authentication, payment processing, and inventory management.
        Deployed using Docker containers with automated CI/CD pipeline.
        """
    }


def create_sample_resume_weak():
    """Create a weaker candidate resume structure."""
    return {
        'name': 'Bob Smith',
        'email': 'bob.smith@email.com', 
        'phone': '(555) 987-6543',
        'skills': [
            'java', 'c++', 'html', 'css', 'mysql'
        ],
        'education': [
            {
                'degree': 'Associate Degree in Information Technology',
                'institution': 'Community College',
                'year': 2020,
                'stream': 'Information Technology'
            }
        ],
        'certifications': [],
        'experience': [
            {
                'title': 'Junior Developer',
                'company': 'Small Company',
                'start': '2020-01',
                'end': '2022-12',
                'bullets': [
                    'Maintained legacy Java applications',
                    'Created simple HTML/CSS websites',
                    'Basic MySQL database operations'
                ]
            }
        ],
        'projects': [
            {
                'title': 'Personal Website',
                'desc': 'Created a personal portfolio website with HTML, CSS, and basic JavaScript'
            }
        ],
        'full_text': """
        Bob Smith
        Junior Developer
        bob.smith@email.com | (555) 987-6543
        
        EXPERIENCE
        Junior Developer at Small Company (2020-2022)
        • Maintained legacy Java applications
        • Created simple HTML and CSS websites
        • Performed basic MySQL database operations
        • Fixed bugs and made minor enhancements
        
        EDUCATION
        Associate Degree in Information Technology
        Community College, 2020
        
        SKILLS
        Programming: Java, C++
        Web: HTML, CSS, basic JavaScript
        Database: MySQL
        
        PROJECTS
        Personal Website
        Created a personal portfolio website using HTML, CSS, and basic JavaScript.
        Showcased academic projects and contact information.
        """
    }


def run_hard_match_demo():
    """Run the hard match scoring demonstration."""
    print("🎯 HARD MATCH SCORING DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    jd = create_sample_jd()
    strong_resume = create_sample_resume_strong()
    weak_resume = create_sample_resume_weak()
    
    print("\n📋 JOB DESCRIPTION REQUIREMENTS:")
    print(f"   Title: {jd['title']}")
    print(f"   Must-have skills: {', '.join(jd['must_have_skills'])}")
    print(f"   Good-to-have skills: {', '.join(jd['good_to_have_skills'])}")
    print(f"   Education: {jd['education_requirements']['level'].title()} in {jd['education_requirements']['field'].title()}")
    print(f"   Certifications: {', '.join(jd['certifications_required'])}")
    
    # Test strong candidate
    print("\n" + "="*60)
    print("👤 CANDIDATE 1: ALICE JOHNSON (Strong Match)")
    print("="*60)
    
    print(f"📄 Resume Summary:")
    print(f"   Name: {strong_resume['name']}")
    print(f"   Skills: {', '.join(strong_resume['skills'][:8])}...")
    print(f"   Education: {strong_resume['education'][0]['degree']}")
    print(f"   Certifications: {', '.join(strong_resume['certifications'])}")
    
    # Compute hard match score
    result_strong = compute_keyword_score(strong_resume, jd)
    
    print(f"\n🔍 DETAILED SCORING RESULTS:")
    print(f"   Must-have skills matched: {result_strong['skill_matches']['must_have']}/{len(jd['must_have_skills'])}")
    print(f"   Good-to-have skills matched: {result_strong['skill_matches']['good_to_have']}/{len(jd['good_to_have_skills'])}")
    print(f"   Education requirement met: {result_strong['education_match']}")
    print(f"   Certifications matched: {len(result_strong['certifications_match'])}")
    print(f"   TF-IDF similarity: {result_strong['tfidf_score']:.3f}")
    
    print(f"\n📊 SCORE BREAKDOWN:")
    breakdown = result_strong['breakdown']
    print(f"   Skill score: {breakdown['skill_score']:.1f}/70 points")
    print(f"   Education score: {breakdown['education_score']:.1f}/10 points")
    print(f"   Certification score: {breakdown['certification_score']:.1f}/5 points")
    print(f"   TF-IDF score: {breakdown['tfidf_score']:.1f}/15 points")
    print(f"   ➤ TOTAL SCORE: {result_strong['raw_score']:.1f}/100")
    
    if result_strong['skill_matches']['missing_must']:
        print(f"   ⚠️  Missing must-have skills: {', '.join(result_strong['skill_matches']['missing_must'])}")
    else:
        print(f"   ✅ All must-have skills covered!")
    
    # Test weak candidate
    print("\n" + "="*60)
    print("👤 CANDIDATE 2: BOB SMITH (Weak Match)")
    print("="*60)
    
    print(f"📄 Resume Summary:")
    print(f"   Name: {weak_resume['name']}")
    print(f"   Skills: {', '.join(weak_resume['skills'])}")
    print(f"   Education: {weak_resume['education'][0]['degree']}")
    print(f"   Certifications: {'None' if not weak_resume['certifications'] else ', '.join(weak_resume['certifications'])}")
    
    # Compute hard match score
    result_weak = compute_keyword_score(weak_resume, jd)
    
    print(f"\n🔍 DETAILED SCORING RESULTS:")
    print(f"   Must-have skills matched: {result_weak['skill_matches']['must_have']}/{len(jd['must_have_skills'])}")
    print(f"   Good-to-have skills matched: {result_weak['skill_matches']['good_to_have']}/{len(jd['good_to_have_skills'])}")
    print(f"   Education requirement met: {result_weak['education_match']}")
    print(f"   Certifications matched: {len(result_weak['certifications_match'])}")
    print(f"   TF-IDF similarity: {result_weak['tfidf_score']:.3f}")
    
    print(f"\n📊 SCORE BREAKDOWN:")
    breakdown_weak = result_weak['breakdown']
    print(f"   Skill score: {breakdown_weak['skill_score']:.1f}/70 points")
    print(f"   Education score: {breakdown_weak['education_score']:.1f}/10 points")  
    print(f"   Certification score: {breakdown_weak['certification_score']:.1f}/5 points")
    print(f"   TF-IDF score: {breakdown_weak['tfidf_score']:.1f}/15 points")
    print(f"   ➤ TOTAL SCORE: {result_weak['raw_score']:.1f}/100")
    
    print(f"   ❌ Missing must-have skills: {', '.join(result_weak['skill_matches']['missing_must'])}")
    
    # Comparison
    print("\n" + "="*60)
    print("📈 CANDIDATE COMPARISON")
    print("="*60)
    
    print(f"Alice Johnson (Strong): {result_strong['raw_score']:.1f}/100")
    print(f"Bob Smith (Weak): {result_weak['raw_score']:.1f}/100")
    print(f"Score difference: {result_strong['raw_score'] - result_weak['raw_score']:.1f} points")
    
    # Demonstrate individual functions
    print("\n" + "="*60)
    print("🔧 INDIVIDUAL FUNCTION DEMONSTRATIONS")
    print("="*60)
    
    # TF-IDF similarity
    tfidf_score = tfidf_similarity(strong_resume['full_text'], jd['full_text'])
    print(f"\n📝 TF-IDF Similarity:")
    print(f"   Alice vs JD: {tfidf_score:.3f}")
    
    tfidf_score_weak = tfidf_similarity(weak_resume['full_text'], jd['full_text'])
    print(f"   Bob vs JD: {tfidf_score_weak:.3f}")
    
    # Fuzzy matching
    print(f"\n🔀 Fuzzy Match Examples:")
    candidate_skills = ['javascript', 'js', 'reactjs', 'postgres']
    jd_skills = ['javascript', 'react', 'postgresql']
    fuzzy_score = fuzzy_match_score(candidate_skills, jd_skills)
    print(f"   Skills: {candidate_skills}")
    print(f"   vs JD: {jd_skills}")
    print(f"   Fuzzy match score: {fuzzy_score:.3f}")
    
    # Show JSON output for integration
    print(f"\n💻 JSON OUTPUT (for API integration):")
    print(json.dumps(result_strong, indent=2))
    
    return result_strong, result_weak


def demo_custom_weights():
    """Demonstrate custom weight configuration."""
    print("\n" + "="*60)
    print("⚖️  CUSTOM WEIGHT CONFIGURATION DEMO")
    print("="*60)
    
    jd = create_sample_jd()
    resume = create_sample_resume_strong()
    
    # Default weights
    result_default = compute_keyword_score(resume, jd)
    
    # Skills-focused weights (for technical roles)
    skills_focused_weights = {
        'must_have_skills': 0.70,     # 70% - Much higher emphasis on must-have skills
        'good_to_have_skills': 0.20,  # 20% - Keep good-to-have importance
        'education_match': 0.05,      # 5% - Reduce education importance
        'certifications_match': 0.05, # 5% - Keep certifications
        'tfidf_similarity': 0.00      # 0% - Ignore text similarity
    }
    
    result_skills = compute_keyword_score(resume, jd, skills_focused_weights)
    
    # Education-focused weights (for academic/research roles)
    education_focused_weights = {
        'must_have_skills': 0.30,     # 30% - Reduce skill importance
        'good_to_have_skills': 0.10,  # 10% - Less good-to-have importance
        'education_match': 0.40,      # 40% - High education importance
        'certifications_match': 0.20, # 20% - Higher certification importance
        'tfidf_similarity': 0.00      # 0% - Ignore text similarity
    }
    
    result_education = compute_keyword_score(resume, jd, education_focused_weights)
    
    print(f"Default weights score: {result_default['raw_score']:.1f}")
    print(f"Skills-focused score: {result_skills['raw_score']:.1f}")
    print(f"Education-focused score: {result_education['raw_score']:.1f}")
    
    print(f"\n💡 Weight Tuning Insights:")
    print(f"   • Skills-focused approach gives higher scores to technical candidates")
    print(f"   • Education-focused approach values academic credentials more")
    print(f"   • Custom weights allow domain-specific optimization")


def main():
    """Main demonstration function."""
    try:
        print("🚀 Starting Hard Match Scoring Demo...")
        
        # Run main demo
        strong_result, weak_result = run_hard_match_demo()
        
        # Show custom weights
        demo_custom_weights()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"💡 Key Takeaways:")
        print(f"   • Hard matching provides fast, interpretable resume screening")
        print(f"   • Combines exact matching, fuzzy matching, and text similarity")
        print(f"   • Weights can be customized for different role requirements")
        print(f"   • Missing skills are clearly identified for feedback")
        print(f"   • Scores are normalized to 0-100 range for easy comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
