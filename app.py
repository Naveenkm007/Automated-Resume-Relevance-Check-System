#!/usr/bin/env python3
"""
Automated Resume Relevance Check System - Vercel Deployment
==========================================================

Ultra-lightweight version optimized for Vercel's 250MB serverless function limit.
Built for Innomatics Research Labs.
"""

import streamlit as st
import re
import json
from typing import Dict, List, Any

# Built-in technical skills database
SKILLS_DATABASE = {
    'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
    'frontend': ['html', 'css', 'react', 'angular', 'vue', 'bootstrap', 'tailwind'],
    'backend': ['node', 'express', 'django', 'flask', 'spring', 'fastapi'],
    'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins'],
    'tools': ['git', 'github', 'gitlab', 'jira', 'confluence'],
    'ai_ml': ['machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'],
    'data': ['data science', 'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi'],
    'mobile': ['android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'],
    'other': ['api', 'rest', 'graphql', 'microservices', 'devops', 'ci/cd', 'agile', 'scrum']
}

# Flatten skills for easy searching
ALL_SKILLS = set()
for category in SKILLS_DATABASE.values():
    ALL_SKILLS.update(category)

def extract_skills_from_text(text: str) -> List[str]:
    """Extract technical skills from resume text."""
    text_lower = text.lower()
    found_skills = []
    
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return sorted(found_skills)

def extract_contact_info(text: str) -> Dict[str, str]:
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
    
    # Name extraction (simple heuristic)
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        words = line.split()
        if len(words) == 2 and all(word.isalpha() for word in words):
            if '@' not in line and len(line) < 50:
                contact['name'] = line
                break
    
    return contact

def analyze_resume(text: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze resume against job requirements."""
    
    # Extract information
    skills_found = extract_skills_from_text(text)
    contact_info = extract_contact_info(text)
    
    # Job requirements
    required_skills = [skill.lower().strip() for skill in job_requirements.get('must_have_skills', [])]
    preferred_skills = [skill.lower().strip() for skill in job_requirements.get('good_to_have_skills', [])]
    
    # Calculate matches
    skills_lower = [skill.lower() for skill in skills_found]
    required_matches = [skill for skill in required_skills if skill in skills_lower]
    preferred_matches = [skill for skill in preferred_skills if skill in skills_lower]
    
    # Scoring
    required_score = (len(required_matches) / max(len(required_skills), 1)) * 100
    preferred_score = (len(preferred_matches) / max(len(preferred_skills), 1)) * 100
    
    # Experience detection
    experience_keywords = ['experience', 'worked', 'developed', 'managed', 'led', 'created', 'built', 'designed']
    experience_count = sum(1 for keyword in experience_keywords if keyword in text.lower())
    experience_score = min(experience_count * 12.5, 100)
    
    # Education detection
    education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'certification']
    education_count = sum(1 for keyword in education_keywords if keyword in text.lower())
    education_score = min(education_count * 20, 100)
    
    # Overall score calculation
    overall_score = (required_score * 0.5 + preferred_score * 0.2 + 
                    experience_score * 0.2 + education_score * 0.1)
    
    # Verdict determination
    if overall_score >= 80:
        verdict = "Excellent Match"
        verdict_color = "🟢"
    elif overall_score >= 60:
        verdict = "Good Match"
        verdict_color = "🟡"
    elif overall_score >= 40:
        verdict = "Fair Match"
        verdict_color = "🟠"
    else:
        verdict = "Poor Match"
        verdict_color = "🔴"
    
    return {
        'contact_info': contact_info,
        'skills_found': skills_found,
        'required_matches': required_matches,
        'preferred_matches': preferred_matches,
        'missing_required': [skill for skill in required_skills if skill not in skills_lower],
        'scores': {
            'required_skills': required_score,
            'preferred_skills': preferred_score,
            'experience': experience_score,
            'education': education_score,
            'overall': overall_score
        },
        'verdict': verdict,
        'verdict_color': verdict_color
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="Resume Relevance Check | Innomatics Labs",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-fair { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Automated Resume Relevance Check System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;"><strong>AI-Powered Resume Evaluation Engine | Built for Innomatics Research Labs</strong></p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📄 Upload Resume",
            type=['txt'],
            help="Upload TXT files (Vercel optimized)"
        )
        
        # Job requirements
        st.markdown("### 🎯 Job Requirements")
        
        job_title = st.text_input("Job Title", value="Python Developer")
        
        must_have_skills = st.text_area(
            "Must-Have Skills (comma-separated)",
            value="python, sql, git, api, django"
        ).split(',')
        must_have_skills = [skill.strip() for skill in must_have_skills if skill.strip()]
        
        good_to_have_skills = st.text_area(
            "Good-to-Have Skills (comma-separated)", 
            value="aws, docker, react, mongodb"
        ).split(',')
        good_to_have_skills = [skill.strip() for skill in good_to_have_skills if skill.strip()]
        
        # Analysis options
        st.markdown("### ⚙️ Options")
        show_skills_breakdown = st.checkbox("Show Skills Breakdown", value=True)
        show_missing_skills = st.checkbox("Show Missing Skills", value=True)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Read uploaded file
            text = str(uploaded_file.read(), "utf-8")
            
            if len(text.strip()) < 50:
                st.error("⚠️ Resume content seems too short. Please upload a proper resume file.")
                return
            
            # Progress
            with st.spinner("🔍 Analyzing resume..."):
                # Job requirements
                job_requirements = {
                    'must_have_skills': must_have_skills,
                    'good_to_have_skills': good_to_have_skills
                }
                
                # Analyze
                analysis = analyze_resume(text, job_requirements)
            
            st.success("✅ Analysis completed!")
            
            # Results Display
            st.markdown("## 📊 Analysis Results")
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>📈 Overall Score</h3>
                    <h2 style="color: #1f77b4;">{analysis['scores']['overall']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>{analysis['verdict_color']} Verdict</h3>
                    <h2>{analysis['verdict']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>🎯 Skills Found</h3>
                    <h2>{len(analysis['skills_found'])}</h2>
                    <p>technical skills</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                candidate_name = analysis['contact_info'].get('name', 'Unknown')
                st.markdown(f"""
                <div class="metric-box">
                    <h3>👤 Candidate</h3>
                    <h2>{candidate_name}</h2>
                    <p>{job_title}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Score Breakdown
            st.markdown("### 📊 Score Breakdown")
            scores = analysis['scores']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Must-Have Skills", f"{scores['required_skills']:.1f}%")
                st.metric("Experience", f"{scores['experience']:.1f}%")
            
            with col2:
                st.metric("Good-to-Have Skills", f"{scores['preferred_skills']:.1f}%")  
                st.metric("Education", f"{scores['education']:.1f}%")
            
            # Skills Analysis
            if show_skills_breakdown:
                st.markdown("### 🎯 Skills Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if analysis['required_matches']:
                        st.markdown("#### ✅ Matched Must-Have Skills")
                        for skill in analysis['required_matches']:
                            st.success(f"✓ {skill.title()}")
                    
                    if analysis['preferred_matches']:
                        st.markdown("#### ✅ Matched Good-to-Have Skills") 
                        for skill in analysis['preferred_matches']:
                            st.info(f"✓ {skill.title()}")
                
                with col2:
                    if show_missing_skills and analysis['missing_required']:
                        st.markdown("#### ⚠️ Missing Must-Have Skills")
                        for skill in analysis['missing_required']:
                            st.warning(f"❌ {skill.title()}")
            
            # Contact Information
            if analysis['contact_info']:
                st.markdown("### 📞 Contact Information")
                contact = analysis['contact_info']
                
                col1, col2 = st.columns(2)
                with col1:
                    if contact.get('email'):
                        st.info(f"📧 Email: {contact['email']}")
                with col2:
                    if contact.get('phone'):
                        st.info(f"📱 Phone: {contact['phone']}")
            
            # Resume Preview
            with st.expander("📄 Resume Content"):
                st.text_area("Full Text", text, height=300)
                
        except Exception as e:
            st.error(f"❌ Error processing resume: {str(e)}")
    
    else:
        # Demo Mode
        st.markdown("## 🎮 Demo Mode")
        st.info("👆 Upload a TXT resume file to start analysis!")
        
        # Sample Results
        st.markdown("### 📊 Sample Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", "82.5%", "Excellent")
        with col2:
            st.metric("Skills Match", "85%", "+15%")
        with col3:
            st.metric("Experience", "80%", "Good")
        with col4:
            st.metric("Education", "90%", "Strong")
        
        # Features
        st.markdown("### 🚀 Key Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### ✨ Core Capabilities
            - 📄 **Text Resume Processing**
            - 🎯 **Smart Skill Extraction**
            - 📊 **Automated Scoring**
            - 🔍 **Gap Analysis**
            - 👤 **Contact Detection**
            """)
        
        with col2:
            st.markdown("""
            #### ⚡ Vercel Optimized
            - 🚀 **Serverless Deployment**
            - ⚡ **Fast Loading**
            - 💰 **Cost Effective**
            - 🔒 **Secure Processing**
            - 📱 **Mobile Responsive**
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '<p><strong>Built for Innomatics Research Labs</strong> | '
        'AI-Powered Resume Intelligence | Vercel Deployment</p>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
