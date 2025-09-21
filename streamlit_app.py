#!/usr/bin/env python3
"""
Automated Resume Relevance Check System - Main Streamlit App
=============================================================

Streamlit Cloud compatible main application for Innomatics Research Labs.
This is the primary entry point for the cloud deployment.

Features:
- Resume parsing and analysis
- AI-powered scoring
- Interactive dashboard
- Cloud-optimized performance

Author: Naveen Kumar K M
Created for: Innomatics Research Labs
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path
import time
import traceback

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import with error handling for cloud deployment
try:
    from resume_parser.extract import extract_text_from_file
    from resume_parser.cleaner import normalize_text  
    from resume_parser.ner import extract_entities
    PARSER_AVAILABLE = True
except ImportError:
    try:
        from cloud_parser import extract_text_simple as extract_text_from_file
        from cloud_parser import normalize_text_simple as normalize_text
        from cloud_parser import extract_entities_simple as extract_entities
        PARSER_AVAILABLE = True
    except ImportError:
        PARSER_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Automated Resume Relevance Check System | Innomatics Labs",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #1f77b4; 
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-high { color: #28a745; font-weight: bold; font-size: 1.2rem; }
    .score-medium { color: #fd7e14; font-weight: bold; font-size: 1.2rem; }
    .score-low { color: #dc3545; font-weight: bold; font-size: 1.2rem; }
    .stAlert { margin: 1rem 0; }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def create_simple_chart(data_dict):
    """Create a simple bar chart for scores."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = px.bar(
        x=list(data_dict.keys()),
        y=list(data_dict.values()),
        title="Resume Analysis Scores",
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Metrics",
        yaxis_title="Scores"
    )
    return fig

def analyze_resume(text, job_requirements):
    """Analyze resume against job requirements."""
    
    # Simple scoring logic for cloud deployment
    score_breakdown = {}
    
    # Skills analysis
    candidate_skills = set()
    required_skills = set(job_requirements.get('must_have_skills', []))
    good_to_have_skills = set(job_requirements.get('good_to_have_skills', []))
    
    # Extract skills from text (simplified)
    text_lower = text.lower()
    for skill in required_skills:
        if skill.lower() in text_lower:
            candidate_skills.add(skill)
    
    for skill in good_to_have_skills:
        if skill.lower() in text_lower:
            candidate_skills.add(skill)
    
    # Calculate scores
    must_have_match = len(candidate_skills & required_skills)
    total_required = len(required_skills)
    must_have_score = (must_have_match / max(total_required, 1)) * 100
    
    good_to_have_match = len(candidate_skills & good_to_have_skills)
    total_good_to_have = len(good_to_have_skills)
    good_to_have_score = (good_to_have_match / max(total_good_to_have, 1)) * 100
    
    # Experience score (simplified)
    experience_keywords = ['experience', 'worked', 'developed', 'managed', 'led', 'created']
    experience_score = min(sum(1 for keyword in experience_keywords if keyword in text_lower) * 10, 100)
    
    # Education score (simplified)  
    education_keywords = ['degree', 'university', 'college', 'bachelor', 'master', 'phd']
    education_score = min(sum(1 for keyword in education_keywords if keyword in text_lower) * 15, 100)
    
    # Overall score (weighted)
    overall_score = (must_have_score * 0.4 + good_to_have_score * 0.2 + 
                    experience_score * 0.3 + education_score * 0.1)
    
    score_breakdown = {
        'Must-Have Skills': must_have_score,
        'Good-to-Have Skills': good_to_have_score,
        'Experience': experience_score,
        'Education': education_score,
        'Overall Score': overall_score
    }
    
    # Determine verdict
    if overall_score >= 80:
        verdict = "High"
        verdict_class = "score-high"
    elif overall_score >= 60:
        verdict = "Medium" 
        verdict_class = "score-medium"
    else:
        verdict = "Low"
        verdict_class = "score-low"
    
    return score_breakdown, verdict, verdict_class, {
        'matched_skills': list(candidate_skills),
        'missing_required': list(required_skills - candidate_skills),
        'total_skills_found': len(candidate_skills)
    }

def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Automated Resume Relevance Check System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"><strong>AI-Powered Resume Evaluation Engine | Built for Innomatics Research Labs</strong></p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📄 Upload Resume",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files (max 200MB)"
        )
        
        # Job requirements
        with st.expander("🎯 Job Requirements", expanded=True):
            job_title = st.text_input("Job Title", value="Python Developer")
            
            must_have_skills = st.text_area(
                "Must-Have Skills (comma-separated)",
                value="python, sql, git, api"
            ).split(',')
            must_have_skills = [skill.strip().lower() for skill in must_have_skills if skill.strip()]
            
            good_to_have_skills = st.text_area(
                "Good-to-Have Skills (comma-separated)",
                value="aws, docker, react, mongodb"
            ).split(',')
            good_to_have_skills = [skill.strip().lower() for skill in good_to_have_skills if skill.strip()]
            
            experience_required = st.selectbox(
                "Experience Required",
                ["0-2 years", "2-5 years", "5+ years"]
            )
        
        # Analysis settings
        with st.expander("⚙️ Settings"):
            show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
            show_missing_skills = st.checkbox("Show Missing Skills", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📊 System Status")
        st.success("✅ Parser Available" if PARSER_AVAILABLE else "❌ Parser Unavailable")
        st.success("✅ Charts Available" if PLOTLY_AVAILABLE else "❌ Charts Unavailable")
        st.info(f"📈 Ready for Analysis")
    
    with col1:
        if uploaded_file is not None:
            try:
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract text
                status_text.text("🔍 Extracting text from document...")
                progress_bar.progress(25)
                
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                elif PARSER_AVAILABLE:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    text = extract_text_from_file(temp_path)
                    os.remove(temp_path)
                else:
                    st.error("❌ PDF/DOCX parsing not available. Please upload a TXT file.")
                    return
                
                progress_bar.progress(50)
                status_text.text("🧠 Analyzing content...")
                
                # Job requirements structure
                job_requirements = {
                    'must_have_skills': must_have_skills,
                    'good_to_have_skills': good_to_have_skills,
                    'experience_required': experience_required
                }
                
                # Analyze resume
                scores, verdict, verdict_class, details = analyze_resume(text, job_requirements)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(1)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("### 📈 Analysis Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Overall Score</h4>
                        <h2 class="{verdict_class}">{scores['Overall Score']:.1f}/100</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>✅ Verdict</h4>
                        <h2 class="{verdict_class}">{verdict}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Skills Match</h4>
                        <h2>{details['total_skills_found']}</h2>
                        <p>skills found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📋 Job Role</h4>
                        <h3>{job_title}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed analysis
                if show_detailed_analysis:
                    st.markdown("### 📊 Detailed Score Breakdown")
                    
                    # Create chart if plotly available
                    if PLOTLY_AVAILABLE:
                        chart_data = {k: v for k, v in scores.items() if k != 'Overall Score'}
                        fig = create_simple_chart(chart_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Score table
                    score_df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
                    score_df['Score'] = score_df['Score'].round(1)
                    st.dataframe(score_df, use_container_width=True)
                
                # Skills analysis
                if show_missing_skills and details['missing_required']:
                    st.markdown("### ⚠️ Missing Critical Skills")
                    for skill in details['missing_required']:
                        st.warning(f"❌ {skill.title()}")
                
                # Matched skills
                if details['matched_skills']:
                    st.markdown("### ✅ Matched Skills")
                    skills_text = " • ".join([skill.title() for skill in details['matched_skills']])
                    st.success(f"🎯 {skills_text}")
                
                # Raw text preview
                with st.expander("📄 Extracted Text Preview"):
                    st.text_area("Resume Content", text[:1000] + "..." if len(text) > 1000 else text, height=200)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.code(traceback.format_exc())
        
        else:
            # Demo mode
            st.markdown("### 🎮 Demo Mode")
            st.info("👆 Upload a resume file to start analysis!")
            
            # Sample data
            st.markdown("#### 📊 Sample Analysis")
            sample_scores = {
                'Must-Have Skills': 85.0,
                'Good-to-Have Skills': 60.0,
                'Experience': 75.0,
                'Education': 90.0
            }
            
            if PLOTLY_AVAILABLE:
                fig = create_simple_chart(sample_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🚀 Features")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✨ Core Features")
                st.markdown("- 📄 PDF/DOCX/TXT file support")
                st.markdown("- 🧠 AI-powered text analysis") 
                st.markdown("- 📊 Real-time scoring")
                st.markdown("- 🎯 Skills gap identification")
            
            with col2:
                st.markdown("#### 🏗️ Technical Stack")
                st.markdown("- 🐍 Python + Streamlit")
                st.markdown("- 📊 Plotly visualizations")
                st.markdown("- 🧠 spaCy NLP processing")
                st.markdown("- ☁️ Cloud-ready deployment")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '<p><strong>Built for Innomatics Research Labs</strong> | '
        'AI-Powered Resume Intelligence | '
        'Scale • Consistency • Automation</p>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
