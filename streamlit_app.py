#!/usr/bin/env python3
"""
Automated Resume Relevance Check System - Streamlit Cloud Version
================================================================

Ultra-minimal standalone application for Streamlit Cloud deployment.
Built for Innomatics Research Labs.
"""

import streamlit as st
import pandas as pd
import re
import time
from typing import Dict, List, Any

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Built-in skill database for cloud deployment
TECHNICAL_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
    'html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab',
    'machine learning', 'ai', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
    'data science', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter',
    'api', 'rest', 'graphql', 'microservices', 'devops', 'ci/cd', 'agile', 'scrum'
}

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using simple matching."""
    text_lower = text.lower()
    found_skills = []
    for skill in TECHNICAL_SKILLS:
        if skill in text_lower:
            found_skills.append(skill)
    return sorted(found_skills)

def extract_basic_info(text: str) -> Dict[str, Any]:
    """Extract basic information from resume text."""
    info = {
        'name': None,
        'email': None, 
        'phone': None,
        'skills': [],
        'has_experience': False,
        'has_education': False
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        info['email'] = email_match.group()
    
    # Extract phone
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        info['phone'] = phone_match.group()
    
    # Extract name (simple heuristic)
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if len(line.split()) == 2 and line.replace(' ', '').isalpha() and len(line) < 50:
            if '@' not in line:
                info['name'] = line
                break
    
    # Extract skills
    info['skills'] = extract_skills_from_text(text)
    
    # Check for sections
    text_lower = text.lower()
    info['has_experience'] = bool(re.search(r'\b(experience|work|employment)\b', text_lower))
    info['has_education'] = bool(re.search(r'\b(education|university|college|degree)\b', text_lower))
    
    return info

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

def analyze_resume_simple(text: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Simplified resume analysis for cloud deployment."""
    
    # Extract basic info
    info = extract_basic_info(text)
    
    # Get job requirements
    required_skills = set([skill.lower().strip() for skill in job_requirements.get('must_have_skills', [])])
    good_to_have_skills = set([skill.lower().strip() for skill in job_requirements.get('good_to_have_skills', [])])
    
    # Calculate skill matches
    candidate_skills = set(info['skills'])
    must_have_matches = candidate_skills & required_skills
    good_to_have_matches = candidate_skills & good_to_have_skills
    
    # Calculate scores
    must_have_score = (len(must_have_matches) / max(len(required_skills), 1)) * 100
    good_to_have_score = (len(good_to_have_matches) / max(len(good_to_have_skills), 1)) * 100
    
    # Simple experience and education scoring
    experience_score = 70 if info['has_experience'] else 30
    education_score = 80 if info['has_education'] else 40
    
    # Overall weighted score
    overall_score = (must_have_score * 0.5 + good_to_have_score * 0.2 + 
                    experience_score * 0.2 + education_score * 0.1)
    
    # Determine verdict
    if overall_score >= 75:
        verdict = "High"
        verdict_class = "score-high"
    elif overall_score >= 50:
        verdict = "Medium"
        verdict_class = "score-medium"  
    else:
        verdict = "Low"
        verdict_class = "score-low"
    
    return {
        'info': info,
        'scores': {
            'Must-Have Skills': must_have_score,
            'Good-to-Have Skills': good_to_have_score, 
            'Experience': experience_score,
            'Education': education_score,
            'Overall Score': overall_score
        },
        'verdict': verdict,
        'verdict_class': verdict_class,
        'details': {
            'matched_skills': list(must_have_matches | good_to_have_matches),
            'missing_required': list(required_skills - candidate_skills),
            'total_skills_found': len(candidate_skills)
        }
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
        with st.expander("⚙️ Analysis Settings"):
            show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
            show_missing_skills = st.checkbox("Show Missing Skills", value=True)
        
        # NEW: Email Automation Settings
        with st.expander("📧 Email Automation", expanded=False):
            email_enabled = st.checkbox("Enable Email Automation", value=False, 
                                      help="Automatically send emails to candidates scoring 90%+")
            
            if email_enabled:
                col1, col2 = st.columns(2)
                with col1:
                    sender_email = st.text_input("Sender Email", placeholder="hr@innomatics.com")
                    hr_email = st.text_input("HR Email", placeholder="hr-team@innomatics.com")
                with col2:
                    score_threshold = st.slider("Score Threshold (%)", 80, 100, 90)
                    st.info(f"📬 Emails sent when score ≥ {score_threshold}%")
                
                if sender_email and hr_email:
                    sender_password = st.text_input("Email Password", type="password", 
                                                  help="Use App Password for Gmail")
                    if sender_password:
                        st.success("✅ Email automation configured!")
                else:
                    st.warning("⚠️ Please configure sender and HR emails")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 📊 System Status")
        st.success("✅ Resume Parser Ready")
        st.success("✅ Charts Available" if PLOTLY_AVAILABLE else "❌ Charts Limited")  
        st.success("✅ PDF/DOCX Support")
        st.info("📈 Ready for Analysis")
    
    with col1:
        if uploaded_file is not None:
            try:
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract text
                status_text.text("🔍 Extracting text from document...")
                progress_bar.progress(25)
                
                # Handle different file types
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    try:
                        import pdfplumber
                        with pdfplumber.open(uploaded_file) as pdf:
                            text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                    except Exception as e:
                        st.error(f"PDF parsing error: {e}")
                        text = "Error reading PDF. Please upload a TXT file."
                elif uploaded_file.name.endswith('.docx'):
                    try:
                        from docx import Document
                        from io import BytesIO
                        doc = Document(BytesIO(uploaded_file.read()))
                        text = ""
                        for paragraph in doc.paragraphs:
                            text += paragraph.text + "\n"
                    except Exception as e:
                        st.error(f"DOCX parsing error: {e}")  
                        text = "Error reading DOCX. Please upload a TXT file."
                else:
                    text = str(uploaded_file.read(), "utf-8")
                
                progress_bar.progress(66)
                status_text.text("🧠 Analyzing content...")
                
                # Job requirements structure
                job_requirements = {
                    'must_have_skills': must_have_skills,
                    'good_to_have_skills': good_to_have_skills,
                    'experience_required': experience_required
                }
                
                # Analyze resume
                analysis_result = analyze_resume_simple(text, job_requirements)
                
                progress_bar.progress(90)
                
                # NEW: Email automation processing
                email_sent = False
                if email_enabled and sender_email and hr_email and sender_password:
                    if analysis_result['scores']['Overall Score'] >= score_threshold:
                        status_text.text("📧 Sending automated emails...")
                        
                        try:
                            # Import email modules
                            from email_automation import EmailAutomation, EmailConfig, CandidateData
                            
                            # Create email config
                            email_config = EmailConfig(
                                sender_email=sender_email,
                                sender_password=sender_password,
                                hr_email=hr_email,
                                sender_name="Innomatics Research Labs"
                            )
                            
                            # Create candidate data
                            candidate = CandidateData(
                                name=analysis_result['info'].get('name', 'Unknown Candidate'),
                                email=analysis_result['info'].get('email', ''),
                                phone=analysis_result['info'].get('phone', ''),
                                overall_score=analysis_result['scores']['Overall Score'],
                                job_title=job_title,
                                matched_skills=analysis_result['details']['matched_skills'],
                                missing_skills=analysis_result['details']['missing_required'],
                                resume_text=text[:500] + "..." if len(text) > 500 else text
                            )
                            
                            # Send emails
                            email_system = EmailAutomation(email_config)
                            email_results = email_system.process_candidate_score(candidate, score_threshold)
                            email_sent = email_results['candidate_email_sent'] or email_results['hr_notification_sent']
                            
                        except Exception as e:
                            st.error(f"❌ Email automation failed: {str(e)}")
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("### 📈 Analysis Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                scores = analysis_result['scores']
                verdict = analysis_result['verdict']
                verdict_class = analysis_result['verdict_class']
                details = analysis_result['details']
                info = analysis_result['info']
                
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
                        <h4>🎯 Skills Found</h4>
                        <h2 class="score-good">{details['total_skills_found']}</h2>
                        <p>technical skills</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    candidate_name = info.get('name', 'Unknown')
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>👤 Candidate</h4>
                        <h3>{candidate_name}</h3>
                        <p>{job_title}</p>
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
                    
                    # NEW: Email automation status
                    if email_enabled:
                        st.markdown("### 📧 Email Automation Status")
                        if scores['Overall Score'] >= score_threshold:
                            if email_sent:
                                st.success(f"✅ Automated emails sent! Score: {scores['Overall Score']:.1f}% ≥ {score_threshold}%")
                                st.info("📬 Candidate and HR team have been notified")
                            else:
                                st.warning("⚠️ Score exceeds threshold but emails not sent (check configuration)")
                        else:
                            st.info(f"📊 Score {scores['Overall Score']:.1f}% below threshold ({score_threshold}%) - No emails sent")
                    
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
                
                # Contact info
                if info.get('email') or info.get('phone'):
                    st.markdown("### 📞 Contact Information")
                    if info.get('email'):
                        st.info(f"📧 Email: {info['email']}")
                    if info.get('phone'):
                        st.info(f"📱 Phone: {info['phone']}")
                
                # Raw text preview
                with st.expander("📄 Resume Text Preview"):
                    st.text_area("Content", text[:1000] + "..." if len(text) > 1000 else text, height=200)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)
        
        else:
            # Demo mode
            st.markdown("### 🎮 Demo Mode")
            st.info("👆 Upload a resume file to start analysis!")
            
            # Sample data
            st.markdown("#### 📊 Sample Analysis Results")
            sample_scores = {
                'Must-Have Skills': 85.0,
                'Good-to-Have Skills': 60.0,
                'Experience': 75.0,
                'Education': 90.0
            }
            
            # Sample metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", "78.5/100", "High Match")
            with col2:
                st.metric("Skills Found", "12", "technical skills")
            with col3:
                st.metric("Experience", "✓", "Detected")
            with col4:
                st.metric("Education", "✓", "Verified")
            
            if PLOTLY_AVAILABLE:
                fig = create_simple_chart(sample_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🚀 System Capabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✨ Core Features")
                st.markdown("- 📄 **Multi-format Support**: PDF, DOCX, TXT files")
                st.markdown("- 🧠 **Intelligent Parsing**: Extract contact info, skills, experience") 
                st.markdown("- 📊 **Real-time Scoring**: Instant relevance analysis")
                st.markdown("- 🎯 **Gap Analysis**: Identify missing critical skills")
                st.markdown("- 📈 **Visual Dashboard**: Interactive charts and metrics")
            
            with col2:
                st.markdown("#### 🏗️ Technical Architecture")
                st.markdown("- 🐍 **Python Backend**: Streamlit + pandas + plotly")
                st.markdown("- 📊 **Advanced Visualization**: Interactive Plotly charts")
                st.markdown("- 🔍 **Text Processing**: Regex + pattern matching")
                st.markdown("- ☁️ **Cloud Optimized**: Streamlit Cloud ready")
                st.markdown("- 🎨 **Modern UI**: Responsive design with custom CSS")
                
            # Sample resume format
            with st.expander("📄 Supported Resume Formats"):
                st.markdown("""
                **Supported File Types:**
                - 📄 **PDF files** - Extracted using pdfplumber
                - 📝 **Word documents (.docx)** - Parsed with python-docx  
                - 📜 **Plain text (.txt)** - Direct text processing
                
                **Information Extracted:**
                - 👤 Name and contact details
                - 🎯 Technical skills and competencies
                - 💼 Work experience and roles
                - 🎓 Educational background
                - 📧 Email addresses and phone numbers
                """)
    
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
