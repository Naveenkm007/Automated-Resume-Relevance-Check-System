#!/usr/bin/env python3
"""
Simple Resume Analysis Dashboard
A standalone Streamlit app that works without backend dependencies
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from resume_parser.extract import extract_text_from_file
    from resume_parser.cleaner import normalize_text  
    from resume_parser.ner import extract_entities
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Resume Analysis Dashboard",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; margin-bottom: 1rem; }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #fd7e14; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .stAlert { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Resume Parser & Analysis** - Upload resumes to extract structured data")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        if PARSER_AVAILABLE:
            st.success("✅ Resume Parser: Ready")
            st.success("✅ NLP Models: Loaded")
            st.success("✅ Text Extraction: Available")
        else:
            st.error("❌ Parser modules not found")
            st.info("Run: pip install -r requirements.txt")
        
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [GitHub Repo](https://github.com/Naveenkm007/Automated-Resume-Relevance-Check-System)")
        st.markdown("- [Documentation](README.md)")
        st.markdown("- [API Demo](demo.py)")
    
    # Main content
    if not PARSER_AVAILABLE:
        st.error("⚠️ Resume parser modules are not available. Please install dependencies:")
        st.code("pip install -r requirements.txt")
        st.code("python -m spacy download en_core_web_sm")
        return
    
    # File upload section
    st.markdown("### 📁 Upload Resume")
    
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx', 'doc'],
        help="Upload PDF or Word document resumes"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the file
        with st.spinner("🔄 Processing resume..."):
            try:
                # Extract text
                raw_text = extract_text_from_file(temp_path)
                
                # Normalize and clean
                sections = normalize_text(raw_text)
                
                # Extract entities
                entities = extract_entities(sections.get('full_text', raw_text))
                
                # Display results
                st.success("✅ Resume processed successfully!")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 Experience", "🎓 Education", "📝 Raw Data"])
                
                with tab1:
                    st.markdown("#### 📊 Candidate Overview")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📧 Contact Information:**")
                        if entities.get('name'):
                            st.write(f"**Name:** {entities['name']}")
                        if entities.get('email'):
                            st.write(f"**Email:** {entities['email']}")
                        if entities.get('phone'):
                            st.write(f"**Phone:** {entities['phone']}")
                    
                    with col2:
                        st.markdown("**🔧 Skills Summary:**")
                        skills = entities.get('skills', [])
                        if skills:
                            st.write(f"**Total Skills:** {len(skills)}")
                            st.write(f"**Top Skills:** {', '.join(skills[:5])}")
                        else:
                            st.write("No skills detected")
                
                with tab2:
                    st.markdown("#### 💼 Work Experience")
                    experience = entities.get('experience', [])
                    
                    if experience:
                        for i, exp in enumerate(experience, 1):
                            with st.expander(f"Position {i}: {exp.get('title', 'Unknown Title')}"):
                                st.write(f"**Company:** {exp.get('company', 'Unknown')}")
                                st.write(f"**Duration:** {exp.get('start', 'Unknown')} - {exp.get('end', 'Unknown')}")
                                if exp.get('bullets'):
                                    st.write("**Responsibilities:**")
                                    for bullet in exp['bullets'][:5]:  # Show first 5
                                        st.write(f"• {bullet}")
                    else:
                        st.info("No work experience detected")
                
                with tab3:
                    st.markdown("#### 🎓 Education")
                    education = entities.get('education', [])
                    
                    if education:
                        for i, edu in enumerate(education, 1):
                            with st.expander(f"Education {i}: {edu.get('degree', 'Unknown Degree')}"):
                                st.write(f"**Institution:** {edu.get('institution', 'Unknown')}")
                                st.write(f"**Year:** {edu.get('year', 'Unknown')}")
                                if edu.get('stream'):
                                    st.write(f"**Stream:** {edu['stream']}")
                    else:
                        st.info("No education information detected")
                
                with tab4:
                    st.markdown("#### 📝 Complete Parsed Data")
                    st.json(entities)
                
                # Skills visualization
                if skills:
                    st.markdown("#### 🔧 Skills Analysis")
                    
                    # Create a simple skills chart
                    skills_df = pd.DataFrame({
                        'Skill': skills[:10],  # Top 10 skills
                        'Detected': [1] * min(10, len(skills))
                    })
                    
                    st.bar_chart(skills_df.set_index('Skill'))
                
            except Exception as e:
                st.error(f"❌ Error processing resume: {e}")
                st.info("💡 Try a different file format or check if the file is corrupted")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Demo section
    st.markdown("---")
    st.markdown("### 🎯 Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run System Validation"):
            with st.spinner("Running validation..."):
                import subprocess
                try:
                    result = subprocess.run(['python', 'validate_system.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ System validation passed!")
                        st.code(result.stdout)
                    else:
                        st.error("❌ System validation failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"Error running validation: {e}")
    
    with col2:
        if st.button("📊 Run Demo Analysis"):
            with st.spinner("Running demo..."):
                import subprocess
                try:
                    result = subprocess.run(['python', 'demo.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ Demo completed!")
                        st.code(result.stdout)
                    else:
                        st.error("❌ Demo failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"Error running demo: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Automated Resume Relevance Check System** | Built with Streamlit & Python")

if __name__ == "__main__":
    main()
