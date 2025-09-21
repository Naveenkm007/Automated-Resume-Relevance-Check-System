#!/usr/bin/env python3
"""
Email Automation Demo
====================

Demonstrates the automatic email feature for candidates scoring 90%+.
Built for Innomatics Research Labs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from enhanced_scoring_with_email import EnhancedResumeAnalyzer
from email_config import EmailSettings, setup_email_config, load_config_from_env
import streamlit as st

def main():
    """Main demo function."""
    
    st.set_page_config(
        page_title="Email Automation Demo | Innomatics Labs",
        page_icon="📧",
        layout="wide"
    )
    
    # Header
    st.markdown("# 📧 Email Automation Demo")
    st.markdown("**Automatic email notifications for high-scoring candidates (90%+)**")
    st.markdown("---")
    
    # Configuration section
    st.markdown("## ⚙️ Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📮 Email Settings")
        
        # Try to load existing config
        existing_config = load_config_from_env()
        
        sender_email = st.text_input(
            "Sender Email", 
            value=existing_config.sender_email if existing_config else "",
            placeholder="hr@innomatics.com"
        )
        
        sender_password = st.text_input(
            "Email Password", 
            type="password",
            help="Use App Password for Gmail"
        )
        
        hr_email = st.text_input(
            "HR Team Email",
            value=existing_config.hr_email if existing_config else "",
            placeholder="hr-team@innomatics.com"
        )
        
    with col2:
        st.markdown("### 🎯 Automation Settings")
        
        score_threshold = st.slider(
            "Score Threshold (%)",
            min_value=80,
            max_value=100,
            value=int(existing_config.score_threshold) if existing_config else 90,
            help="Minimum score to trigger automatic emails"
        )
        
        auto_send = st.checkbox(
            "Enable Auto-Send",
            value=existing_config.auto_send_enabled if existing_config else True
        )
        
        test_mode = st.checkbox(
            "Test Mode (send to sender email only)",
            value=True,
            help="For testing, sends all emails to sender address"
        )
    
    # Demo section
    st.markdown("---")
    st.markdown("## 🧪 Demo Resume Analysis")
    
    # Sample resumes
    sample_resumes = {
        "High Score Candidate (95%)": """
John Smith
john.smith@example.com
+91-9876543210

SENIOR PYTHON DEVELOPER

SUMMARY:
Experienced Python developer with 6+ years of expertise in Django, PostgreSQL, 
REST APIs, AWS, Docker, and React. Led multiple projects and mentored junior developers.

SKILLS:
• Programming: Python, JavaScript, TypeScript, SQL
• Frameworks: Django, Flask, React, Node.js
• Databases: PostgreSQL, MongoDB, Redis
• Cloud & DevOps: AWS, Docker, Kubernetes, Jenkins
• Tools: Git, Linux, API development

EXPERIENCE:
Senior Python Developer | TechCorp (2020-2024)
- Led development of microservices architecture using Django and PostgreSQL
- Implemented REST APIs serving 10M+ requests daily
- Deployed applications on AWS using Docker and Kubernetes
- Mentored 5 junior developers

Python Developer | StartupXYZ (2018-2020)
- Developed web applications using Django and React
- Managed PostgreSQL databases and optimized queries
- Implemented CI/CD pipelines with Jenkins

EDUCATION:
B.Tech Computer Science | ABC University (2018)
""",
        
        "Medium Score Candidate (75%)": """
Sarah Johnson  
sarah.j@email.com
+91-8765432109

PYTHON DEVELOPER

SUMMARY:
Python developer with 3 years of experience. Skilled in web development and databases.

SKILLS:
• Python, HTML, CSS, JavaScript
• Django, PostgreSQL
• Git, Linux

EXPERIENCE:
Python Developer | WebCompany (2021-2024)
- Developed web applications using Django
- Worked with PostgreSQL databases
- Bug fixes and feature development

Junior Developer | LocalFirm (2020-2021)  
- Learning Python and web development
- Basic database operations

EDUCATION:
B.Sc Computer Science | XYZ College (2020)
""",
        
        "Low Score Candidate (45%)": """
Mike Wilson
mike.w@email.com  
+91-7654321098

DEVELOPER

SUMMARY:
Recent graduate looking for opportunities in software development.

SKILLS:
• Basic Python, Java
• HTML, CSS
• Microsoft Office

EDUCATION:
B.Tech Information Technology | DEF University (2023)

PROJECTS:
- Calculator app in Java
- Simple website using HTML/CSS
"""
    }
    
    selected_resume = st.selectbox(
        "Select Sample Resume",
        options=list(sample_resumes.keys())
    )
    
    # Job requirements
    st.markdown("### 💼 Job Requirements")
    col1, col2 = st.columns(2)
    
    with col1:
        job_title = st.text_input("Job Title", value="Senior Python Developer")
        must_have_skills = st.text_area(
            "Must-Have Skills (comma-separated)",
            value="python, django, postgresql, rest api, aws"
        ).split(',')
        must_have_skills = [skill.strip().lower() for skill in must_have_skills if skill.strip()]
        
    with col2:
        good_to_have_skills = st.text_area(
            "Good-to-Have Skills (comma-separated)",
            value="docker, react, kubernetes, mongodb"
        ).split(',')
        good_to_have_skills = [skill.strip().lower() for skill in good_to_have_skills if skill.strip()]
    
    # Analysis button
    if st.button("🔍 Analyze Resume", type="primary"):
        if not sender_email or not sender_password or not hr_email:
            st.error("❌ Please configure all email settings first!")
            return
            
        # Create job requirements
        job_requirements = {
            'job_title': job_title,
            'must_have_skills': must_have_skills,
            'good_to_have_skills': good_to_have_skills,
            'experience_required': 3
        }
        
        # Save sample resume to temp file
        resume_text = sample_resumes[selected_resume]
        temp_file = "temp_resume.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(resume_text)
        
        try:
            # Create email config
            from email_automation import EmailConfig
            
            email_config = EmailConfig(
                sender_email=sender_email,
                sender_password=sender_password,
                hr_email=hr_email if not test_mode else sender_email,  # Use sender for test mode
                sender_name="Innomatics Research Labs - Demo"
            )
            
            # Analyze resume with email automation
            analyzer = EnhancedResumeAnalyzer(email_config)
            
            with st.spinner("🔍 Analyzing resume and processing emails..."):
                results = analyzer.analyze_resume_with_email(
                    temp_file,
                    job_requirements,
                    send_emails=auto_send,
                    score_threshold=score_threshold
                )
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            score = results['scoring_results']['overall_score']
            verdict = results['scoring_results']['verdict']
            
            with col1:
                color = "🟢" if score >= 90 else "🟡" if score >= 70 else "🔴"
                st.metric("Overall Score", f"{score}%", delta=f"{verdict}")
                st.markdown(f"{color} **{verdict}**")
            
            with col2:
                matched = len(results['scoring_results']['matched_skills'])
                st.metric("Matched Skills", matched)
            
            with col3:
                missing = len(results['scoring_results']['missing_skills']) 
                st.metric("Missing Skills", missing)
            
            with col4:
                threshold_met = results['email_automation']['threshold_met']
                st.metric("Threshold Met", "✅ YES" if threshold_met else "❌ NO")
            
            # Email automation results
            st.markdown("### 📧 Email Automation Results")
            
            email_results = results['email_automation']
            
            if email_results['threshold_met']:
                st.success(f"🎯 **Score {score}% exceeds threshold {score_threshold}%**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if email_results['candidate_email_sent']:
                        st.success("✅ **Candidate email sent successfully!**")
                        st.info("📧 Congratulations email delivered to candidate")
                    else:
                        st.error("❌ **Candidate email failed**")
                
                with col2:
                    if email_results['hr_notification_sent']:
                        st.success("✅ **HR notification sent successfully!**") 
                        st.info("🚨 HR team alerted about high-scoring candidate")
                    else:
                        st.error("❌ **HR notification failed**")
                        
            else:
                st.info(f"📊 **Score {score}% below threshold {score_threshold}%** - No emails sent")
            
            # Detailed analysis
            with st.expander("📋 Detailed Analysis Results"):
                st.json(results)
            
            # Email templates preview
            if email_results['threshold_met']:
                with st.expander("📧 Email Templates Preview"):
                    from email_automation import EmailAutomation, CandidateData
                    
                    candidate = CandidateData(
                        name=results['candidate_info']['name'],
                        email=results['candidate_info']['email'],
                        phone=results['candidate_info']['phone'],
                        overall_score=score,
                        job_title=job_title,
                        matched_skills=results['scoring_results']['matched_skills'],
                        missing_skills=results['scoring_results']['missing_skills'],
                        resume_text=resume_text[:500] + "..."
                    )
                    
                    email_system = EmailAutomation(email_config)
                    
                    st.markdown("**Candidate Email:**")
                    candidate_template = email_system.generate_candidate_email_template(candidate)
                    st.components.v1.html(candidate_template, height=600, scrolling=True)
                    
                    st.markdown("**HR Notification:**")
                    hr_template = email_system.generate_hr_notification_template(candidate)
                    st.components.v1.html(hr_template, height=600, scrolling=True)
            
        except Exception as e:
            st.error(f"❌ **Analysis failed:** {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Instructions
    st.markdown("---")
    st.markdown("## 📝 Setup Instructions")
    
    with st.expander("🔧 How to Configure Email Automation"):
        st.markdown("""
        ### Gmail Configuration (Recommended):
        
        1. **Enable 2-Factor Authentication** on your Gmail account
        2. **Generate App Password:**
           - Go to Gmail → Settings → Security → 2-Step Verification 
           - Select "App passwords" → Generate password for "Mail"
           - Use this app password (not your regular password)
        
        3. **Enter Configuration:**
           - **Sender Email:** Your Gmail address (e.g., hr@innomatics.com)
           - **Password:** The app password generated above
           - **HR Email:** Where notifications should be sent
        
        ### Other Email Providers:
        - **Outlook:** Use SMTP: smtp-mail.outlook.com, Port: 587
        - **Yahoo:** Use SMTP: smtp.mail.yahoo.com, Port: 587
        - **Custom SMTP:** Configure in email_automation.py
        
        ### Security Notes:
        - Never commit passwords to version control
        - Use environment variables in production
        - Consider using OAuth2 for enhanced security
        """)
    
    with st.expander("🎯 How the Automation Works"):
        st.markdown("""
        ### Automated Email Flow:
        
        1. **Resume Analysis:** System calculates overall score based on:
           - Required skills match (50% weight)
           - Preferred skills match (20% weight)  
           - Experience level (20% weight)
           - Education background (10% weight)
        
        2. **Score Evaluation:** If score ≥ threshold (default 90%):
           - **Candidate Email:** Congratulations email with score and next steps
           - **HR Notification:** Alert with candidate details and recommendations
        
        3. **Email Content:**
           - **Professional templates** with Innomatics branding
           - **Personalized content** with candidate name and scores
           - **Actionable insights** for HR team follow-up
        
        ### Benefits:
        - ⚡ **Immediate response** to high-quality candidates
        - 🎯 **Consistent communication** across all applications  
        - 📊 **Data-driven decisions** based on AI scoring
        - 🚀 **Competitive advantage** in candidate acquisition
        """)

if __name__ == "__main__":
    main()
