#!/usr/bin/env python3
"""
Resume Relevance Check - Placement Team Dashboard

A Streamlit-based MVP dashboard for placement teams to:
- Upload job descriptions
- View and filter resumes by score, verdict, role
- View detailed evaluation results
- Export shortlisted candidates to CSV

Usage:
    streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import io
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_API_TOKEN = "your-api-token-here"  # Set from environment or config

# Page configuration
st.set_page_config(
    page_title="Resume Relevance Check Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; margin-bottom: 1rem; }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #fd7e14; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

class APIClient:
    """API client for backend communication."""
    
    def __init__(self, base_url: str, api_token: str = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    
    def upload_jd(self, jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload job description via JSON API."""
        try:
            response = requests.post(
                f"{self.base_url}/upload-jd",
                data={"jd_data": json.dumps(jd_data)},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_resumes(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search and filter resumes."""
        try:
            params = filters or {}
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_evaluation_results(self, resume_id: str) -> Dict[str, Any]:
        """Get detailed evaluation results for a resume."""
        try:
            response = requests.get(
                f"{self.base_url}/results/{resume_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

def get_api_client() -> APIClient:
    """Get configured API client."""
    api_token = st.session_state.get('api_token', DEFAULT_API_TOKEN)
    return APIClient(API_BASE_URL, api_token)

def render_score_badge(score: Optional[int], verdict: Optional[str]) -> str:
    """Render colored score badge based on verdict."""
    if score is None or verdict is None:
        return "⚪ Not Evaluated"
    
    if verdict == "high":
        return f"🟢 High ({score}/100)"
    elif verdict == "medium":  
        return f"🟡 Medium ({score}/100)"
    else:
        return f"🔴 Low ({score}/100)"

def upload_jd_section():
    """Job Description upload section."""
    st.markdown("### 📝 Upload Job Description")
    
    with st.form("jd_upload_form", clear_on_submit=False):
        st.markdown("**Job Information**")
        
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Job Title*", placeholder="e.g. Senior Python Developer", key="jd_title")
            company = st.text_input("Company Name*", placeholder="e.g. TechCorp Inc.", key="jd_company")
        
        with col2:
            location = st.text_input("Location", placeholder="e.g. San Francisco, CA", key="jd_location")
        
        description = st.text_area(
            "Job Description*", 
            placeholder="Paste the complete job description here...",
            height=150,
            key="jd_description"
        )
        
        st.markdown("**Requirements (Optional - can be extracted automatically)**")
        
        col3, col4 = st.columns(2)
        with col3:
            must_have = st.text_area(
                "Must-Have Skills", 
                placeholder="python, django, postgresql (comma-separated)",
                key="jd_must_have"
            )
        
        with col4:
            good_to_have = st.text_area(
                "Good-to-Have Skills",
                placeholder="aws, docker, react (comma-separated)",  
                key="jd_good_to_have"
            )
        
        submitted = st.form_submit_button("📤 Upload Job Description", type="primary")
        
        if submitted:
            if not title or not company or not description:
                st.error("Please fill in all required fields (Title, Company, Description)")
                return
            
            # Prepare JD data
            jd_data = {
                "title": title,
                "company": company,
                "location": location or "",
                "description": description,
                "must_have_skills": [s.strip() for s in must_have.split(",") if s.strip()] if must_have else [],
                "good_to_have_skills": [s.strip() for s in good_to_have.split(",") if s.strip()] if good_to_have else [],
            }
            
            # Upload via API
            with st.spinner("Uploading job description..."):
                client = get_api_client()
                result = client.upload_jd(jd_data)
                
                if result["success"]:
                    data = result["data"]
                    st.success(f"✅ Job description uploaded successfully!")
                    st.info(f"**JD ID:** {data['jd_id']}")
                    
                    # Clear form
                    for key in ['jd_title', 'jd_company', 'jd_location', 'jd_description', 'jd_must_have', 'jd_good_to_have']:
                        if key in st.session_state:
                            del st.session_state[key]
                else:
                    st.error(f"❌ Upload failed: {result['error']}")

def resume_filter_section():
    """Resume filtering controls."""
    st.markdown("### 🔍 Filter Resumes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        role_filter = st.text_input("Role/Title", placeholder="e.g. python, developer", key="role_filter")
    
    with col2:
        min_score = st.number_input("Min Score", min_value=0, max_value=100, value=0, key="min_score_filter")
    
    with col3:
        verdict_filter = st.selectbox("Verdict", ["All", "high", "medium", "low"], key="verdict_filter")
    
    with col4:
        location_filter = st.text_input("Location", placeholder="e.g. san francisco", key="location_filter")
    
    # Build filter parameters
    filters = {}
    if role_filter:
        filters["role"] = role_filter
    if min_score > 0:
        filters["min_score"] = min_score
    if verdict_filter != "All":
        filters["verdict"] = verdict_filter
    if location_filter:
        filters["location"] = location_filter
    
    return filters

def resume_list_section(filters: Dict[str, Any]):
    """Resume list with filtering."""
    st.markdown("### 👥 Resume List")
    
    # Fetch resumes
    client = get_api_client()
    
    with st.spinner("Loading resumes..."):
        result = client.search_resumes(filters)
        
        if not result["success"]:
            st.error(f"Failed to load resumes: {result['error']}")
            return
        
        resumes = result["data"]
    
    if not resumes:
        st.info("No resumes found matching the current filters.")
        return
    
    # Convert to DataFrame for better display
    df_data = []
    for resume in resumes:
        df_data.append({
            "Resume ID": resume["resume_id"],
            "Candidate": resume["candidate_name"] or "Unknown",
            "Filename": resume["filename"],
            "Score": resume["final_score"],
            "Verdict": resume["verdict"],
            "Job Title": resume["jd_title"],
            "Company": resume["jd_company"],
            "Date": resume["created_at"][:10] if resume["created_at"] else "Unknown"
        })
    
    df = pd.DataFrame(df_data)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Total Resumes</h4>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_count = len(df[df["Verdict"] == "high"])
        st.markdown(f"""
        <div class="metric-card">
            <h4>🟢 High Quality</h4>
            <h2 class="score-high">{high_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        medium_count = len(df[df["Verdict"] == "medium"])  
        st.markdown(f"""
        <div class="metric-card">
            <h4>🟡 Medium Quality</h4>
            <h2 class="score-medium">{medium_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = df["Score"].mean() if not df["Score"].isna().all() else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Avg Score</h4>
            <h2>{avg_score:.1f}/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CSV Export functionality
    col_export, col_threshold = st.columns([1, 3])
    
    with col_export:
        if st.button("📥 Export to CSV"):
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"resumes_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_threshold:
        export_threshold = st.slider("Export Score Threshold", 0, 100, 70, help="Only export resumes above this score")
    
    # Resume table with clickable rows
    st.markdown("**Click on a resume to view detailed evaluation:**")
    
    for idx, row in df.iterrows():
        with st.container():
            col_info, col_score, col_action = st.columns([3, 1, 1])
            
            with col_info:
                st.write(f"**{row['Candidate']}** - {row['Filename']}")
                st.write(f"Applied for: {row['Job Title']} at {row['Company']} | Date: {row['Date']}")
            
            with col_score:
                badge = render_score_badge(row['Score'], row['Verdict'])
                st.markdown(badge)
                
                if row['Score'] is not None:
                    st.progress(row['Score'] / 100)
            
            with col_action:
                if st.button(f"View Details", key=f"view_{row['Resume ID']}"):
                    st.session_state['selected_resume'] = row['Resume ID']
                    st.experimental_rerun()
        
        st.markdown("---")

def evaluation_detail_section(resume_id: str):
    """Detailed evaluation results for a specific resume."""
    st.markdown(f"### 📄 Evaluation Details")
    
    client = get_api_client()
    
    with st.spinner("Loading evaluation results..."):
        result = client.get_evaluation_results(resume_id)
        
        if not result["success"]:
            st.error(f"Failed to load evaluation: {result['error']}")
            return
        
        eval_data = result["data"]
    
    if eval_data["status"] != "completed":
        st.warning(f"Evaluation status: {eval_data['status']}")
        if eval_data.get("error_message"):
            st.error(f"Error: {eval_data['error_message']}")
        return
    
    # Score summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hard Score", f"{eval_data['hard_score']:.1f}/100")
    
    with col2:
        st.metric("Semantic Score", f"{eval_data['semantic_score']:.1f}/100")  
    
    with col3:
        final_score = eval_data['final_score']
        verdict = eval_data['verdict']
        st.metric("Final Score", f"{final_score}/100", f"Verdict: {verdict.upper()}")
    
    # Detailed breakdown
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🎯 Missing Skills")
        missing = eval_data.get("missing_elements", [])
        
        if missing:
            for element in missing:
                st.markdown(f"**{element['type'].replace('_', ' ').title()}:**")
                for item in element['items'][:5]:  # Show top 5
                    st.markdown(f"• {item}")
        else:
            st.success("No critical missing elements!")
    
    with col_right:
        st.markdown("#### 💡 Improvement Suggestions")
        suggestions = eval_data.get("feedback_suggestions", [])
        
        if suggestions:
            for i, suggestion in enumerate(suggestions[:3], 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion['priority'], "⚪")
                st.markdown(f"**{i}. {priority_emoji} {suggestion['action']}**")
                st.markdown(f"   *Example: {suggestion['example']}*")
        else:
            st.info("No specific suggestions generated.")
    
    # Parsed resume data (if available)
    if eval_data.get("parsed_resume"):
        with st.expander("📋 Parsed Resume Data", expanded=False):
            parsed = eval_data["parsed_resume"]
            
            if parsed.get("skills"):
                st.markdown("**Skills:**")
                st.write(", ".join(parsed["skills"][:20]))  # Show first 20 skills
            
            if parsed.get("experience"):
                st.markdown("**Experience:**")
                for exp in parsed["experience"][:3]:  # Show first 3 jobs
                    st.write(f"• {exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')}")
    
    # Back button
    if st.button("← Back to Resume List"):
        if 'selected_resume' in st.session_state:
            del st.session_state['selected_resume']
        st.experimental_rerun()

def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">📋 Resume Relevance Check Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Placement Team Portal** - Upload JDs, evaluate resumes, and find the best candidates")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API configuration
        api_token = st.text_input("API Token", value=DEFAULT_API_TOKEN, type="password", key="api_token")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        try:
            response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                st.metric("Total Resumes", stats.get("total_resumes", 0))
                st.metric("Total JDs", stats.get("total_jds", 0))
                st.metric("Evaluations", stats.get("total_evaluations", 0))
        except:
            st.error("Could not load stats")
        
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [API Docs](http://localhost:8000/docs)")
        st.markdown("- [Worker Monitor](http://localhost:5555)")
        st.markdown("- [System Health](http://localhost:8000/health)")
    
    # Main content area
    if st.session_state.get('selected_resume'):
        # Show detailed evaluation
        evaluation_detail_section(st.session_state['selected_resume'])
    else:
        # Show main dashboard
        
        # Job Description Upload
        upload_jd_section()
        
        st.markdown("---")
        
        # Resume Filtering and List
        filters = resume_filter_section()
        resume_list_section(filters)

if __name__ == "__main__":
    main()
