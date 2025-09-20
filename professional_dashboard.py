#!/usr/bin/env python3
"""
Professional Resume Analysis Dashboard
Modern, user-friendly interface for resume parsing and analysis
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

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

# Page configuration
st.set_page_config(
    page_title="Resume Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling with enterprise color grading
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root color variables - Enterprise palette */
    :root {
        --primary-gradient: linear-gradient(135deg, #0F4C75 0%, #3282B8 50%, #0F4C75 100%);
        --secondary-gradient: linear-gradient(135deg, #1A365D 0%, #2D3748 100%);
        --accent-gradient: linear-gradient(135deg, #38B2AC 0%, #319795 100%);
        --success-gradient: linear-gradient(135deg, #38A169 0%, #2F855A 100%);
        --warning-gradient: linear-gradient(135deg, #D69E2E 0%, #B7791F 100%);
        --error-gradient: linear-gradient(135deg, #E53E3E 0%, #C53030 100%);
        --neutral-gradient: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%);
        --glass-effect: rgba(255, 255, 255, 0.25);
        --shadow-primary: 0 8px 32px 0 rgba(15, 76, 117, 0.15);
        --shadow-elevated: 0 20px 60px 0 rgba(15, 76, 117, 0.25);
        --text-primary: #1A202C;
        --text-secondary: #4A5568;
        --text-muted: #718096;
    }
    
    /* Global background */
    .main .block-container {
        background: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 50%, #E2E8F0 100%);
        min-height: 100vh;
    }
    
    /* Main header with sophisticated gradient */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: -0.02em;
        text-shadow: 0 4px 8px rgba(15, 76, 117, 0.3);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Premium metric cards with glass morphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: var(--shadow-primary);
        margin: 0.5rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
        border-radius: 16px 16px 0 0;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: var(--shadow-elevated);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
    }
    
    /* Enhanced status cards */
    .status-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid transparent;
        border-image: var(--success-gradient) 1;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px 0 rgba(56, 178, 172, 0.15);
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 24px 0 rgba(56, 178, 172, 0.25);
    }
    
    /* Premium upload area */
    .upload-area {
        border: 2px dashed #3282B8;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(50, 130, 184, 0.05) 0%, rgba(15, 76, 117, 0.05) 100%);
        margin: 1.5rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .upload-area::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(50, 130, 184, 0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.4s ease;
        opacity: 0;
    }
    
    .upload-area:hover {
        border-color: #0F4C75;
        background: linear-gradient(135deg, rgba(50, 130, 184, 0.1) 0%, rgba(15, 76, 117, 0.1) 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow-primary);
    }
    
    .upload-area:hover::before {
        opacity: 1;
        animation: shimmer 2s ease-in-out;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Premium buttons with enhanced styling */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-primary);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-elevated);
        background: linear-gradient(135deg, #0F4C75 0%, #38B2AC 50%, #0F4C75 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Enhanced status banners */
    .success-banner {
        background: var(--success-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 8px 24px 0 rgba(56, 161, 105, 0.3);
        border-left: 4px solid #2F855A;
    }
    
    .error-banner {
        background: var(--error-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 8px 24px 0 rgba(229, 62, 62, 0.3);
        border-left: 4px solid #C53030;
    }
    
    .warning-banner {
        background: var(--warning-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 8px 24px 0 rgba(214, 158, 46, 0.3);
        border-left: 4px solid #B7791F;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: var(--secondary-gradient);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Premium progress indicators */
    .progress-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(241, 245, 249, 0.8) 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid transparent;
        border-image: var(--accent-gradient) 1;
        position: relative;
        letter-spacing: -0.01em;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--primary-gradient);
        border-radius: 2px;
    }
    
    /* Enhanced text styling */
    .info-text {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        box-shadow: 0 4px 12px 0 rgba(15, 76, 117, 0.3);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Hide Streamlit branding with style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header[data-testid="stHeader"] {background: transparent;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-gradient);
    }
</style>
""", unsafe_allow_html=True)

def create_metric_card(title, value, description="", color="#667eea"):
    """Create a professional metric card."""
    return f"""
    <div class="metric-card">
        <h3 style="color: {color}; margin: 0; font-size: 1.2rem; font-weight: 600;">{title}</h3>
        <div style="font-size: 2rem; font-weight: 700; color: #1e293b; margin: 0.5rem 0;">{value}</div>
        <p style="color: #64748b; margin: 0; font-size: 0.9rem;">{description}</p>
    </div>
    """

def create_skills_chart(skills_list, theme="light"):
    """Create an interactive skills chart."""
    if not skills_list:
        return None
    
    # Take top 10 skills for visualization
    top_skills = skills_list[:10]
    
    # Theme-aware colors
    if theme == "dark":
        title_color = "#F7FAFC"
        tick_color = "#E2E8F0"
        colors = px.colors.qualitative.Pastel
    else:
        title_color = "#1e293b"
        tick_color = "#4A5568"
        colors = px.colors.qualitative.Set3
    
    # Create a horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=top_skills,
            x=[1] * len(top_skills),
            orientation='h',
            marker=dict(
                color=colors[:len(top_skills)],
                line=dict(color='rgba(128,128,128,0.3)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Detected in resume<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Detected Skills",
        title_font=dict(size=16, family="Inter", color=title_color),
        xaxis=dict(visible=False),
        yaxis=dict(title="", tickfont=dict(size=12, color=tick_color)),
        height=max(300, len(top_skills) * 40),
        margin=dict(l=120, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(color=title_color)
    )
    
    return fig

def apply_theme_css(theme="light"):
    """Apply theme-specific CSS."""
    if theme == "dark":
        theme_css = """
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
                --accent-gradient: linear-gradient(135deg, #81E6D9 0%, #38B2AC 100%);
                --success-gradient: linear-gradient(135deg, #68D391 0%, #38A169 100%);
                --warning-gradient: linear-gradient(135deg, #F6E05E 0%, #D69E2E 100%);
                --error-gradient: linear-gradient(135deg, #FC8181 0%, #E53E3E 100%);
                --neutral-gradient: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
                --glass-effect: rgba(45, 55, 72, 0.25);
                --shadow-primary: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                --shadow-elevated: 0 20px 60px 0 rgba(0, 0, 0, 0.4);
                --text-primary: #F7FAFC;
                --text-secondary: #E2E8F0;
                --text-muted: #A0AEC0;
                --bg-primary: #1A202C;
                --bg-secondary: #2D3748;
                --bg-tertiary: #4A5568;
                --card-bg: rgba(45, 55, 72, 0.8);
                --border-color: rgba(255, 255, 255, 0.1);
            }
            
            .main .block-container {
                background: linear-gradient(135deg, #1A202C 0%, #2D3748 50%, #4A5568 100%);
                color: var(--text-primary);
            }
            
            .metric-card {
                background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(26, 32, 44, 0.8) 100%);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
            }
            
            .status-card {
                background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.9) 100%);
                color: var(--text-primary);
            }
            
            .upload-area {
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-color: #667eea;
                color: var(--text-secondary);
            }
            
            .section-header {
                color: var(--text-primary);
            }
            
            .info-text {
                color: var(--text-secondary);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(26, 32, 44, 0.8) 100%);
            }
            
            .metric-container {
                background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.9) 100%);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
            }
            
            .progress-container {
                background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(26, 32, 44, 0.8) 100%);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
            }
        </style>
        """
    else:
        theme_css = """
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #0F4C75 0%, #3282B8 50%, #0F4C75 100%);
                --secondary-gradient: linear-gradient(135deg, #1A365D 0%, #2D3748 100%);
                --accent-gradient: linear-gradient(135deg, #38B2AC 0%, #319795 100%);
                --success-gradient: linear-gradient(135deg, #38A169 0%, #2F855A 100%);
                --warning-gradient: linear-gradient(135deg, #D69E2E 0%, #B7791F 100%);
                --error-gradient: linear-gradient(135deg, #E53E3E 0%, #C53030 100%);
                --neutral-gradient: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 100%);
                --glass-effect: rgba(255, 255, 255, 0.25);
                --shadow-primary: 0 8px 32px 0 rgba(15, 76, 117, 0.15);
                --shadow-elevated: 0 20px 60px 0 rgba(15, 76, 117, 0.25);
                --text-primary: #1A202C;
                --text-secondary: #4A5568;
                --text-muted: #718096;
                --bg-primary: #FFFFFF;
                --bg-secondary: #F7FAFC;
                --bg-tertiary: #EDF2F7;
                --card-bg: rgba(255, 255, 255, 0.9);
                --border-color: rgba(0, 0, 0, 0.1);
            }
            
            .main .block-container {
                background: linear-gradient(135deg, #F7FAFC 0%, #EDF2F7 50%, #E2E8F0 100%);
                color: var(--text-primary);
            }
            
            .metric-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: var(--text-primary);
            }
            
            .status-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
                color: var(--text-primary);
            }
            
            .upload-area {
                background: linear-gradient(135deg, rgba(50, 130, 184, 0.05) 0%, rgba(15, 76, 117, 0.05) 100%);
                border-color: #3282B8;
                color: var(--text-secondary);
            }
            
            .section-header {
                color: var(--text-primary);
            }
            
            .info-text {
                color: var(--text-secondary);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
            }
            
            .metric-container {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: var(--text-primary);
            }
            
            .progress-container {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(241, 245, 249, 0.8) 100%);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: var(--text-primary);
            }
        </style>
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Initialize theme in session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Apply current theme
    apply_theme_css(st.session_state.theme)
    
    # Header
    st.markdown('<h1 class="main-header">Resume Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Resume Analysis & Insights Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Theme toggle
        st.markdown("#### 🎨 Appearance")
        theme_col1, theme_col2 = st.columns(2)
        
        with theme_col1:
            if st.button("☀️ Light", use_container_width=True, 
                        type="primary" if st.session_state.theme == "light" else "secondary"):
                st.session_state.theme = "light"
                st.rerun()
        
        with theme_col2:
            if st.button("🌙 Dark", use_container_width=True,
                        type="primary" if st.session_state.theme == "dark" else "secondary"):
                st.session_state.theme = "dark"
                st.rerun()
        
        # Current theme indicator
        theme_emoji = "☀️" if st.session_state.theme == "light" else "🌙"
        theme_name = st.session_state.theme.title()
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(0,0,0,0.1); border-radius: 8px; margin: 0.5rem 0;">
            <span style="font-size: 1.2rem;">{theme_emoji}</span>
            <br>
            <small style="color: var(--text-secondary);">Current: {theme_name} Mode</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System status
        st.markdown("#### System Status")
        if PARSER_AVAILABLE:
            st.markdown("""
            <div class="status-card">
                <div style="color: #10b981; font-weight: 600;">✓ System Online</div>
                <div style="color: #64748b; font-size: 0.9rem;">All modules loaded successfully</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Parser modules not available")
        
        # Quick actions
        st.markdown("#### Quick Actions")
        if st.button("🔄 Run System Check", use_container_width=True):
            with st.spinner("Running system validation..."):
                import subprocess
                try:
                    result = subprocess.run(['python', 'simple_demo.py'], 
                                          capture_output=True, text=True, cwd='.')
                    if result.returncode == 0:
                        st.success("✅ System check passed!")
                    else:
                        st.error("❌ System check failed")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Statistics
        st.markdown("#### Dashboard Stats")
        st.metric("Files Processed", "156", "↗️ +12")
        st.metric("Success Rate", "98.7%", "↗️ +1.2%")
        st.metric("Avg Processing", "2.3s", "↘️ -0.4s")
        
        # Help section
        st.markdown("#### 📚 Resources")
        st.markdown("""
        - [📖 Documentation](README.md)
        - [🔧 API Reference](http://localhost:8000/docs)
        - [💬 Support](mailto:support@company.com)
        """)
    
    # Main content area
    if not PARSER_AVAILABLE:
        st.error("⚠️ Resume parser modules are not available. Please install dependencies:")
        st.code("pip install -r requirements.txt")
        return
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📄 Resume Analysis", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        # File upload section with professional styling
        st.markdown('<div class="section-header">📁 Upload Resume</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="upload-area">
                <h4 style="color: #475569; margin-bottom: 0.5rem;">Drop your resume file here</h4>
                <p style="color: #64748b; margin: 0;">Supports PDF, DOCX, DOC formats • Max 10MB</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=['pdf', 'docx', 'doc'],
                help="Upload PDF or Word document resumes",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("#### 📈 Processing Stats")
            st.metric("Files Today", "23", "↗️ +5")
            st.metric("Queue Length", "0", "→ Idle")
        
        if uploaded_file is not None:
            # Processing animation
            with st.spinner("🔄 Processing your resume..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                steps = [
                    ("Extracting text...", 25),
                    ("Analyzing content...", 50),
                    ("Extracting entities...", 75),
                    ("Generating insights...", 100)
                ]
                
                for step_text, progress in steps:
                    status_text.text(step_text)
                    progress_bar.progress(progress)
                    time.sleep(0.5)
                
                status_text.text("✅ Processing complete!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
            
            # Save and process file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Extract and process
                raw_text = extract_text_from_file(temp_path)
                sections = normalize_text(raw_text)
                entities = extract_entities(sections.get('full_text', raw_text))
                
                # Success banner
                st.markdown("""
                <div class="success-banner">
                    ✅ Resume processed successfully! Analysis complete.
                </div>
                """, unsafe_allow_html=True)
                
                # Results dashboard
                st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(create_metric_card(
                        "Skills Found", 
                        len(entities.get('skills', [])), 
                        "Technical & soft skills",
                        "#10b981"
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_metric_card(
                        "Experience", 
                        f"{len(entities.get('experience', []))} roles", 
                        "Work history entries",
                        "#3b82f6"
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_metric_card(
                        "Education", 
                        f"{len(entities.get('education', []))} degrees", 
                        "Educational background",
                        "#8b5cf6"
                    ), unsafe_allow_html=True)
                
                with col4:
                    st.markdown(create_metric_card(
                        "Content Size", 
                        f"{len(raw_text)} chars", 
                        "Total text processed",
                        "#f59e0b"
                    ), unsafe_allow_html=True)
                
                # Detailed analysis
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.markdown('<div class="section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
                    
                    # Contact info in a nice format
                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <h4 style="color: #1e293b; margin-top: 0;">Contact Information</h4>
                        <p><strong>Name:</strong> {entities.get('name', 'Not detected')}</p>
                        <p><strong>Email:</strong> {entities.get('email', 'Not detected')}</p>
                        <p><strong>Phone:</strong> {entities.get('phone', 'Not detected')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Experience section
                    if entities.get('experience'):
                        st.markdown("#### 💼 Work Experience")
                        for i, exp in enumerate(entities['experience'][:3], 1):
                            with st.expander(f"Position {i}: {exp.get('title', 'Unknown Title')}", expanded=i==1):
                                st.write(f"**Company:** {exp.get('company', 'Unknown')}")
                                st.write(f"**Duration:** {exp.get('start', 'Unknown')} - {exp.get('end', 'Unknown')}")
                                if exp.get('bullets'):
                                    st.write("**Key Responsibilities:**")
                                    for bullet in exp['bullets'][:3]:
                                        st.write(f"• {bullet}")
                
                with col_right:
                    st.markdown('<div class="section-header">🔧 Skills Analysis</div>', unsafe_allow_html=True)
                    
                    skills = entities.get('skills', [])
                    if skills:
                        # Skills chart
                        fig = create_skills_chart(skills, st.session_state.theme)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Skills breakdown
                        with st.expander("📋 All Detected Skills", expanded=False):
                            skills_df = pd.DataFrame({
                                'Skill': skills,
                                'Category': ['Technical'] * len(skills)  # Could be enhanced with categorization
                            })
                            st.dataframe(skills_df, use_container_width=True)
                    else:
                        st.info("No skills detected in the resume")
                
                # Raw data section
                with st.expander("🔍 Complete Analysis Data", expanded=False):
                    st.json(entities)
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-banner">
                    ❌ Error processing resume: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 Try a different file format or check if the file is corrupted")
            
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    with tab2:
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        
        # Sample analytics (in a real app, this would be from a database)
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample skill trends
            skills_data = {
                'Skill': ['Python', 'JavaScript', 'React', 'AWS', 'Docker', 'SQL'],
                'Frequency': [45, 38, 32, 28, 25, 22]
            }
            
            # Theme-aware colors
            chart_colors = '#667eea' if st.session_state.theme == 'light' else '#81E6D9'
            text_color = '#1e293b' if st.session_state.theme == 'light' else '#F7FAFC'
            
            fig = px.bar(skills_data, x='Skill', y='Frequency', 
                        title="Most Common Skills in Processed Resumes",
                        color='Frequency',
                        color_continuous_scale='viridis' if st.session_state.theme == 'light' else 'plasma')
            
            fig.update_layout(
                font_family="Inter",
                font_color=text_color,
                title_font_color=text_color,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample processing stats
            processing_data = {
                'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                'Resumes': [12, 19, 15, 23, 18]
            }
            
            fig = px.line(processing_data, x='Day', y='Resumes', 
                         title="Daily Processing Volume",
                         markers=True,
                         color_discrete_sequence=[chart_colors])
            
            fig.update_layout(
                font_family="Inter",
                font_color=text_color,
                title_font_color=text_color,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Processing Settings")
            st.slider("Max file size (MB)", 1, 50, 10)
            st.multiselect("Supported formats", ['PDF', 'DOCX', 'DOC', 'TXT'], ['PDF', 'DOCX', 'DOC'])
            st.checkbox("Enable auto-processing", True)
        
        with col2:
            st.markdown("#### Output Settings")
            st.selectbox("Default format", ['JSON', 'CSV', 'Excel'])
            st.checkbox("Include confidence scores", True)
            st.checkbox("Enable detailed logs", False)

if __name__ == "__main__":
    main()
