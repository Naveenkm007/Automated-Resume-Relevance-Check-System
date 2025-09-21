#!/usr/bin/env python3
"""
🚀 ULTIMATE PROFESSIONAL AI DASHBOARD - Vercel Edition
=====================================================
Enterprise-grade AI-powered resume analytics with stunning visuals.
Built for Innomatics Research Labs.
"""

from flask import Flask, request, jsonify, render_template_string
import re
import json
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta

app = Flask(__name__)

# Extended skills database matching your original
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
ALL_SKILLS = []
for category in SKILLS_DATABASE.values():
    ALL_SKILLS.extend(category)

def extract_skills(text: str) -> List[str]:
    """Extract skills from resume text."""
    text_lower = text.lower()
    found = [skill for skill in ALL_SKILLS if skill in text_lower]
    return sorted(found)

def extract_contact(text: str) -> Dict[str, str]:
    """Extract contact information."""
    contact = {}
    
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        contact['email'] = email_match.group()
    
    # Phone
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        contact['phone'] = phone_match.group()
    
    # Name (simple heuristic)
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if len(line.split()) == 2 and line.replace(' ', '').isalpha() and len(line) < 50:
            if '@' not in line:
                contact['name'] = line
                break
    
    return contact

def analyze_resume(text: str, required_skills: List[str]) -> Dict[str, Any]:
    """Analyze resume against job requirements."""
    
    # Extract information
    found_skills = extract_skills(text)
    contact_info = extract_contact(text)
    
    # Skill matching
    required_lower = [skill.lower() for skill in required_skills]
    found_lower = [skill.lower() for skill in found_skills]
    matches = [skill for skill in required_lower if skill in found_lower]
    
    # Scoring
    skill_score = (len(matches) / max(len(required_skills), 1)) * 100
    
    # Experience detection
    exp_keywords = ['experience', 'worked', 'developed', 'managed', 'led']
    exp_score = min(sum(1 for kw in exp_keywords if kw in text.lower()) * 20, 100)
    
    # Overall score
    overall_score = (skill_score * 0.7) + (exp_score * 0.3)
    
    return {
        'contact': contact_info,
        'skills_found': found_skills,
        'skills_matched': matches,
        'skills_missing': [s for s in required_skills if s.lower() not in found_lower],
        'scores': {
            'skills': skill_score,
            'experience': exp_score,
            'overall': overall_score
        },
        'verdict': 'Excellent' if overall_score >= 80 else 'Good' if overall_score >= 60 else 'Fair'
    }

# EXACT AI Insights Dashboard HTML Template - Matching Screenshot
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI INSIGHTS DASHBOARD | Innomatics Labs</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f0f2f7 0%, #e6e9f0 25%, #dde1ea 50%, #d4d9e3 75%, #cbd1dc 100%);
            color: #2d3748;
            min-height: 100vh;
            margin: 0;
            overflow-x: hidden;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .header { text-align: center; margin-bottom: 2rem; }
        .main-header {
            font-family: 'Orbitron', monospace;
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(45deg, #00d4ff, #091a7a, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
            margin-bottom: 1rem;
        }
        
        .ai-badge {
            display: inline-block;
            background: linear-gradient(45deg, #ff006e, #8338ec, #3a86ff);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.8rem;
            font-weight: bold;
            color: white;
            margin: 0.2rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 0, 110, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 0, 110, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 0, 110, 0); }
        }
        
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 300px;
            height: 100vh;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            padding: 2rem;
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-content { margin-left: 320px; padding: 1rem; }
        
        .form-group { margin-bottom: 1.5rem; }
        .form-group label { 
            display: block; 
            margin-bottom: 0.5rem; 
            font-weight: 600; 
            color: #00d4ff;
            font-size: 0.9rem;
        }
        .form-group input, .form-group textarea { 
            width: 100%; 
            padding: 12px; 
            border: 2px solid rgba(255, 255, 255, 0.1); 
            border-radius: 8px; 
            font-size: 14px; 
            background: rgba(255, 255, 255, 0.05);
            color: #e0e6ed;
            transition: border-color 0.3s; 
        }
        .form-group input:focus, .form-group textarea:focus { 
            outline: none; 
            border-color: #00d4ff; 
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        
        .btn { 
            background: linear-gradient(135deg, #00d4ff 0%, #091a7a 100%); 
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 8px; 
            font-size: 16px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: transform 0.2s; 
            width: 100%;
        }
        .btn:hover { transform: translateY(-2px); }
        
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 1.5rem; 
            margin-bottom: 2rem; 
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
            text-align: center;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        }
        
        .metric-card h3 { 
            color: #00d4ff; 
            margin-bottom: 0.5rem; 
            font-size: 1.1rem;
        }
        
        .metric-value { 
            font-size: 2.5rem; 
            font-weight: bold; 
            margin: 0.5rem 0;
        }
        
        .score-excellent { color: #00ff88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        .score-good { color: #00d4ff; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5); }
        .score-average { color: #ffaa00; text-shadow: 0 0 10px rgba(255, 170, 0, 0.5); }
        .score-poor { color: #ff4757; text-shadow: 0 0 10px rgba(255, 71, 87, 0.5); }
        
        .charts-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 2rem; 
            margin: 2rem 0; 
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chart-title {
            color: #00d4ff;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .skills-container { margin-top: 2rem; }
        .skills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
        .skill { 
            background: linear-gradient(45deg, #00d4ff, #091a7a); 
            color: white; 
            padding: 0.5rem 1rem; 
            border-radius: 20px; 
            font-size: 0.9rem; 
            font-weight: 500;
        }
        .missing { background: linear-gradient(45deg, #ff4757, #c44569); }
        
        .ai-insight {
            background: linear-gradient(135deg, rgba(131, 56, 236, 0.1), rgba(58, 134, 255, 0.1));
            border-left: 4px solid #8338ec;
            padding: 1rem;
            border-radius: 0 10px 10px 0;
            margin: 1rem 0;
        }
        
        .footer { 
            text-align: center; 
            margin-top: 3rem; 
            padding: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        @media (max-width: 768px) {
            .sidebar { position: relative; width: 100%; height: auto; }
            .main-content { margin-left: 0; }
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Resume Relevance Check System</h1>
            <p>AI-Powered Resume Evaluation | Built for Innomatics Research Labs</p>
        </div>
        
        <div class="card">
            <form id="resumeForm">
                <div class="form-group">
                    <label for="resumeText">📄 Resume Text</label>
                    <textarea id="resumeText" rows="10" placeholder="Paste your resume text here..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="jobTitle">🎯 Job Title</label>
                    <input type="text" id="jobTitle" value="Python Developer" required>
                </div>
                
                <div class="form-group">
                    <label for="requiredSkills">🔧 Required Skills (comma-separated)</label>
                    <input type="text" id="requiredSkills" value="python, sql, git, api, django" required>
                </div>
                
                <button type="submit" class="btn">🔍 Analyze Resume</button>
            </form>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <div class="card">
                <h2>📊 Analysis Results</h2>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Built for Innomatics Research Labs</strong> | Scale • Consistency • Automation</p>
        </div>
    </div>

    <script>
        document.getElementById('resumeForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resumeText = document.getElementById('resumeText').value;
            const jobTitle = document.getElementById('jobTitle').value;
            const requiredSkills = document.getElementById('requiredSkills').value.split(',').map(s => s.trim());
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ resume_text: resumeText, job_title: jobTitle, required_skills: requiredSkills })
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                alert('Error analyzing resume: ' + error.message);
            }
        });
        
        function displayResults(result) {
            const verdictClass = result.verdict.toLowerCase();
            const html = `
                <div class="metric">
                    <h3>📈 Overall Score</h3>
                    <div class="score ${verdictClass}">${result.scores.overall.toFixed(1)}%</div>
                    <p>Verdict: <strong>${result.verdict} Match</strong></p>
                </div>
                
                <div class="metric">
                    <h3>🎯 Skills Analysis</h3>
                    <p>Skills Score: <strong>${result.scores.skills.toFixed(1)}%</strong></p>
                    <p>Experience Score: <strong>${result.scores.experience.toFixed(1)}%</strong></p>
                </div>
                
                ${result.contact.name ? `
                <div class="metric">
                    <h3>👤 Contact Information</h3>
                    <p><strong>Name:</strong> ${result.contact.name}</p>
                    ${result.contact.email ? `<p><strong>Email:</strong> ${result.contact.email}</p>` : ''}
                    ${result.contact.phone ? `<p><strong>Phone:</strong> ${result.contact.phone}</p>` : ''}
                </div>
                ` : ''}
                
                <div class="metric">
                    <h3>✅ Matched Skills</h3>
                    <div class="skills">
                        ${result.skills_matched.map(skill => `<span class="skill">${skill}</span>`).join('')}
                    </div>
                </div>
                
                ${result.skills_missing.length > 0 ? `
                <div class="metric">
                    <h3>⚠️ Missing Skills</h3>
                    <div class="skills">
                        ${result.skills_missing.map(skill => `<span class="skill missing">${skill}</span>`).join('')}
                    </div>
                </div>
                ` : ''}
                
                <div class="metric">
                    <h3>🔍 All Skills Found</h3>
                    <div class="skills">
                        ${result.skills_found.map(skill => `<span class="skill">${skill}</span>`).join('')}
                    </div>
                </div>
            `;
            
            document.getElementById('resultContent').innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Render the main application page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for resume analysis."""
    try:
        data = request.get_json()
        
        resume_text = data.get('resume_text', '')
        job_title = data.get('job_title', '')
        required_skills = data.get('required_skills', [])
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
        
        # Perform analysis
        result = analyze_resume(resume_text, required_skills)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Resume Relevance Check System'})

if __name__ == '__main__':
    app.run(debug=True)
