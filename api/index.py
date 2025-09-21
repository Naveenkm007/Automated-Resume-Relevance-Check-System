#!/usr/bin/env python3
"""
Ultra-Minimal Resume Relevance Check - Flask Version for Vercel
==============================================================
Lightweight Flask app optimized for Vercel's 250MB limit.
Built for Innomatics Research Labs.
"""

from flask import Flask, request, jsonify, render_template_string
import re
from typing import Dict, List, Any

app = Flask(__name__)

# Built-in skills database
SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'html', 'css', 'react', 'angular', 'vue',
    'node', 'express', 'django', 'flask', 'spring', 'sql', 'mysql', 'postgresql', 'mongodb',
    'aws', 'azure', 'docker', 'kubernetes', 'git', 'api', 'rest', 'microservices', 'devops'
]

def extract_skills(text: str) -> List[str]:
    """Extract skills from resume text."""
    text_lower = text.lower()
    found = [skill for skill in SKILLS if skill in text_lower]
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

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Relevance Check | Innomatics Labs</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 2rem; }
        .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .card { background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 1.5rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: 600; color: #555; }
        .form-group input, .form-group textarea { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }
        .form-group input:focus, .form-group textarea:focus { outline: none; border-color: #667eea; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .results { margin-top: 2rem; }
        .metric { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea; }
        .metric h3 { color: #667eea; margin-bottom: 0.5rem; }
        .score { font-size: 2rem; font-weight: bold; }
        .excellent { color: #28a745; }
        .good { color: #17a2b8; }
        .fair { color: #ffc107; }
        .skills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
        .skill { background: #667eea; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
        .missing { background: #dc3545; }
        .footer { text-align: center; color: white; margin-top: 2rem; opacity: 0.8; }
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
