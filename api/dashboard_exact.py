#!/usr/bin/env python3
"""
🤖 AI INSIGHTS DASHBOARD - Exact Screenshot Match for Vercel
===========================================================
Recreating the exact beautiful UI from the screenshot.
Built for Innomatics Research Labs.
"""

from flask import Flask, request, jsonify, render_template_string
import json
import random

app = Flask(__name__)

# Exact HTML Template Matching Screenshot
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI INSIGHTS DASHBOARD</title>
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
        }
        
        .layout {
            display: flex;
            min-height: 100vh;
        }
        
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-right: 1px solid rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
        }
        
        .main-content {
            flex: 1;
            padding: 1.5rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
        }
        
        .theme-toggle {
            position: absolute;
            top: 0;
            left: 0;
            background: rgba(255, 255, 255, 0.9);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .ai-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 1rem;
            vertical-align: middle;
        }
        
        .subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        
        .form-section {
            margin-bottom: 1.5rem;
        }
        
        .form-section h3 {
            font-size: 0.9rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 0.8rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            font-size: 0.8rem;
            font-weight: 500;
            color: #4b5563;
            margin-bottom: 0.3rem;
        }
        
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 0.7rem;
            border: 1.5px solid #e5e7eb;
            border-radius: 6px;
            font-size: 0.85rem;
            background: white;
            transition: border-color 0.2s;
        }
        
        .form-group input:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .collapsible {
            margin-bottom: 1rem;
        }
        
        .collapsible-header {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.8rem;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            color: #374151;
        }
        
        .collapsible-content {
            border: 1px solid #e5e7eb;
            border-top: none;
            border-radius: 0 0 6px 6px;
            padding: 1rem;
            background: white;
        }
        
        .checkbox-group {
            margin: 0.5rem 0;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: auto;
            margin-right: 0.5rem;
        }
        
        .checkbox-group label {
            display: inline;
            font-size: 0.8rem;
            color: #4b5563;
        }
        
        .slider-group {
            margin: 1rem 0;
        }
        
        .slider-group label {
            font-size: 0.8rem;
            color: #4b5563;
        }
        
        .demo-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .demo-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        .demo-title::before {
            content: "🎮";
            margin-right: 0.5rem;
        }
        
        .upload-hint {
            background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
            border: 1px solid #93c5fd;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            color: #1e40af;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }
        
        .chart-section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 1rem;
        }
        
        .chart-placeholder {
            width: 100%;
            height: 300px;
            background: #f8fafc;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .layout {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
            }
            
            .charts-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="layout">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="form-section">
                <h3>Job Title</h3>
                <div class="form-group">
                    <input type="text" id="jobTitle" placeholder="Senior Python Developer" value="Senior Python Developer">
                </div>
            </div>
            
            <div class="form-section">
                <h3>Must-Have Skills (comma-separated)</h3>
                <div class="form-group">
                    <textarea id="mustHaveSkills" rows="3" placeholder="python, django, postgresql, rest api">python, django, postgresql, rest api</textarea>
                </div>
            </div>
            
            <div class="form-section">
                <h3>Good-to-Have Skills (comma-separated)</h3>
                <div class="form-group">
                    <textarea id="goodToHaveSkills" rows="3" placeholder="aws, docker, react">aws, docker, react</textarea>
                </div>
            </div>
            
            <div class="collapsible">
                <div class="collapsible-header">🤖 AI Settings</div>
                <div class="collapsible-content">
                    <div class="slider-group">
                        <label>Analysis Depth</label>
                        <input type="range" min="1" max="4" value="2" style="width: 100%; margin-top: 0.5rem;">
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">Standard</div>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="showPredictions" checked>
                        <label for="showPredictions">Show AI Predictions</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="showRecommendations" checked>
                        <label for="showRecommendations">Show Recommendations</label>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Header -->
            <div class="header">
                <button class="theme-toggle">🌓 Toggle Theme</button>
                <div class="main-title">🤖 AI INSIGHTS DASHBOARD</div>
                <div class="ai-badge">AI POWERED</div>
                <div class="subtitle">Next-Generation Resume Analytics - Powered by Machine Learning & Advanced Visualizations</div>
            </div>
            
            <!-- Demo Section -->
            <div class="demo-section">
                <div class="demo-title">Demo Mode</div>
                
                <div class="upload-hint">
                    👆 Upload a resume file to see the AI analysis in action!
                </div>
                
                <!-- Charts -->
                <div class="charts-container">
                    <div class="chart-section">
                        <div class="chart-title">Sample Skill Analysis</div>
                        <div id="skillChart" style="width: 100%; height: 300px;"></div>
                    </div>
                    
                    <div class="chart-section">
                        <div class="chart-title">Sample Score Evolution</div>
                        <div id="evolutionChart" style="width: 100%; height: 300px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Create Skill Radar Chart
        function createSkillChart() {
            const data = [{
                type: 'scatterpolar',
                r: [80, 90, 70, 95, 60],
                theta: ['Python', 'PostgreSQL', 'Django', 'React', 'AWS'],
                fill: 'toself',
                name: 'Candidate Skills',
                fillcolor: 'rgba(102, 126, 234, 0.3)',
                line: { color: 'rgba(102, 126, 234, 1)' }
            }, {
                type: 'scatterpolar',
                r: [95, 85, 90, 70, 80],
                theta: ['Python', 'PostgreSQL', 'Django', 'React', 'AWS'],
                fill: 'toself',
                name: 'Job Requirements',
                fillcolor: 'rgba(118, 75, 162, 0.3)',
                line: { color: 'rgba(118, 75, 162, 1)' }
            }];
            
            const layout = {
                polar: {
                    radialaxis: {
                        visible: true,
                        range: [0, 100],
                        gridcolor: 'rgba(0,0,0,0.1)'
                    }
                },
                showlegend: true,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter, sans-serif', size: 11 },
                margin: { t: 40, b: 40, l: 40, r: 40 }
            };
            
            Plotly.newPlot('skillChart', data, layout, {displayModeBar: false, responsive: true});
        }
        
        // Create Score Evolution Chart
        function createEvolutionChart() {
            const dates = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05'];
            const candidateScores = [45, 58, 67, 74, 82];
            const jobRequirements = [80, 80, 80, 80, 80];
            
            const data = [{
                x: dates,
                y: candidateScores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Candidate Skills',
                line: { color: '#667eea', width: 3 },
                marker: { size: 6, color: '#667eea' }
            }, {
                x: dates,
                y: jobRequirements,
                type: 'scatter',
                mode: 'lines',
                name: 'Job Requirements',
                line: { color: '#764ba2', width: 2, dash: 'dash' }
            }];
            
            const layout = {
                showlegend: true,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { family: 'Inter, sans-serif', size: 11 },
                xaxis: { 
                    gridcolor: 'rgba(0,0,0,0.1)',
                    title: ''
                },
                yaxis: { 
                    gridcolor: 'rgba(0,0,0,0.1)',
                    title: 'Score',
                    range: [0, 100]
                },
                margin: { t: 40, b: 40, l: 60, r: 40 }
            };
            
            Plotly.newPlot('evolutionChart', data, layout, {displayModeBar: false, responsive: true});
        }
        
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            createSkillChart();
            createEvolutionChart();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Render the exact AI Insights Dashboard matching the screenshot."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle resume analysis requests."""
    try:
        data = request.get_json()
        
        # Mock analysis for demonstration
        result = {
            'overall_score': random.uniform(75, 95),
            'skill_match': random.uniform(60, 90),
            'experience': random.randint(3, 8),
            'ai_confidence': random.uniform(80, 95),
            'verdict': 'Excellent Match'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'service': '🤖 AI Insights Dashboard',
        'version': '2.0',
        'features': ['AI Analysis', 'Interactive Charts', 'Professional UI']
    })

if __name__ == '__main__':
    app.run(debug=True)
