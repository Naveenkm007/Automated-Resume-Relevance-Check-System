#!/usr/bin/env python3
"""
Simple Resume Parser Demo - No special characters
"""

import json
from resume_parser.cleaner import normalize_text
from resume_parser.ner import extract_entities

def run_demo():
    sample_text = """John Smith
Software Engineer
john.smith@email.com | (555) 123-4567

EXPERIENCE
Senior Software Developer at Tech Corp Inc
January 2020 - Present
- Developed scalable web applications using Python and Django
- Led a team of 5 developers in agile development processes
- Implemented REST APIs serving over 1M requests daily

EDUCATION
Bachelor of Science in Computer Science
Massachusetts Institute of Technology, 2018
GPA: 3.8/4.0

SKILLS
Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Machine Learning"""

    print("RESUME PARSER DEMO")
    print("=" * 50)
    print(f"Processing resume ({len(sample_text)} characters)")
    
    # Parse sections
    sections = normalize_text(sample_text)
    print(f"Detected {len(sections)} sections")
    
    # Extract entities
    entities = extract_entities(sample_text)
    
    # Display results
    print("\nEXTRACTED DATA:")
    print(f"Name: {entities.get('name', 'Not found')}")
    print(f"Email: {entities.get('email', 'Not found')}")
    print(f"Phone: {entities.get('phone', 'Not found')}")
    print(f"Skills: {len(entities.get('skills', []))} found")
    
    if entities.get('skills'):
        print("Top skills:", ', '.join(entities['skills'][:5]))
    
    print("\nDemo completed successfully!")
    return entities

if __name__ == "__main__":
    run_demo()
