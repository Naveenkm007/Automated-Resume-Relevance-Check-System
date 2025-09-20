#!/usr/bin/env python3
"""
Resume Parser Demo Script

This script demonstrates the resume parser functionality using sample text.
Since we don't have actual PDF/DOCX files, we'll use text processing directly.
"""

import json
from resume_parser.cleaner import normalize_text
from resume_parser.ner import extract_entities
from resume_parser.utils import extract_email, extract_phone, extract_skills_from_text

def demo_parsing():
    """Demonstrate the resume parsing pipeline."""
    
    # Read sample resume text
    with open('sample_resume.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print("=" * 60)
    print("🤖 RESUME PARSER DEMO")
    print("=" * 60)
    print(f"📄 Processing sample resume ({len(sample_text)} characters)")
    print()
    
    # Step 1: Text Normalization
    print("Step 1: Text Normalization & Section Detection")
    print("-" * 50)
    sections = normalize_text(sample_text)
    
    print(f"✅ Detected {len(sections)} sections:")
    for section_name in sections.keys():
        if section_name != 'full_text':
            content_preview = sections[section_name][:100] + "..." if len(sections[section_name]) > 100 else sections[section_name]
            print(f"   • {section_name.upper()}: {content_preview}")
    print()
    
    # Step 2: Basic Utilities Demo
    print("Step 2: Basic Information Extraction")
    print("-" * 50)
    email = extract_email(sample_text)
    phone = extract_phone(sample_text)
    skills = extract_skills_from_text(sample_text)
    
    print(f"📧 Email: {email}")
    print(f"📞 Phone: {phone}")
    print(f"🛠️  Skills Found: {len(skills)}")
    print(f"   {', '.join(skills[:10])}{'...' if len(skills) > 10 else ''}")
    print()
    
    # Step 3: Full Entity Extraction
    print("Step 3: Complete Entity Extraction")
    print("-" * 50)
    entities = extract_entities(sections['full_text'])
    
    print("📊 EXTRACTED ENTITIES:")
    print(f"   Name: {entities.get('name', 'Not detected')}")
    print(f"   Email: {entities.get('email', 'Not detected')}")
    print(f"   Phone: {entities.get('phone', 'Not detected')}")
    print(f"   Skills: {len(entities.get('skills', []))} found")
    print(f"   Education: {len(entities.get('education', []))} entries")
    print(f"   Experience: {len(entities.get('experience', []))} entries")
    print(f"   Projects: {len(entities.get('projects', []))} entries")
    print()
    
    # Step 4: Detailed Results
    print("Step 4: Detailed Results")
    print("-" * 50)
    
    # Skills breakdown
    if entities.get('skills'):
        print(f"🔧 SKILLS ({len(entities['skills'])}):")
        for i, skill in enumerate(entities['skills'][:15], 1):  # Show first 15
            print(f"   {i:2d}. {skill}")
        if len(entities['skills']) > 15:
            print(f"   ... and {len(entities['skills']) - 15} more")
        print()
    
    # Experience details
    if entities.get('experience'):
        print(f"💼 EXPERIENCE ({len(entities['experience'])} entries):")
        for i, exp in enumerate(entities['experience'], 1):
            print(f"   {i}. {exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')}")
            if exp.get('start') or exp.get('end'):
                print(f"      Duration: {exp.get('start', '?')} - {exp.get('end', '?')}")
            if exp.get('bullets'):
                print(f"      Responsibilities: {len(exp['bullets'])} bullet points")
        print()
    
    # Education details  
    if entities.get('education'):
        print(f"🎓 EDUCATION ({len(entities['education'])} entries):")
        for i, edu in enumerate(entities['education'], 1):
            print(f"   {i}. {edu.get('degree', 'Unknown')} from {edu.get('institution', 'Unknown')}")
            if edu.get('year'):
                print(f"      Year: {edu['year']}")
        print()
    
    # Projects
    if entities.get('projects'):
        print(f"🚀 PROJECTS ({len(entities['projects'])} found):")
        for i, proj in enumerate(entities['projects'], 1):
            print(f"   {i}. {proj.get('title', 'Untitled')}")
            if proj.get('desc'):
                desc_preview = proj['desc'][:100] + "..." if len(proj['desc']) > 100 else proj['desc']
                print(f"      {desc_preview}")
        print()
    
    # Step 5: JSON Output
    print("Step 5: Complete JSON Output")
    print("-" * 50)
    print("💾 Full structured output:")
    print(json.dumps(entities, indent=2, ensure_ascii=False))
    
    return entities

if __name__ == '__main__':
    try:
        result = demo_parsing()
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
