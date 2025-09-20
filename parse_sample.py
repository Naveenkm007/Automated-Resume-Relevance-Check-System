#!/usr/bin/env python3
"""
Resume Parser CLI Script

This script provides a command-line interface for the resume parser.
It takes a resume file (PDF or DOCX) as input and outputs structured
JSON data with extracted information.

Usage:
    python parse_sample.py <file_path>
    python parse_sample.py samples/resume1.pdf
    python parse_sample.py resumes/john_doe.docx

The script will:
1. Extract text from the resume file
2. Clean and normalize the text
3. Extract structured entities using NLP
4. Output the results as formatted JSON
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Import the resume parser modules
try:
    from resume_parser.extract import extract_text_from_file
    from resume_parser.cleaner import normalize_text
    from resume_parser.ner import extract_entities
except ImportError as e:
    print(f"Error importing resume_parser modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def parse_resume(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Parse a resume file and return structured data.
    
    Args:
        file_path (str): Path to the resume file (PDF or DOCX)
        verbose (bool): Whether to print verbose output
        
    Returns:
        Dict[str, Any]: Structured resume data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
        Exception: For other parsing errors
    """
    if verbose:
        print(f"Processing file: {file_path}")
    
    # Step 1: Extract text from file
    if verbose:
        print("Step 1: Extracting text from file...")
    
    try:
        raw_text = extract_text_from_file(file_path)
        if verbose:
            print(f"Extracted {len(raw_text)} characters of text")
    except Exception as e:
        raise Exception(f"Failed to extract text from {file_path}: {e}")
    
    # Step 2: Clean and normalize text
    if verbose:
        print("Step 2: Cleaning and normalizing text...")
    
    try:
        sections = normalize_text(raw_text)
        cleaned_text = sections.get('full_text', raw_text)
        if verbose:
            print(f"Identified {len(sections)} sections: {list(sections.keys())}")
    except Exception as e:
        print(f"Warning: Text cleaning failed: {e}")
        cleaned_text = raw_text
        sections = {'full_text': raw_text}
    
    # Step 3: Extract structured entities
    if verbose:
        print("Step 3: Extracting structured entities...")
    
    try:
        entities = extract_entities(cleaned_text)
        if verbose:
            print(f"Extracted entities: name, email, phone, {len(entities.get('skills', []))} skills, "
                  f"{len(entities.get('experience', []))} experience entries, "
                  f"{len(entities.get('education', []))} education entries")
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
        entities = {
            'name': None,
            'email': None,
            'phone': None,
            'skills': [],
            'education': [],
            'experience': [],
            'projects': []
        }
    
    # Combine results
    result = {
        'file_path': file_path,
        'processing_info': {
            'raw_text_length': len(raw_text),
            'sections_found': list(sections.keys()),
            'entities_extracted': True if entities else False
        },
        'parsed_data': entities
    }
    
    return result


def format_output(data: Dict[str, Any], format_type: str = 'json') -> str:
    """
    Format the parsed data for output.
    
    Args:
        data (Dict[str, Any]): Parsed resume data
        format_type (str): Output format ('json' or 'summary')
        
    Returns:
        str: Formatted output string
    """
    if format_type == 'json':
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    elif format_type == 'summary':
        parsed = data.get('parsed_data', {})
        
        summary = []
        summary.append("=== RESUME PARSING SUMMARY ===")
        summary.append(f"File: {data.get('file_path', 'Unknown')}")
        summary.append("")
        
        # Personal Information
        summary.append("PERSONAL INFORMATION:")
        summary.append(f"  Name: {parsed.get('name', 'Not found')}")
        summary.append(f"  Email: {parsed.get('email', 'Not found')}")
        summary.append(f"  Phone: {parsed.get('phone', 'Not found')}")
        summary.append("")
        
        # Skills
        skills = parsed.get('skills', [])
        summary.append(f"SKILLS ({len(skills)} found):")
        if skills:
            for skill in skills[:10]:  # Show first 10 skills
                summary.append(f"  • {skill}")
            if len(skills) > 10:
                summary.append(f"  ... and {len(skills) - 10} more")
        else:
            summary.append("  None found")
        summary.append("")
        
        # Experience
        experience = parsed.get('experience', [])
        summary.append(f"EXPERIENCE ({len(experience)} entries):")
        for i, exp in enumerate(experience[:3], 1):  # Show first 3 entries
            summary.append(f"  {i}. {exp.get('title', 'Unknown Title')} at {exp.get('company', 'Unknown Company')}")
            if exp.get('start') or exp.get('end'):
                summary.append(f"     Duration: {exp.get('start', '?')} - {exp.get('end', '?')}")
        if len(experience) > 3:
            summary.append(f"  ... and {len(experience) - 3} more entries")
        summary.append("")
        
        # Education
        education = parsed.get('education', [])
        summary.append(f"EDUCATION ({len(education)} entries):")
        for i, edu in enumerate(education, 1):
            summary.append(f"  {i}. {edu.get('degree', 'Unknown Degree')} from {edu.get('institution', 'Unknown Institution')}")
            if edu.get('year'):
                summary.append(f"     Year: {edu.get('year')}")
        summary.append("")
        
        # Projects
        projects = parsed.get('projects', [])
        summary.append(f"PROJECTS ({len(projects)} found):")
        for i, proj in enumerate(projects[:3], 1):  # Show first 3 projects
            summary.append(f"  {i}. {proj.get('title', 'Untitled Project')}")
        if len(projects) > 3:
            summary.append(f"  ... and {len(projects) - 3} more projects")
        
        return '\n'.join(summary)
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def main():
    """Main function for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Parse resume files and extract structured information",
        epilog="Example: python parse_sample.py samples/resume1.pdf"
    )
    
    parser.add_argument(
        'file_path',
        help='Path to the resume file (PDF or DOCX)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'summary'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path (if not specified, prints to stdout)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)
    
    if file_path.suffix.lower() not in ['.pdf', '.docx', '.doc']:
        print(f"Error: Unsupported file format: {file_path.suffix}", file=sys.stderr)
        print("Supported formats: .pdf, .docx, .doc", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse the resume
        result = parse_resume(str(file_path), verbose=args.verbose)
        
        # Format the output
        formatted_output = format_output(result, args.format)
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            if args.verbose:
                print(f"Output written to: {args.output}")
        else:
            print(formatted_output)
    
    except Exception as e:
        print(f"Error parsing resume: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
