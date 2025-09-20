#!/usr/bin/env python3
"""
Semantic Matching and LLM Feedback Demo

This script demonstrates the semantic matching system including:
- Text embedding generation (local vs OpenAI)
- Semantic similarity computation between resume and JD
- Combined scoring (hard + semantic)
- LLM-powered feedback generation
- Vector storage and similarity search

Usage:
    python examples/semantic_demo.py
    
Environment variables:
    USE_OPENAI_EMBEDDINGS=true/false (default: false - uses local models)
    OPENAI_API_KEY=your_key_here (required for OpenAI embeddings and feedback)
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from semantic.embeddings import get_embedding, get_embedding_info, compute_similarity
from semantic.similarity import compute_semantic_score, find_semantic_matches
from semantic.feedback import generate_feedback
from semantic.combined_score import compute_combined_score
from scoring.hard_match import compute_keyword_score


def create_sample_data():
    """Create sample resume and job description data for demonstration."""
    
    sample_jd = {
        'title': 'Senior Python Developer',
        'must_have_skills': ['python', 'django', 'postgresql', 'git', 'rest api'],
        'good_to_have_skills': ['react', 'aws', 'docker', 'redis', 'machine learning'],
        'education_requirements': {
            'level': 'bachelor',
            'field': 'computer science'
        },
        'certifications_required': [],
        'full_text': """
        Senior Python Developer Position
        
        We are looking for an experienced Python developer to join our growing team.
        You will be responsible for developing scalable web applications using Django,
        working with PostgreSQL databases, and building REST APIs.
        
        Required Skills:
        - 5+ years of Python development experience
        - Strong proficiency in Django web framework
        - Experience with PostgreSQL database design and optimization
        - Git version control and collaborative development
        - REST API development and integration
        
        Preferred Skills:
        - Frontend development with React.js
        - AWS cloud services (EC2, S3, RDS)
        - Containerization with Docker
        - Redis caching and session management
        - Machine learning and data analysis experience
        
        Education: Bachelor's degree in Computer Science or related field
        
        You will work on challenging projects involving e-commerce platforms,
        data processing pipelines, and user-facing web applications. Strong
        problem-solving skills and attention to detail are essential.
        """
    }
    
    strong_resume = {
        'name': 'Sarah Chen',
        'email': 'sarah.chen@email.com',
        'skills': [
            'python', 'django', 'postgresql', 'git', 'rest api', 'react',
            'aws', 'docker', 'javascript', 'html', 'css', 'redis'
        ],
        'education': [
            {
                'degree': 'Bachelor of Science in Computer Science',
                'institution': 'Stanford University',
                'year': 2018,
                'stream': 'Computer Science'
            }
        ],
        'certifications': [],
        'experience': [
            {
                'title': 'Python Developer',
                'company': 'TechStart',
                'start': '2019-01',
                'end': 'Present',
                'bullets': [
                    'Developed e-commerce platform using Django and PostgreSQL',
                    'Built REST APIs serving 100k+ daily requests',
                    'Implemented caching with Redis improving response time by 40%',
                    'Deployed applications on AWS using Docker containers',
                    'Collaborated with frontend team using React.js'
                ]
            }
        ],
        'projects': [
            {
                'title': 'ML-Powered Recommendation Engine',
                'desc': 'Built machine learning recommendation system using Python and scikit-learn'
            }
        ],
        'full_text': """
        Sarah Chen
        Python Developer
        sarah.chen@email.com
        
        EXPERIENCE
        Python Developer at TechStart (2019-Present)
        • Developed scalable e-commerce platform using Django web framework and PostgreSQL
        • Designed and implemented REST APIs handling 100,000+ daily requests
        • Optimized database queries and implemented Redis caching, improving response times by 40%
        • Deployed applications on AWS cloud infrastructure using Docker containerization
        • Collaborated with frontend developers to integrate React.js user interfaces
        • Maintained code quality through Git version control and code reviews
        
        EDUCATION
        Bachelor of Science in Computer Science
        Stanford University, 2018
        
        SKILLS
        Programming: Python, JavaScript, HTML, CSS
        Frameworks: Django, React.js
        Databases: PostgreSQL, Redis
        Cloud: AWS (EC2, S3, RDS)
        Tools: Docker, Git, REST APIs
        
        PROJECTS
        ML-Powered Recommendation Engine
        Built machine learning recommendation system using Python, scikit-learn, and Django.
        Analyzed user behavior data to suggest relevant products, increasing engagement by 25%.
        """
    }
    
    weak_resume = {
        'name': 'John Smith',
        'email': 'john.smith@email.com',
        'skills': ['java', 'spring', 'mysql', 'html', 'css'],
        'education': [
            {
                'degree': 'Associate Degree in Web Development',
                'institution': 'Community College',
                'year': 2021
            }
        ],
        'certifications': [],
        'experience': [
            {
                'title': 'Junior Web Developer',
                'company': 'Small Agency',
                'start': '2021-06',
                'end': '2023-01',
                'bullets': [
                    'Built simple websites using HTML and CSS',
                    'Worked on Java Spring applications',
                    'Maintained MySQL databases'
                ]
            }
        ],
        'projects': [],
        'full_text': """
        John Smith
        Junior Web Developer
        john.smith@email.com
        
        EXPERIENCE
        Junior Web Developer at Small Agency (2021-2023)
        • Built simple websites using HTML and CSS
        • Worked on basic Java Spring applications
        • Maintained MySQL databases and performed simple queries
        • Fixed bugs and made minor feature updates
        
        EDUCATION
        Associate Degree in Web Development
        Community College, 2021
        
        SKILLS
        Programming: Java, HTML, CSS
        Framework: Spring
        Database: MySQL
        """
    }
    
    return sample_jd, strong_resume, weak_resume


def demo_embedding_generation():
    """Demonstrate text embedding generation."""
    print("🔧 EMBEDDING GENERATION DEMO")
    print("-" * 50)
    
    # Show current configuration
    embedding_info = get_embedding_info()
    print("📊 Current Configuration:")
    print(f"   Using OpenAI: {embedding_info['use_openai']}")
    print(f"   OpenAI Available: {embedding_info['openai_available']}")
    print(f"   Sentence Transformers Available: {embedding_info['sentence_transformers_available']}")
    print(f"   Default Model: {embedding_info['default_model']}")
    print(f"   Expected Dimensions: {embedding_info['expected_dimensions']}")
    
    # Test embedding generation
    sample_texts = [
        "Python web development with Django",
        "Machine learning and data science",
        "Frontend development with React"
    ]
    
    print(f"\n🧮 Generating embeddings for {len(sample_texts)} sample texts...")
    
    embeddings = []
    for i, text in enumerate(sample_texts, 1):
        try:
            embedding = get_embedding(text)
            embeddings.append(embedding)
            print(f"   {i}. '{text}' -> {len(embedding)} dimensions")
        except Exception as e:
            print(f"   {i}. Failed to embed '{text}': {e}")
    
    # Demonstrate similarity computation
    if len(embeddings) >= 2:
        similarity = compute_similarity(sample_texts[0], sample_texts[1])
        print(f"\n🔗 Similarity between texts 1 and 2: {similarity:.3f}")
    
    return embeddings


def demo_semantic_scoring(jd, resume_strong, resume_weak):
    """Demonstrate semantic similarity scoring."""
    print("\n📈 SEMANTIC SIMILARITY SCORING")
    print("-" * 50)
    
    # Test semantic scores
    print("Computing semantic scores...")
    
    strong_semantic_score = compute_semantic_score(resume_strong['full_text'], jd['full_text'])
    weak_semantic_score = compute_semantic_score(resume_weak['full_text'], jd['full_text'])
    
    print(f"\n📊 Results:")
    print(f"   Strong Resume (Sarah): {strong_semantic_score:.1f}/100")
    print(f"   Weak Resume (John): {weak_semantic_score:.1f}/100")
    print(f"   Score Difference: {strong_semantic_score - weak_semantic_score:.1f} points")
    
    # Find semantic matches
    print(f"\n🔍 Top Semantic Matches for Strong Resume:")
    jd_requirements = [
        "5+ years of Python development experience",
        "Strong proficiency in Django web framework", 
        "Experience with PostgreSQL database design",
        "REST API development and integration",
        "AWS cloud services experience"
    ]
    
    matches = find_semantic_matches(resume_strong['full_text'], jd_requirements, top_k=3)
    
    for i, match in enumerate(matches, 1):
        print(f"   {i}. Score: {match['score']:.1f}")
        print(f"      Resume: {match['resume_text'][:80]}...")
        print(f"      JD Match: {match['jd_match'][:80]}...")
        print()
    
    return strong_semantic_score, weak_semantic_score


def demo_combined_scoring(jd, resume_strong, resume_weak):
    """Demonstrate combined hard + semantic scoring."""
    print("\n⚖️  COMBINED SCORING (Hard + Semantic)")
    print("-" * 50)
    
    results = {}
    
    for name, resume in [("Strong Resume (Sarah)", resume_strong), ("Weak Resume (John)", resume_weak)]:
        print(f"\n👤 Analyzing: {name}")
        
        # Compute hard score first
        hard_result = compute_keyword_score(resume, jd)
        
        # Compute combined score
        combined_result = compute_combined_score(resume, jd, hard_result)
        
        results[name] = combined_result
        
        print(f"   Hard Score: {combined_result['hard_score']:.1f}/100")
        print(f"   Semantic Score: {combined_result['semantic_score']:.1f}/100")
        print(f"   Final Score: {combined_result['final_score']}/100")
        print(f"   Verdict: {combined_result['verdict'].upper()}")
        
        # Show missing elements
        missing = combined_result['missing_elements']
        if missing:
            print(f"   Missing Elements: {len(missing)} identified")
            for element in missing[:2]:  # Show first 2
                print(f"      • {element['type']}: {', '.join(element['items'][:3])}")
        
        # Show top semantic matches
        matches = combined_result['top_semantic_matches']
        if matches:
            print(f"   Top Semantic Match: {matches[0]['score']:.1f} - {matches[0]['resume_text'][:60]}...")
    
    return results


def demo_llm_feedback(jd, resume_strong, combined_results):
    """Demonstrate LLM-powered feedback generation."""
    print("\n🤖 LLM FEEDBACK GENERATION")
    print("-" * 50)
    
    # Check if OpenAI is available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OpenAI API key not found - will use template feedback")
        print("   Set OPENAI_API_KEY environment variable for LLM-powered feedback")
    else:
        print("✅ OpenAI API key found - generating personalized feedback")
    
    # Generate feedback for the strong resume
    strong_combined = combined_results["Strong Resume (Sarah)"]
    
    try:
        feedback_suggestions = generate_feedback(
            resume_strong, 
            jd, 
            strong_combined['score_breakdown'],
            num_suggestions=3
        )
        
        print(f"\n💡 Personalized Improvement Suggestions for Sarah:")
        
        for i, suggestion in enumerate(feedback_suggestions, 1):
            print(f"\n   {i}. {suggestion['action']}")
            print(f"      Example: {suggestion['example']}")
            print(f"      Priority: {suggestion['priority'].title()}")
            print(f"      Category: {suggestion['category'].title()}")
        
        return feedback_suggestions
        
    except Exception as e:
        print(f"❌ Feedback generation failed: {e}")
        return []


def demo_vector_storage():
    """Demonstrate vector storage and similarity search."""
    print("\n🗄️  VECTOR STORAGE & SEARCH DEMO")
    print("-" * 50)
    
    # Sample resume texts for indexing
    sample_resumes = [
        "Experienced Python developer with Django and PostgreSQL expertise",
        "Frontend specialist with React, JavaScript, and modern web technologies",
        "Data scientist with machine learning, Python, and statistical analysis skills",
        "DevOps engineer with AWS, Docker, and Kubernetes experience",
        "Full-stack developer with Python backend and React frontend skills"
    ]
    
    try:
        from semantic.embeddings import embed_and_index, search_similar_texts
        
        print(f"📚 Indexing {len(sample_resumes)} sample resumes...")
        
        # Create vector index
        vector_index = embed_and_index(
            sample_resumes, 
            persist=False,  # In-memory for demo
            backend="faiss" if hasattr(demo_vector_storage, '_use_faiss') else "chromadb"
        )
        
        if vector_index:
            print("✅ Vector index created successfully")
            
            # Test similarity search
            query = "Looking for Python web developer with database experience"
            print(f"\n🔍 Searching for: '{query}'")
            
            similar_docs = search_similar_texts(query, vector_index, top_k=3)
            
            print("📋 Top matches:")
            for i, doc in enumerate(similar_docs, 1):
                print(f"   {i}. Score: {doc['score']:.3f}")
                print(f"      Text: {doc['text']}")
                print()
        else:
            print("❌ Vector index creation failed")
            
    except Exception as e:
        print(f"❌ Vector storage demo failed: {e}")


def demo_performance_analysis(combined_results):
    """Analyze performance characteristics of the scoring system."""
    print("\n📊 PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    strong_result = combined_results["Strong Resume (Sarah)"]
    weak_result = combined_results["Weak Resume (John)"]
    
    print("🎯 Score Discrimination:")
    score_diff = strong_result['final_score'] - weak_result['final_score']
    print(f"   Score Range: {weak_result['final_score']} - {strong_result['final_score']}")
    print(f"   Discrimination: {score_diff} points")
    print(f"   Verdict Change: {weak_result['verdict']} -> {strong_result['verdict']}")
    
    print(f"\n⚖️  Component Analysis:")
    print(f"   Strong Resume:")
    print(f"      Hard: {strong_result['hard_score']:.1f}, Semantic: {strong_result['semantic_score']:.1f}")
    print(f"   Weak Resume:")
    print(f"      Hard: {weak_result['hard_score']:.1f}, Semantic: {weak_result['semantic_score']:.1f}")
    
    # Score balance analysis
    strong_balance = strong_result['score_breakdown']['performance_analysis']['balance']
    print(f"\n🔄 Score Balance:")
    print(f"   Strong Resume: {strong_balance}")
    
    # Recommendations comparison
    print(f"\n💼 Recommendations:")
    print(f"   Strong Resume: {len(strong_result['recommendations'])} recommendations")
    print(f"   Weak Resume: {len(weak_result['recommendations'])} recommendations")


def main():
    """Main demonstration function."""
    print("🚀 SEMANTIC MATCHING & LLM FEEDBACK DEMO")
    print("=" * 60)
    
    # Create sample data
    jd, resume_strong, resume_weak = create_sample_data()
    
    try:
        # Demo 1: Embedding generation
        embeddings = demo_embedding_generation()
        
        # Demo 2: Semantic scoring
        semantic_scores = demo_semantic_scoring(jd, resume_strong, resume_weak)
        
        # Demo 3: Combined scoring
        combined_results = demo_combined_scoring(jd, resume_strong, resume_weak)
        
        # Demo 4: LLM feedback
        feedback = demo_llm_feedback(jd, resume_strong, combined_results)
        
        # Demo 5: Vector storage (optional)
        try:
            demo_vector_storage()
        except Exception as e:
            print(f"⚠️  Vector storage demo skipped: {e}")
        
        # Demo 6: Performance analysis
        demo_performance_analysis(combined_results)
        
        print("\n" + "=" * 60)
        print("✅ SEMANTIC DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n🎯 Key Results:")
        strong_result = combined_results["Strong Resume (Sarah)"]
        weak_result = combined_results["Weak Resume (John)"]
        
        print(f"   Strong Candidate: {strong_result['final_score']}/100 ({strong_result['verdict']})")
        print(f"   Weak Candidate: {weak_result['final_score']}/100 ({weak_result['verdict']})")
        print(f"   System Discrimination: {strong_result['final_score'] - weak_result['final_score']} points")
        
        print(f"\n🛠️  Next Steps:")
        print(f"   • Try with your own resume/JD data")
        print(f"   • Experiment with different weight configurations")
        print(f"   • Set OPENAI_API_KEY for enhanced LLM feedback")
        print(f"   • Use USE_OPENAI_EMBEDDINGS=true for higher quality embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
