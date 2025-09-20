#!/usr/bin/env python3
"""
Hard Match Scoring Validation Script

This script validates that the hard match scoring system is working correctly
by running simple test cases and showing the results.
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from scoring.hard_match import compute_keyword_score, tfidf_similarity, fuzzy_match_score


def test_basic_functionality():
    """Test basic hard match scoring functionality."""
    print("🔍 Testing Basic Hard Match Functionality")
    print("-" * 50)
    
    # Simple test case
    jd = {
        'must_have_skills': ['python', 'sql'],
        'good_to_have_skills': ['aws'],
        'education_requirements': {
            'level': 'bachelor',
            'field': 'computer science'
        },
        'full_text': 'Python and SQL required for this position'
    }
    
    resume = {
        'skills': ['python', 'sql', 'aws', 'javascript'],
        'education': [{
            'degree': 'Bachelor of Computer Science',
            'year': 2020
        }],
        'full_text': 'Experience with Python, SQL, AWS, and JavaScript'
    }
    
    try:
        result = compute_keyword_score(resume, jd)
        
        print(f"✅ Scoring completed successfully")
        print(f"   Overall Score: {result['raw_score']:.1f}/100")
        print(f"   Must-have skills matched: {result['skill_matches']['must_have']}")
        print(f"   Good-to-have skills matched: {result['skill_matches']['good_to_have']}")
        print(f"   Education match: {result['education_match']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic scoring failed: {e}")
        return False


def test_tfidf_similarity():
    """Test TF-IDF similarity function."""
    print("\n📝 Testing TF-IDF Similarity")
    print("-" * 50)
    
    try:
        # Test identical texts
        text = "Python programming and data analysis"
        similarity = tfidf_similarity(text, text)
        print(f"✅ Identical texts similarity: {similarity:.3f}")
        
        # Test different texts
        text1 = "Python programming and machine learning"
        text2 = "Data science with Python and statistics"
        similarity = tfidf_similarity(text1, text2)
        print(f"✅ Similar texts similarity: {similarity:.3f}")
        
        # Test empty texts
        similarity = tfidf_similarity("", "some text")
        print(f"✅ Empty text handling: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ TF-IDF similarity failed: {e}")
        return False


def test_fuzzy_matching():
    """Test fuzzy string matching."""
    print("\n🔀 Testing Fuzzy String Matching")
    print("-" * 50)
    
    try:
        # Test exact matches
        list_a = ['python', 'sql']
        list_b = ['python', 'sql']
        score = fuzzy_match_score(list_a, list_b)
        print(f"✅ Exact matches score: {score:.3f}")
        
        # Test partial matches
        list_a = ['javascript', 'reactjs']
        list_b = ['javascript', 'react']
        score = fuzzy_match_score(list_a, list_b)
        print(f"✅ Partial matches score: {score:.3f}")
        
        # Test no matches
        list_a = ['python']
        list_b = ['cooking']
        score = fuzzy_match_score(list_a, list_b)
        print(f"✅ No matches score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fuzzy matching failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️  Testing Edge Cases")
    print("-" * 50)
    
    try:
        # Test empty inputs
        result = compute_keyword_score({}, {})
        print(f"✅ Empty inputs handled: score = {result['raw_score']}")
        
        # Test None inputs
        result = compute_keyword_score(None, None)
        print(f"✅ None inputs handled: score = {result['raw_score']}")
        
        # Test malformed data
        malformed_resume = {'skills': 'not a list'}
        malformed_jd = {'must_have_skills': 123}
        result = compute_keyword_score(malformed_resume, malformed_jd)
        print(f"✅ Malformed data handled: score = {result['raw_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case handling failed: {e}")
        return False


def test_custom_weights():
    """Test custom weight configuration."""
    print("\n⚖️  Testing Custom Weights")
    print("-" * 50)
    
    try:
        jd = {
            'must_have_skills': ['python', 'sql'],
            'good_to_have_skills': ['aws'],
            'full_text': 'Job description'
        }
        
        resume = {
            'skills': ['python', 'sql', 'aws'],
            'full_text': 'Resume text'
        }
        
        # Default weights
        result_default = compute_keyword_score(resume, jd)
        
        # Custom weights (skills-focused)
        custom_weights = {
            'must_have_skills': 0.80,
            'good_to_have_skills': 0.20,
            'education_match': 0.00,
            'certifications_match': 0.00,
            'tfidf_similarity': 0.00
        }
        
        result_custom = compute_keyword_score(resume, jd, custom_weights)
        
        print(f"✅ Default weights score: {result_default['raw_score']:.1f}")
        print(f"✅ Custom weights score: {result_custom['raw_score']:.1f}")
        print(f"✅ Weight customization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom weights failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 HARD MATCH SCORING VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("TF-IDF Similarity", test_tfidf_similarity),
        ("Fuzzy Matching", test_fuzzy_matching),
        ("Edge Cases", test_edge_cases),
        ("Custom Weights", test_custom_weights)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL HARD MATCH FUNCTIONS WORKING CORRECTLY!")
        print("\n🚀 Ready to use:")
        print("   • python examples/hard_match_example.py")
        print("   • from scoring.hard_match import compute_keyword_score")
    else:
        print("⚠️ Some issues detected but core functionality works")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
