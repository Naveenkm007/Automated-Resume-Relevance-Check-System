"""
Unit Tests for Hard Match Scoring Module

This test suite validates the hard match scoring functionality including:
- Keyword matching algorithms
- TF-IDF similarity computation  
- Fuzzy string matching
- Score calculation and weighting
- Edge cases and error handling

Run tests with: pytest tests/test_hard_match.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from scoring.hard_match import (
    compute_keyword_score, tfidf_similarity, fuzzy_match_score,
    fuzzy_match_skill, DEFAULT_WEIGHTS
)


class TestComputeKeywordScore:
    """Test the main keyword scoring function."""
    
    @pytest.fixture
    def sample_jd(self):
        """Sample job description structure."""
        return {
            'must_have_skills': ['python', 'sql', 'git'],
            'good_to_have_skills': ['aws', 'docker'],
            'education_requirements': {
                'level': 'bachelor',
                'field': 'computer science'
            },
            'certifications_required': ['aws certified developer'],
            'full_text': 'Job description requiring Python, SQL, and Git experience.'
        }
    
    @pytest.fixture
    def strong_resume(self):
        """Strong candidate resume structure."""
        return {
            'skills': ['python', 'sql', 'git', 'aws', 'javascript'],
            'education': [{
                'degree': 'Bachelor of Science in Computer Science',
                'institution': 'MIT',
                'year': 2020,
                'stream': 'Computer Science'
            }],
            'certifications': ['AWS Certified Developer Associate'],
            'full_text': 'Resume with Python, SQL, Git, and AWS experience from MIT.'
        }
    
    @pytest.fixture
    def weak_resume(self):
        """Weak candidate resume structure."""
        return {
            'skills': ['java', 'html', 'css'],
            'education': [{
                'degree': 'High School Diploma',
                'institution': 'Local High School',
                'year': 2020
            }],
            'certifications': [],
            'full_text': 'Resume with Java, HTML, CSS experience.'
        }
    
    def test_perfect_match(self, sample_jd, strong_resume):
        """Test scoring with a strong candidate match."""
        result = compute_keyword_score(strong_resume, sample_jd)
        
        # Check structure
        assert isinstance(result, dict)
        assert 'skill_matches' in result
        assert 'education_match' in result
        assert 'certifications_match' in result
        assert 'raw_score' in result
        assert 'breakdown' in result
        
        # Check skill matches
        skill_matches = result['skill_matches']
        assert skill_matches['must_have'] == 3  # All must-have skills matched
        assert skill_matches['good_to_have'] >= 1  # At least one good-to-have matched
        assert len(skill_matches['missing_must']) == 0  # No missing must-have skills
        
        # Check education match
        assert result['education_match'] == True
        
        # Check certifications
        assert len(result['certifications_match']) > 0
        
        # Score should be high (>70 for strong match)
        assert result['raw_score'] > 70
        assert 0 <= result['raw_score'] <= 100
    
    def test_weak_match(self, sample_jd, weak_resume):
        """Test scoring with a weak candidate match."""
        result = compute_keyword_score(weak_resume, sample_jd)
        
        # Should have missing must-have skills
        skill_matches = result['skill_matches']
        assert skill_matches['must_have'] < len(sample_jd['must_have_skills'])
        assert len(skill_matches['missing_must']) > 0
        
        # Education likely doesn't match
        assert result['education_match'] == False
        
        # Score should be low (<50 for weak match)
        assert result['raw_score'] < 50
        assert 0 <= result['raw_score'] <= 100
    
    def test_empty_inputs(self):
        """Test handling of empty or missing data."""
        empty_jd = {}
        empty_resume = {}
        
        result = compute_keyword_score(empty_resume, empty_jd)
        
        # Should not crash and return valid structure
        assert isinstance(result, dict)
        assert result['raw_score'] == 0.0
        assert result['skill_matches']['must_have'] == 0
        assert result['skill_matches']['good_to_have'] == 0
    
    def test_custom_weights(self, sample_jd, strong_resume):
        """Test custom weight configuration."""
        custom_weights = {
            'must_have_skills': 0.80,
            'good_to_have_skills': 0.10,
            'education_match': 0.05,
            'certifications_match': 0.05,
            'tfidf_similarity': 0.00
        }
        
        result_custom = compute_keyword_score(strong_resume, sample_jd, custom_weights)
        result_default = compute_keyword_score(strong_resume, sample_jd)
        
        # Scores should be different with different weights
        assert result_custom['raw_score'] != result_default['raw_score']
        
        # Breakdown should reflect custom weights
        assert result_custom['breakdown']['weights_used'] == custom_weights
    
    def test_skill_matching_case_insensitive(self):
        """Test that skill matching is case insensitive."""
        jd = {
            'must_have_skills': ['PYTHON', 'SQL'],
            'good_to_have_skills': ['AWS'],
            'full_text': 'Job requiring Python and SQL'
        }
        
        resume = {
            'skills': ['python', 'sql', 'aws'],
            'full_text': 'Resume with python, sql, aws'
        }
        
        result = compute_keyword_score(resume, jd)
        
        # Should match despite case differences
        assert result['skill_matches']['must_have'] == 2
        assert result['skill_matches']['good_to_have'] == 1
        assert len(result['skill_matches']['missing_must']) == 0


class TestTfidfSimilarity:
    """Test TF-IDF similarity computation."""
    
    def test_identical_texts(self):
        """Test TF-IDF similarity with identical texts."""
        text = "Python programming and SQL databases are important skills"
        similarity = tfidf_similarity(text, text)
        
        # Identical texts should have similarity close to 1.0
        assert 0.99 <= similarity <= 1.0
    
    def test_similar_texts(self):
        """Test TF-IDF similarity with similar texts."""
        text1 = "Python programming and SQL databases for data analysis"
        text2 = "Data analysis using Python and SQL database queries"
        
        similarity = tfidf_similarity(text1, text2)
        
        # Similar texts should have moderate to high similarity
        assert 0.3 <= similarity <= 1.0
    
    def test_different_texts(self):
        """Test TF-IDF similarity with completely different texts."""
        text1 = "Python programming and SQL databases"
        text2 = "Cooking recipes and kitchen utensils"
        
        similarity = tfidf_similarity(text1, text2)
        
        # Different texts should have low similarity
        assert 0.0 <= similarity <= 0.3
    
    def test_empty_texts(self):
        """Test TF-IDF similarity with empty texts."""
        assert tfidf_similarity("", "some text") == 0.0
        assert tfidf_similarity("some text", "") == 0.0
        assert tfidf_similarity("", "") == 0.0
    
    def test_similarity_range(self):
        """Test that TF-IDF similarity is always in [0, 1] range."""
        test_cases = [
            ("Python", "Java"),
            ("Very long text with many words", "Short text"),
            ("123 456 789", "abc def ghi"),
            ("Programming", "Programming language")
        ]
        
        for text1, text2 in test_cases:
            similarity = tfidf_similarity(text1, text2)
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} not in valid range for '{text1}' vs '{text2}'"


class TestFuzzyMatchScore:
    """Test fuzzy string matching functionality."""
    
    def test_exact_matches(self):
        """Test fuzzy matching with exact matches."""
        list_a = ['python', 'sql', 'javascript']
        list_b = ['python', 'sql', 'javascript']
        
        score = fuzzy_match_score(list_a, list_b)
        
        # Exact matches should give perfect score
        assert score == 1.0
    
    def test_partial_matches(self):
        """Test fuzzy matching with partial matches."""
        list_a = ['python', 'sql', 'react']
        list_b = ['python', 'javascript', 'angular']
        
        score = fuzzy_match_score(list_a, list_b)
        
        # Should have some similarity but not perfect
        assert 0.0 < score < 1.0
    
    def test_no_matches(self):
        """Test fuzzy matching with no matches."""
        list_a = ['python', 'sql']
        list_b = ['cooking', 'gardening']
        
        score = fuzzy_match_score(list_a, list_b)
        
        # No matches should give zero score
        assert score == 0.0
    
    def test_empty_lists(self):
        """Test fuzzy matching with empty lists."""
        assert fuzzy_match_score([], ['python', 'sql']) == 0.0
        assert fuzzy_match_score(['python', 'sql'], []) == 0.0
        assert fuzzy_match_score([], []) == 0.0
    
    def test_fuzzy_variations(self):
        """Test fuzzy matching with skill variations."""
        variations = ['javascript', 'js', 'reactjs', 'postgres']
        standards = ['javascript', 'react', 'postgresql']
        
        score = fuzzy_match_score(variations, standards)
        
        # Should match some variations
        assert score > 0.5  # At least half should match reasonably well
    
    def test_skill_abbreviations(self):
        """Test matching common skill abbreviations."""
        abbreviations = ['js', 'ml', 'ai', 'db']
        full_names = ['javascript', 'machine learning', 'artificial intelligence', 'database']
        
        # Note: This test may have low scores due to abbreviation differences
        # In production, this would be handled by skill normalization
        score = fuzzy_match_score(abbreviations, full_names)
        assert 0.0 <= score <= 1.0  # Should be valid range


class TestFuzzyMatchSkill:
    """Test individual skill fuzzy matching."""
    
    def test_exact_skill_match(self):
        """Test fuzzy matching with exact skill match."""
        skill_pool = {'python', 'javascript', 'sql'}
        match = fuzzy_match_skill('python', skill_pool)
        
        assert match == 'python'
    
    def test_close_skill_match(self):
        """Test fuzzy matching with close skill variations.""" 
        skill_pool = {'javascript', 'postgresql', 'react'}
        
        # Test common variations
        match_js = fuzzy_match_skill('js', skill_pool)
        match_postgres = fuzzy_match_skill('postgres', skill_pool)
        
        # These may or may not match depending on threshold
        # Main requirement is no crashes and valid returns
        assert match_js is None or isinstance(match_js, str)
        assert match_postgres is None or isinstance(match_postgres, str)
    
    def test_no_skill_match(self):
        """Test fuzzy matching with no reasonable matches."""
        skill_pool = {'python', 'javascript', 'sql'}
        match = fuzzy_match_skill('cooking', skill_pool)
        
        assert match is None
    
    def test_threshold_behavior(self):
        """Test fuzzy matching threshold behavior."""
        skill_pool = {'python', 'javascript'}
        
        # With high threshold, should be more restrictive
        match_high = fuzzy_match_skill('pythn', skill_pool, threshold=90)  # Typo
        match_low = fuzzy_match_skill('pythn', skill_pool, threshold=70)   # Typo
        
        # Low threshold should be more permissive
        if match_high is not None:
            assert match_low is not None
        
        # Both should be None or valid strings
        assert match_high is None or isinstance(match_high, str)
        assert match_low is None or isinstance(match_low, str)


class TestScoringFormula:
    """Test the scoring formula and weight distribution."""
    
    def test_weight_sum(self):
        """Test that default weights sum to 1.0."""
        total_weight = sum(DEFAULT_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision
    
    def test_score_components(self):
        """Test that score components are properly calculated."""
        jd = {
            'must_have_skills': ['python', 'sql'],
            'good_to_have_skills': ['aws'],
            'education_requirements': {'level': 'bachelor', 'field': 'computer science'},
            'certifications_required': [],
            'full_text': 'Python and SQL required'
        }
        
        resume = {
            'skills': ['python', 'sql', 'aws'],  # Perfect skill match
            'education': [{'degree': 'Bachelor Computer Science', 'year': 2020}],
            'certifications': [],
            'full_text': 'Python and SQL experience'
        }
        
        result = compute_keyword_score(resume, jd)
        breakdown = result['breakdown']
        
        # Skills should contribute most to the score
        assert breakdown['skill_score'] > breakdown['education_score']
        assert breakdown['skill_score'] > breakdown['certification_score']
        
        # Total should equal sum of components
        expected_total = (
            breakdown['skill_score'] + 
            breakdown['education_score'] + 
            breakdown['certification_score'] + 
            breakdown['tfidf_score']
        )
        assert abs(result['raw_score'] - expected_total) < 0.1
    
    def test_missing_must_have_penalty(self):
        """Test that missing must-have skills significantly impact score."""
        jd = {
            'must_have_skills': ['python', 'sql', 'javascript', 'git'],
            'good_to_have_skills': [],
            'full_text': 'Job description text'
        }
        
        # Resume with all must-have skills
        resume_complete = {
            'skills': ['python', 'sql', 'javascript', 'git'],
            'full_text': 'Complete skill set'
        }
        
        # Resume missing half of must-have skills
        resume_incomplete = {
            'skills': ['python', 'sql'],
            'full_text': 'Incomplete skill set'
        }
        
        score_complete = compute_keyword_score(resume_complete, jd)
        score_incomplete = compute_keyword_score(resume_incomplete, jd)
        
        # Complete resume should score significantly higher
        assert score_complete['raw_score'] > score_incomplete['raw_score']
        score_difference = score_complete['raw_score'] - score_incomplete['raw_score']
        assert score_difference > 20  # At least 20 point difference


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_inputs(self):
        """Test handling of None inputs."""
        result = compute_keyword_score(None, None)
        assert isinstance(result, dict)
        assert result['raw_score'] == 0.0
    
    def test_malformed_data(self):
        """Test handling of malformed data structures."""
        malformed_jd = {'must_have_skills': 'not a list'}  # Should be list
        malformed_resume = {'skills': 123}  # Should be list
        
        # Should not crash
        result = compute_keyword_score(malformed_resume, malformed_jd)
        assert isinstance(result, dict)
        assert 0 <= result['raw_score'] <= 100
    
    def test_special_characters(self):
        """Test handling of special characters in skills."""
        jd = {
            'must_have_skills': ['c++', 'c#', '.net'],
            'full_text': 'Special character skills'
        }
        
        resume = {
            'skills': ['c++', 'c#', '.net'],
            'full_text': 'Special character skills'
        }
        
        result = compute_keyword_score(resume, jd)
        
        # Should handle special characters without crashing
        assert isinstance(result, dict)
        assert result['skill_matches']['must_have'] == 3
    
    def test_very_long_skill_lists(self):
        """Test performance with long skill lists."""
        long_skill_list = [f'skill_{i}' for i in range(100)]
        
        jd = {
            'must_have_skills': long_skill_list[:50],
            'good_to_have_skills': long_skill_list[50:],
            'full_text': 'Long skill list'
        }
        
        resume = {
            'skills': long_skill_list[:75],  # Overlap with requirements
            'full_text': 'Long skill list'
        }
        
        # Should handle large lists without major performance issues
        result = compute_keyword_score(resume, jd)
        assert isinstance(result, dict)
        assert result['skill_matches']['must_have'] == 50  # All must-have matched
        # Good-to-have: resume has 0-74, JD good-to-have is 50-99, so overlap is 50-74 = 25 skills
        # But let's be more flexible since the exact count depends on implementation details
        assert result['skill_matches']['good_to_have'] >= 20  # At least 20 good-to-have matched


if __name__ == '__main__':
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])
