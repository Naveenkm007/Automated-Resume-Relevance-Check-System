"""
Unit Tests for Semantic Matching Module

This test suite validates the semantic matching functionality including:
- Text embedding generation and similarity computation
- Semantic scoring and cosine similarity mapping
- Combined scoring integration
- LLM feedback generation (with mocking)
- Vector storage and retrieval operations

Run tests with: pytest tests/test_semantic.py -v
"""

import pytest
import sys
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from semantic.embeddings import get_embedding, compute_similarity, get_embedding_info
from semantic.similarity import compute_semantic_score, find_semantic_matches, analyze_semantic_coverage
from semantic.combined_score import compute_combined_score, get_role_specific_weights
from semantic.feedback import generate_feedback


class TestEmbeddings:
    """Test embedding generation and similarity computation."""
    
    def test_get_embedding_basic(self):
        """Test basic embedding generation."""
        text = "Python web development with Django"
        
        try:
            embedding = get_embedding(text)
            
            # Should return numpy array
            assert isinstance(embedding, np.ndarray)
            
            # Should have reasonable dimensions (384 for sentence-transformers, 1536 for OpenAI)
            assert embedding.shape[0] in [384, 1536]
            
            # Should be normalized (values between -1 and 1 roughly)
            assert np.all(np.abs(embedding) <= 2.0)  # Allow some flexibility
            
        except Exception as e:
            # If no embedding backend available, should handle gracefully
            assert "embedding backend" in str(e).lower()
    
    def test_get_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        embedding = get_embedding("")
        
        # Should return zero vector
        assert isinstance(embedding, np.ndarray)
        assert np.allclose(embedding, 0.0)
    
    def test_compute_similarity_identical(self):
        """Test similarity computation with identical texts."""
        text = "Machine learning and data science"
        
        similarity = compute_similarity(text, text)
        
        # Identical texts should have high similarity (close to 1.0)
        assert 0.95 <= similarity <= 1.0
    
    def test_compute_similarity_different(self):
        """Test similarity computation with different texts."""
        text1 = "Python programming and web development"
        text2 = "Cooking recipes and kitchen management"
        
        similarity = compute_similarity(text1, text2)
        
        # Different domain texts should have low similarity
        assert 0.0 <= similarity <= 0.5
    
    def test_compute_similarity_related(self):
        """Test similarity computation with related texts."""
        text1 = "Python web development with Django"
        text2 = "Web application development using Python framework"
        
        similarity = compute_similarity(text1, text2)
        
        # Related texts should have moderate to high similarity
        assert 0.3 <= similarity <= 1.0
    
    def test_get_embedding_info(self):
        """Test embedding configuration information."""
        info = get_embedding_info()
        
        # Should return dict with expected keys
        required_keys = [
            'use_openai', 'openai_available', 'sentence_transformers_available',
            'chromadb_available', 'faiss_available', 'default_model', 'expected_dimensions'
        ]
        
        for key in required_keys:
            assert key in info
        
        # Dimensions should be reasonable
        assert info['expected_dimensions'] in [384, 1536]


class TestSemanticSimilarity:
    """Test semantic similarity scoring functions."""
    
    def test_compute_semantic_score_range(self):
        """Test that semantic scores are in valid range."""
        resume_text = "Experienced Python developer with Django and PostgreSQL"
        jd_text = "Looking for Python developer with web framework experience"
        
        score = compute_semantic_score(resume_text, jd_text)
        
        # Score should be in 0-100 range
        assert 0.0 <= score <= 100.0
        assert isinstance(score, float)
    
    def test_compute_semantic_score_similar_texts(self):
        """Test semantic scoring with similar texts."""
        resume_text = "Python web developer with Django REST API experience"
        jd_text = "Seeking Python developer for Django web application development"
        
        score = compute_semantic_score(resume_text, jd_text)
        
        # Similar texts should score high
        assert score >= 60.0  # Should be good match
    
    def test_compute_semantic_score_different_texts(self):
        """Test semantic scoring with different texts."""
        resume_text = "Experienced chef with culinary arts background"
        jd_text = "Looking for Python developer with web framework experience"
        
        score = compute_semantic_score(resume_text, jd_text)
        
        # Different domains should score low
        assert score <= 40.0  # Should be poor match
    
    def test_compute_semantic_score_empty_texts(self):
        """Test semantic scoring with empty texts."""
        assert compute_semantic_score("", "some text") == 0.0
        assert compute_semantic_score("some text", "") == 0.0
        assert compute_semantic_score("", "") == 0.0
    
    def test_find_semantic_matches(self):
        """Test finding semantic matches between resume and JD sentences."""
        resume_text = """
        Python developer with 5 years experience.
        Built web applications using Django framework.
        Worked with PostgreSQL databases and REST APIs.
        """
        
        jd_sentences = [
            "5+ years of Python development experience",
            "Django web framework proficiency required",
            "Database design and REST API development"
        ]
        
        matches = find_semantic_matches(resume_text, jd_sentences, top_k=3)
        
        # Should return list of matches
        assert isinstance(matches, list)
        assert len(matches) <= 3
        
        # Each match should have required fields
        for match in matches:
            assert 'resume_text' in match
            assert 'jd_match' in match
            assert 'score' in match
            assert isinstance(match['score'], float)
            assert 0.0 <= match['score'] <= 100.0
    
    def test_analyze_semantic_coverage(self):
        """Test semantic coverage analysis."""
        resume_text = "Python developer with Django and PostgreSQL experience"
        requirements = [
            "Python programming skills required",
            "Django framework experience needed",
            "Database management with PostgreSQL",
            "Machine learning expertise preferred"  # Not covered
        ]
        
        analysis = analyze_semantic_coverage(resume_text, requirements)
        
        # Should return analysis dict
        assert isinstance(analysis, dict)
        required_keys = ['coverage_score', 'covered_requirements', 'gaps', 'total_requirements']
        for key in required_keys:
            assert key in analysis
        
        # Coverage score should be reasonable
        assert 0.0 <= analysis['coverage_score'] <= 100.0
        
        # Should have some covered requirements and some gaps
        assert analysis['total_requirements'] == 4
        assert len(analysis['covered_requirements']) + len(analysis['gaps']) == 4


class TestCombinedScoring:
    """Test combined hard + semantic scoring."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for combined scoring tests."""
        resume_struct = {
            'skills': ['python', 'django', 'postgresql'],
            'education': [{'degree': 'Bachelor Computer Science', 'year': 2020}],
            'certifications': [],
            'full_text': 'Python developer with Django and PostgreSQL experience'
        }
        
        jd_struct = {
            'must_have_skills': ['python', 'django'],
            'good_to_have_skills': ['postgresql', 'redis'],
            'education_requirements': {'level': 'bachelor'},
            'certifications_required': [],
            'full_text': 'Looking for Python Django developer'
        }
        
        hard_score_result = {
            'raw_score': 75.0,
            'skill_matches': {
                'must_have': 2,
                'good_to_have': 1,
                'missing_must': [],
                'missing_good': ['redis'],
                'matched_must': ['python', 'django'],
                'matched_good': ['postgresql']
            },
            'education_match': True,
            'certifications_match': [],
            'breakdown': {
                'skill_score': 65.0,
                'education_score': 10.0,
                'certification_score': 0.0,
                'tfidf_score': 0.0
            }
        }
        
        return resume_struct, jd_struct, hard_score_result
    
    def test_compute_combined_score_basic(self, sample_data):
        """Test basic combined scoring functionality."""
        resume_struct, jd_struct, hard_score_result = sample_data
        
        result = compute_combined_score(resume_struct, jd_struct, hard_score_result)
        
        # Should return dict with required keys
        required_keys = [
            'final_score', 'verdict', 'hard_score', 'semantic_score',
            'missing_elements', 'top_semantic_matches', 'score_breakdown', 'recommendations'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Final score should be in valid range
        assert 0 <= result['final_score'] <= 100
        assert isinstance(result['final_score'], int)
        
        # Verdict should be valid
        assert result['verdict'] in ['high', 'medium', 'low']
        
        # Should have semantic score
        assert 0.0 <= result['semantic_score'] <= 100.0
    
    def test_compute_combined_score_weights(self, sample_data):
        """Test combined scoring with custom weights."""
        resume_struct, jd_struct, hard_score_result = sample_data
        
        # Test with skills-focused weights
        result1 = compute_combined_score(
            resume_struct, jd_struct, hard_score_result,
            hard_weight=0.8, semantic_weight=0.2
        )
        
        # Test with semantic-focused weights
        result2 = compute_combined_score(
            resume_struct, jd_struct, hard_score_result,
            hard_weight=0.3, semantic_weight=0.7
        )
        
        # Results should be different
        assert result1['final_score'] != result2['final_score']
        
        # Both should be valid
        assert 0 <= result1['final_score'] <= 100
        assert 0 <= result2['final_score'] <= 100
    
    def test_get_role_specific_weights(self):
        """Test role-specific weight configurations."""
        # Test different role types
        roles = ['technical', 'creative', 'sales', 'management', 'entry_level', 'senior']
        
        for role in roles:
            hard_weight, semantic_weight = get_role_specific_weights(role)
            
            # Weights should sum to 1.0
            assert abs(hard_weight + semantic_weight - 1.0) < 0.001
            
            # Both weights should be positive
            assert 0.0 < hard_weight < 1.0
            assert 0.0 < semantic_weight < 1.0
        
        # Test default case
        hard_weight, semantic_weight = get_role_specific_weights('unknown_role')
        assert abs(hard_weight + semantic_weight - 1.0) < 0.001
    
    def test_combined_score_missing_elements(self, sample_data):
        """Test extraction of missing elements."""
        resume_struct, jd_struct, hard_score_result = sample_data
        
        # Modify to have missing elements
        hard_score_result['skill_matches']['missing_must'] = ['git', 'docker']
        hard_score_result['education_match'] = False
        
        result = compute_combined_score(resume_struct, jd_struct, hard_score_result)
        
        missing_elements = result['missing_elements']
        
        # Should identify missing elements
        assert len(missing_elements) > 0
        
        # Each element should have required fields
        for element in missing_elements:
            assert 'type' in element
            assert 'items' in element
            assert 'priority' in element
            assert element['priority'] in ['critical', 'moderate', 'low']


class TestFeedbackGeneration:
    """Test LLM feedback generation."""
    
    @pytest.fixture
    def feedback_data(self):
        """Sample data for feedback generation tests."""
        resume_struct = {
            'name': 'John Doe',
            'skills': ['python', 'html', 'css'],
            'education': [{'degree': 'Bachelor Computer Science'}],
            'certifications': [],
            'experience': [{'title': 'Junior Developer', 'company': 'StartupXYZ'}],
            'projects': []
        }
        
        jd_struct = {
            'title': 'Senior Python Developer',
            'must_have_skills': ['python', 'django', 'postgresql'],
            'good_to_have_skills': ['react', 'aws'],
            'education_requirements': {'level': 'bachelor'},
            'certifications_required': []
        }
        
        score_breakdown = {
            'raw_score': 45.0,
            'skill_matches': {
                'missing_must': ['django', 'postgresql'],
                'missing_good': ['react', 'aws']
            },
            'education_match': True
        }
        
        return resume_struct, jd_struct, score_breakdown
    
    def test_generate_feedback_structure(self, feedback_data):
        """Test feedback generation returns proper structure."""
        resume_struct, jd_struct, score_breakdown = feedback_data
        
        # Mock OpenAI to test structure without API calls
        with patch('semantic.feedback.OPENAI_AVAILABLE', False):
            suggestions = generate_feedback(resume_struct, jd_struct, score_breakdown, num_suggestions=3)
        
        # Should return list of suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        
        # Each suggestion should have required fields
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            required_fields = ['action', 'example', 'priority', 'category']
            for field in required_fields:
                assert field in suggestion
                assert isinstance(suggestion[field], str)
                assert len(suggestion[field]) > 0
            
            # Priority should be valid
            assert suggestion['priority'] in ['high', 'medium', 'low']
            
            # Category should be valid
            assert suggestion['category'] in ['skill', 'experience', 'education', 'format']
    
    @patch('semantic.feedback.OPENAI_AVAILABLE', True)
    @patch('semantic.feedback.OPENAI_API_KEY', 'test-key')
    def test_generate_feedback_with_mock_llm(self, feedback_data):
        """Test feedback generation with mocked LLM."""
        resume_struct, jd_struct, score_breakdown = feedback_data
        
        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        [
            {
                "action": "Learn Django web framework to meet job requirements",
                "example": "Complete Django tutorial and build a web application",
                "priority": "high",
                "category": "skill"
            }
        ]
        '''
        
        with patch('semantic.feedback.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            suggestions = generate_feedback(resume_struct, jd_struct, score_breakdown, num_suggestions=1)
        
        # Should parse LLM response correctly
        assert len(suggestions) == 1
        assert suggestions[0]['action'] == "Learn Django web framework to meet job requirements"
        assert suggestions[0]['priority'] == "high"
        assert suggestions[0]['category'] == "skill"
    
    def test_generate_feedback_fallback(self, feedback_data):
        """Test feedback generation fallback when LLM unavailable."""
        resume_struct, jd_struct, score_breakdown = feedback_data
        
        # Test without OpenAI
        with patch('semantic.feedback.OPENAI_AVAILABLE', False):
            suggestions = generate_feedback(resume_struct, jd_struct, score_breakdown, num_suggestions=2)
        
        # Should still return suggestions (template-based)
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 2
        
        # Should focus on missing skills from score breakdown
        suggestion_text = ' '.join([s['action'] + s['example'] for s in suggestions])
        missing_skills = score_breakdown['skill_matches']['missing_must']
        
        # At least one missing skill should be mentioned
        assert any(skill in suggestion_text.lower() for skill in missing_skills)


class TestSemanticIntegration:
    """Integration tests for the complete semantic matching pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete semantic matching pipeline."""
        # Sample data
        resume_text = "Python developer with Django web framework experience and PostgreSQL database skills"
        jd_text = "Looking for experienced Python developer with Django and database experience"
        
        # Test semantic score computation
        semantic_score = compute_semantic_score(resume_text, jd_text)
        assert 0.0 <= semantic_score <= 100.0
        
        # Test semantic matches
        jd_requirements = ["Python development experience", "Django framework knowledge"]
        matches = find_semantic_matches(resume_text, jd_requirements, top_k=2)
        assert isinstance(matches, list)
        assert len(matches) <= 2
    
    def test_score_consistency(self):
        """Test that scoring is consistent across multiple runs."""
        resume_text = "Machine learning engineer with Python and TensorFlow"
        jd_text = "Seeking ML engineer with Python and deep learning frameworks"
        
        # Compute score multiple times
        scores = []
        for _ in range(3):
            score = compute_semantic_score(resume_text, jd_text)
            scores.append(score)
        
        # Scores should be identical (deterministic)
        assert all(abs(score - scores[0]) < 0.1 for score in scores)
    
    def test_similarity_properties(self):
        """Test mathematical properties of similarity function."""
        texts = [
            "Python web development",
            "Machine learning with Python", 
            "Frontend JavaScript development"
        ]
        
        # Test symmetry: sim(A,B) = sim(B,A)
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                sim1 = compute_semantic_score(texts[i], texts[j])
                sim2 = compute_semantic_score(texts[j], texts[i])
                assert abs(sim1 - sim2) < 0.1  # Should be nearly identical
        
        # Test identity: sim(A,A) should be high
        for text in texts:
            sim = compute_semantic_score(text, text)
            assert sim >= 95.0  # Self-similarity should be very high


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test None inputs
        score = compute_semantic_score(None, "test text")
        assert score == 0.0
        
        score = compute_semantic_score("test text", None)
        assert score == 0.0
        
        # Test empty inputs
        score = compute_semantic_score("", "")
        assert score == 0.0
    
    def test_very_long_texts(self):
        """Test handling of very long texts."""
        long_text = "Python development " * 1000  # Very long text
        normal_text = "Python programming experience"
        
        # Should not crash and return valid score
        score = compute_semantic_score(long_text, normal_text)
        assert 0.0 <= score <= 100.0
    
    def test_special_characters(self):
        """Test handling of special characters and encoding."""
        text_with_special = "Python développeur with émojis 🐍💻 and speciál chäracteŕs"
        normal_text = "Python developer with programming skills"
        
        # Should handle special characters gracefully
        score = compute_semantic_score(text_with_special, normal_text)
        assert 0.0 <= score <= 100.0
    
    def test_different_languages(self):
        """Test handling of different languages."""
        english_text = "Python programming and web development"
        non_english_text = "Programación en Python y desarrollo web"  # Spanish
        
        # Should return low but valid similarity
        score = compute_semantic_score(english_text, non_english_text)
        assert 0.0 <= score <= 100.0


if __name__ == '__main__':
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])
