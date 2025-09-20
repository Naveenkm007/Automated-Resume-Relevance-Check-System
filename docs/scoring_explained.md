# Resume Scoring System - Technical Guide

## Overview

The Resume Relevance Check system uses a **hybrid scoring approach** that combines keyword-based hard matching with semantic similarity analysis to provide comprehensive candidate evaluation.

## Scoring Architecture

```
Resume + Job Description
         ↓
    ┌────────────────┬────────────────┐
    │  Hard Match    │ Semantic Match │
    │  (Keywords)    │  (Embeddings)  │
    │    Weight: α   │   Weight: β    │
    └────────────────┴────────────────┘
         ↓
    Final Score = α × Hard + β × Semantic
         ↓
    Verdict Classification
```

## 1. Hard Matching Component

### Algorithm
Hard matching uses exact keyword matching with fuzzy string similarity for skill assessment.

**Core Formula:**
```
Hard Score = Σ(Component Weight × Component Score)

Components:
- Skills Match (40%): Must-have + Good-to-have skills
- Education Match (20%): Degree level and field alignment  
- Experience Match (25%): Years and domain relevance
- Certification Match (15%): Required certifications present
```

### Skills Scoring
```python
def compute_skill_score(resume_skills, jd_skills):
    # Must-have skills (critical)
    must_have_matches = fuzzy_match(resume_skills, jd_must_have)
    must_have_score = (len(must_have_matches) / len(jd_must_have)) * 100
    
    # Good-to-have skills (bonus)
    good_to_have_matches = fuzzy_match(resume_skills, jd_good_to_have)
    good_to_have_score = min((len(good_to_have_matches) / len(jd_good_to_have)) * 100, 50)
    
    # Combined: Must-have is 70%, Good-to-have is 30%
    skill_score = (must_have_score * 0.7) + (good_to_have_score * 0.3)
    return min(skill_score, 100)
```

### Fuzzy Matching
Uses `rapidfuzz` library with Levenshtein distance:
```python
def fuzzy_match(skill_list, target_skills, threshold=80):
    matches = []
    for skill in skill_list:
        for target in target_skills:
            similarity = fuzz.ratio(skill.lower(), target.lower())
            if similarity >= threshold:
                matches.append((skill, target, similarity))
    return matches
```

### Education Scoring
```python
EDUCATION_HIERARCHY = {
    'phd': 4, 'doctorate': 4,
    'masters': 3, 'mba': 3,
    'bachelors': 2, 'undergraduate': 2,
    'associates': 1, 'diploma': 1
}

def education_score(resume_education, required_education):
    resume_level = EDUCATION_HIERARCHY.get(resume_education, 0)
    required_level = EDUCATION_HIERARCHY.get(required_education, 0)
    
    if resume_level >= required_level:
        return 100  # Meets requirement
    elif resume_level == required_level - 1:
        return 75   # Close match
    else:
        return max(0, 50 - (required_level - resume_level) * 15)
```

## 2. Semantic Matching Component

### Embedding Generation
Uses sentence-transformers for text vectorization:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional vectors

def get_embedding(text):
    # Preprocess text
    clean_text = preprocess_text(text)
    
    # Generate embedding
    embedding = model.encode(clean_text, normalize_embeddings=True)
    return embedding
```

### Cosine Similarity Calculation
```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

def semantic_score(resume_text, jd_text):
    resume_embedding = get_embedding(resume_text)
    jd_embedding = get_embedding(jd_text)
    
    similarity = cosine_similarity(resume_embedding, jd_embedding)
    
    # Convert to 0-100 scale with improved mapping
    score = ((similarity + 1) / 2) * 100  # Map [-1,1] to [0,100]
    return min(max(score, 0), 100)
```

### Text Preprocessing
```python
def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(words)
```

## 3. Combined Scoring Formula

### Default Weight Configuration
```python
DEFAULT_WEIGHTS = {
    'hard_match_weight': 0.6,      # 60% - Keyword matching
    'semantic_match_weight': 0.4   # 40% - Semantic similarity
}

def compute_final_score(hard_score, semantic_score, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    final_score = (
        hard_score * weights['hard_match_weight'] +
        semantic_score * weights['semantic_match_weight']
    )
    
    return round(final_score)
```

### Verdict Classification
```python
def classify_verdict(final_score):
    if final_score >= 80:
        return "high"      # Strong match - recommend for interview
    elif final_score >= 50:
        return "medium"    # Moderate match - consider for interview
    else:
        return "low"       # Weak match - likely not suitable
```

## 4. Weight Tuning Guidelines

### Industry-Specific Adjustments

**Technical Roles (Software Engineering):**
```python
TECH_WEIGHTS = {
    'hard_match_weight': 0.7,    # Higher emphasis on specific skills
    'semantic_match_weight': 0.3
}
```

**Creative Roles (Marketing, Design):**
```python
CREATIVE_WEIGHTS = {
    'hard_match_weight': 0.4,    # Lower emphasis on exact keywords
    'semantic_match_weight': 0.6  # Higher emphasis on conceptual match
}
```

**Senior Roles (Management):**
```python
SENIOR_WEIGHTS = {
    'hard_match_weight': 0.5,    # Balanced approach
    'semantic_match_weight': 0.5,
    'experience_weight': 1.2     # Boost experience component
}
```

### Performance Tuning

**Optimizing for Precision (fewer false positives):**
```python
HIGH_PRECISION_WEIGHTS = {
    'hard_match_weight': 0.8,
    'semantic_match_weight': 0.2,
    'must_have_skills_threshold': 0.8  # Require 80% must-have skills
}
```

**Optimizing for Recall (fewer false negatives):**
```python
HIGH_RECALL_WEIGHTS = {
    'hard_match_weight': 0.4,
    'semantic_match_weight': 0.6,
    'fuzzy_threshold': 70  # Lower fuzzy matching threshold
}
```

## 5. Advanced Scoring Features

### Skill Importance Weighting
```python
SKILL_WEIGHTS = {
    'python': 1.5,      # Critical for Python roles
    'leadership': 1.3,   # Important for senior roles
    'communication': 1.2,
    'teamwork': 1.0,    # Standard weight
    'basic skills': 0.8  # Lower importance
}

def weighted_skill_score(matched_skills):
    total_weight = sum(SKILL_WEIGHTS.get(skill, 1.0) for skill in matched_skills)
    max_weight = sum(SKILL_WEIGHTS.get(skill, 1.0) for skill in all_required_skills)
    return (total_weight / max_weight) * 100
```

### Experience Level Adjustment
```python
def experience_adjustment(base_score, resume_years, required_years):
    if resume_years >= required_years:
        # Bonus for exceeding requirements
        bonus = min((resume_years - required_years) * 2, 10)
        return min(base_score + bonus, 100)
    else:
        # Penalty for insufficient experience
        penalty = (required_years - resume_years) * 5
        return max(base_score - penalty, 0)
```

### Recency Weighting
```python
def apply_recency_weight(skills, experience_data):
    current_year = datetime.now().year
    weighted_skills = []
    
    for skill in skills:
        # Find most recent usage
        recent_usage = max([
            exp.get('end_year', current_year) 
            for exp in experience_data 
            if skill.lower() in exp.get('description', '').lower()
        ], default=current_year - 10)
        
        # Apply decay: recent = weight 1.0, 5+ years ago = weight 0.6
        years_ago = current_year - recent_usage
        weight = max(1.0 - (years_ago * 0.08), 0.6)
        
        weighted_skills.append((skill, weight))
    
    return weighted_skills
```

## 6. Configuration Management

### Environment-Based Weights
```python
class ScoringConfig:
    def __init__(self, environment='production'):
        if environment == 'development':
            self.weights = DEV_WEIGHTS
            self.thresholds = DEV_THRESHOLDS
        elif environment == 'production':
            self.weights = PROD_WEIGHTS
            self.thresholds = PROD_THRESHOLDS
        
        self.load_custom_weights()
    
    def load_custom_weights(self):
        # Load from database or config file
        custom_weights = self.get_stored_weights()
        if custom_weights:
            self.weights.update(custom_weights)
```

### A/B Testing Framework
```python
def get_scoring_variant(user_id):
    # Consistent assignment based on user ID
    hash_value = hashlib.md5(user_id.encode()).hexdigest()
    variant = int(hash_value[:8], 16) % 100
    
    if variant < 50:
        return 'control'  # Current weights
    else:
        return 'experiment'  # New weights being tested

def score_with_variant(resume, jd, user_id):
    variant = get_scoring_variant(user_id)
    weights = VARIANT_WEIGHTS[variant]
    
    return compute_combined_score(resume, jd, weights)
```

## 7. Performance Metrics & Monitoring

### Key Performance Indicators
```python
def calculate_scoring_metrics(predictions, actual_hires):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score,
        'accuracy': (true_positives + true_negatives) / total_predictions
    }
```

### Score Distribution Analysis
```python
def analyze_score_distribution(scores):
    return {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'percentiles': {
            '25th': np.percentile(scores, 25),
            '75th': np.percentile(scores, 75),
            '90th': np.percentile(scores, 90)
        }
    }
```

## 8. Troubleshooting Common Issues

### Low Scores for Good Candidates
1. **Check fuzzy matching threshold** - Lower if too strict
2. **Review semantic preprocessing** - Ensure key terms aren't removed
3. **Adjust semantic weight** - Increase for conceptual matches
4. **Verify skill extraction** - Check if resume parsing missed skills

### High Scores for Poor Candidates  
1. **Increase hard matching weight** - Emphasize exact requirements
2. **Raise must-have skills threshold** - Require higher percentage
3. **Add negative keywords** - Penalize irrelevant experience
4. **Review education requirements** - Ensure proper validation

### Inconsistent Scoring
1. **Check text preprocessing consistency**
2. **Verify embedding model version**
3. **Ensure weight configuration is stable**
4. **Monitor for data quality issues**

## 9. Future Enhancements

### Machine Learning Integration
- **Learning from hiring decisions** to automatically adjust weights
- **Candidate success prediction** based on performance data
- **Dynamic skill importance** based on market trends

### Advanced NLP Features
- **Named Entity Recognition** for better skill extraction
- **Context-aware matching** (e.g., "Python" programming vs "Python" snake)
- **Multilingual support** for international candidates

### Personalization
- **Hiring manager preferences** integration
- **Role-specific scoring models**
- **Company culture fit analysis**

---

*For implementation details and code examples, see the source code in `/scoring/` and `/semantic/` directories.*
