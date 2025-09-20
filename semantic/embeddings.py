"""
Text Embedding Generation Module

This module handles converting text into numerical vector representations (embeddings)
that capture semantic meaning. Embeddings allow us to measure semantic similarity
between resumes and job descriptions beyond simple keyword matching.

What are embeddings?
Embeddings are dense numerical vectors that represent the meaning of text. Similar
texts have similar embeddings, even if they use different words. For example,
"Python developer" and "Software engineer with Python expertise" would have 
similar embeddings despite different wording.

Why embeddings work:
- Capture semantic relationships between words and concepts
- Handle synonyms and paraphrasing naturally
- Work across different writing styles and formats
- Enable similarity search in high-dimensional semantic space

Supported backends:
1. Sentence Transformers (local): Fast, private, works offline
2. OpenAI Embeddings (cloud): Higher quality, requires API key
3. Vector storage: ChromaDB (local) or FAISS (in-memory)
"""

import os
import logging
import numpy as np
from typing import List, Union, Optional, Dict, Any
import pickle
from pathlib import Path

# Environment configuration
USE_OPENAI_EMBEDDINGS = os.getenv('USE_OPENAI_EMBEDDINGS', 'false').lower() == 'true'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Sentence Transformers (local embeddings)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - install with: pip install sentence-transformers")

# OpenAI embeddings (cloud)
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available - install with: pip install openai")

# Vector storage backends
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("chromadb not available - install with: pip install chromadb")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available - install with: pip install faiss-cpu")

# Default model configurations
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's current embedding model

# Global model instances (lazy loaded)
_sentence_model = None
_openai_client = None


def get_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding vector for input text.
    
    This function converts text into a dense numerical vector that captures
    its semantic meaning. The choice of embedding model affects quality:
    
    - Sentence Transformers (local): 
      * Pros: Fast, private, works offline, good for most use cases
      * Cons: Limited by model size, less nuanced than large cloud models
      * Best for: Production systems prioritizing speed and privacy
    
    - OpenAI Embeddings (cloud):
      * Pros: State-of-the-art quality, handles complex language well  
      * Cons: Requires API key, costs money, needs internet, slower
      * Best for: High-accuracy applications with budget for API calls
    
    The embedding vector can then be used for:
    - Similarity search (find similar resumes/jobs)
    - Clustering (group similar documents)
    - Classification (match to categories)
    - Anomaly detection (find unusual content)
    
    Args:
        text (str): Input text to embed
        model_name (Optional[str]): Specific model to use, overrides global setting
        
    Returns:
        np.ndarray: Dense embedding vector (typically 384-1536 dimensions)
        
    Raises:
        ValueError: If no embedding backend is available
        Exception: For API or model loading errors
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        if USE_OPENAI_EMBEDDINGS and OPENAI_AVAILABLE:
            return np.zeros(1536)  # OpenAI embedding dimension
        else:
            return np.zeros(384)   # Sentence transformer dimension
    
    # Clean text
    text = text.strip()
    
    try:
        if USE_OPENAI_EMBEDDINGS and OPENAI_AVAILABLE and OPENAI_API_KEY:
            return _get_openai_embedding(text, model_name)
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            return _get_sentence_transformer_embedding(text, model_name)
        else:
            raise ValueError("No embedding backend available. Install sentence-transformers or set OPENAI_API_KEY")
    
    except Exception as e:
        logging.error(f"Embedding generation failed for text length {len(text)}: {e}")
        # Return fallback zero vector
        if USE_OPENAI_EMBEDDINGS:
            return np.zeros(1536)
        else:
            return np.zeros(384)


def _get_sentence_transformer_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding using Sentence Transformers (local model).
    
    Sentence Transformers are neural networks fine-tuned to produce semantically
    meaningful embeddings. The default model 'all-MiniLM-L6-v2' offers a good
    balance of speed, size, and quality:
    
    - Size: 22MB download
    - Speed: ~100 sentences/second on CPU
    - Quality: Good for most semantic similarity tasks
    - Dimensions: 384 (manageable for storage/computation)
    
    Alternative models:
    - 'all-mpnet-base-v2': Higher quality, slower, 768 dimensions
    - 'all-distilroberta-v1': Balanced performance, 768 dimensions
    - 'paraphrase-multilingual': Supports multiple languages
    """
    global _sentence_model
    
    # Lazy load model to avoid startup delays
    if _sentence_model is None:
        model_to_load = model_name or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        logging.info(f"Loading Sentence Transformer model: {model_to_load}")
        _sentence_model = SentenceTransformer(model_to_load)
    
    # Generate embedding
    embedding = _sentence_model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)  # Use float32 to save memory


def _get_openai_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embedding using OpenAI's embedding API.
    
    OpenAI's embeddings are created by large transformer models trained on
    diverse internet text. They typically provide higher quality semantic
    representations but come with costs and latency:
    
    - Quality: State-of-the-art semantic understanding
    - Speed: Network latency (100-500ms per request)
    - Cost: ~$0.0001 per 1K tokens (very affordable)
    - Dimensions: 1536 (requires more storage/computation)
    
    Best practices:
    - Batch multiple texts in single API call when possible
    - Cache embeddings to avoid re-computation
    - Consider rate limits for production use
    """
    global _openai_client
    
    # Initialize OpenAI client if needed
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    model_to_use = model_name or OPENAI_EMBEDDING_MODEL
    
    try:
        response = _openai_client.embeddings.create(
            model=model_to_use,
            input=text
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
        
    except Exception as e:
        logging.error(f"OpenAI embedding failed: {e}")
        raise


def embed_and_index(resume_texts: List[str], persist: bool = False, 
                   backend: str = "chromadb", collection_name: str = "resumes") -> Optional[Any]:
    """
    Generate embeddings for multiple texts and store in vector database.
    
    Vector databases enable efficient similarity search over large collections
    of documents. Instead of computing similarity against every document,
    vector DBs use indexing structures (like HNSW graphs) to find similar
    documents quickly.
    
    Why use vector databases:
    - Fast similarity search: O(log n) instead of O(n)
    - Scalability: Handle millions of documents efficiently  
    - Persistence: Store embeddings on disk for reuse
    - Filtering: Combine vector search with metadata filters
    - Analytics: Track search patterns and optimize
    
    Supported backends:
    - ChromaDB: Good for development, local storage, easy setup
    - FAISS: Fast in-memory search, good for batch processing
    - Pinecone: Cloud-hosted, production-ready, but requires subscription
    
    Args:
        resume_texts (List[str]): List of resume texts to embed and index
        persist (bool): Whether to save the index to disk
        backend (str): Vector database backend ("chromadb", "faiss")
        collection_name (str): Name for the document collection
        
    Returns:
        Optional[Any]: Vector database client/index object, or None if failed
        
    Example:
        texts = ["Software engineer with Python", "Data scientist with ML"]
        index = embed_and_index(texts, persist=True, backend="chromadb")
        
        # Later, search for similar texts
        query_embedding = get_embedding("Python developer")
        similar_docs = index.query(query_embedding, n_results=5)
    """
    if not resume_texts:
        logging.warning("No texts provided for embedding")
        return None
    
    logging.info(f"Embedding {len(resume_texts)} texts using {backend} backend")
    
    # Generate embeddings for all texts
    embeddings = []
    for i, text in enumerate(resume_texts):
        if i % 100 == 0:  # Progress logging
            logging.info(f"Embedding progress: {i}/{len(resume_texts)}")
        
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    # Store in chosen backend
    if backend.lower() == "chromadb":
        return _store_in_chromadb(resume_texts, embeddings, persist, collection_name)
    elif backend.lower() == "faiss":
        return _store_in_faiss(resume_texts, embeddings, persist, collection_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'chromadb' or 'faiss'")


def _store_in_chromadb(texts: List[str], embeddings: List[np.ndarray], 
                      persist: bool, collection_name: str) -> Optional[Any]:
    """
    Store embeddings in ChromaDB collection.
    
    ChromaDB is a lightweight vector database designed for AI applications.
    It provides:
    - Simple Python API
    - Local file storage or in-memory operation
    - Automatic embedding generation (we override with our own)
    - Metadata filtering and hybrid search
    """
    if not CHROMADB_AVAILABLE:
        logging.error("ChromaDB not available - install with: pip install chromadb")
        return None
    
    try:
        # Initialize ChromaDB client
        if persist:
            # Persistent storage
            db_path = Path("./chroma_db")
            db_path.mkdir(exist_ok=True)
            client = chromadb.PersistentClient(path=str(db_path))
        else:
            # In-memory storage
            client = chromadb.Client()
        
        # Create or get collection
        try:
            collection = client.get_collection(collection_name)
            logging.info(f"Using existing ChromaDB collection: {collection_name}")
        except:
            collection = client.create_collection(collection_name)
            logging.info(f"Created new ChromaDB collection: {collection_name}")
        
        # Add documents with embeddings
        ids = [f"doc_{i}" for i in range(len(texts))]
        metadatas = [{"text_length": len(text), "index": i} for i, text in enumerate(texts)]
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logging.info(f"Stored {len(texts)} documents in ChromaDB collection '{collection_name}'")
        return collection
        
    except Exception as e:
        logging.error(f"ChromaDB storage failed: {e}")
        return None


def _store_in_faiss(texts: List[str], embeddings: List[np.ndarray], 
                   persist: bool, collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Store embeddings in FAISS index.
    
    FAISS (Facebook AI Similarity Search) is a highly optimized library for
    efficient similarity search and clustering of dense vectors. It's particularly
    good for:
    - Large-scale similarity search (millions+ vectors)
    - High-performance batch processing
    - Memory-efficient operations
    - Advanced indexing algorithms (IVF, HNSW, etc.)
    
    Trade-offs vs ChromaDB:
    - Faster search performance
    - More memory efficient
    - Less convenient API (no automatic metadata handling)
    - Primarily in-memory (persistence requires manual save/load)
    """
    if not FAISS_AVAILABLE:
        logging.error("FAISS not available - install with: pip install faiss-cpu")
        return None
    
    try:
        # Convert embeddings to numpy array
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        dimension = embedding_matrix.shape[1]
        
        # Create FAISS index
        # Using IndexFlatIP for cosine similarity (inner product with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Add embeddings to index
        index.add(embedding_matrix)
        
        # Create metadata storage
        metadata = {
            'texts': texts,
            'dimension': dimension,
            'size': len(texts),
            'collection_name': collection_name
        }
        
        # Persist if requested
        if persist:
            index_path = f"./faiss_{collection_name}.index"
            metadata_path = f"./faiss_{collection_name}_metadata.pkl"
            
            faiss.write_index(index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logging.info(f"Saved FAISS index to {index_path}")
        
        result = {
            'index': index,
            'metadata': metadata,
            'embedding_matrix': embedding_matrix
        }
        
        logging.info(f"Created FAISS index with {len(texts)} documents, dimension {dimension}")
        return result
        
    except Exception as e:
        logging.error(f"FAISS storage failed: {e}")
        return None


def search_similar_texts(query_text: str, index_or_collection: Any, 
                        top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Search for texts similar to query in the vector index.
    
    This function demonstrates how vector similarity search works:
    1. Convert query text to embedding vector
    2. Compute similarity against all stored embeddings
    3. Return top-k most similar documents with scores
    
    Args:
        query_text (str): Text to find similar documents for
        index_or_collection: ChromaDB collection or FAISS index object
        top_k (int): Number of similar documents to return
        threshold (float): Minimum similarity score to include
        
    Returns:
        List[Dict]: Similar documents with scores and metadata
    """
    query_embedding = get_embedding(query_text)
    
    try:
        # Handle ChromaDB collection
        if hasattr(index_or_collection, 'query'):
            results = index_or_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            similar_texts = []
            for i in range(len(results['documents'][0])):
                score = results['distances'][0][i] if 'distances' in results else 1.0
                if score >= threshold:
                    similar_texts.append({
                        'text': results['documents'][0][i],
                        'score': score,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    })
            
            return similar_texts
        
        # Handle FAISS index
        elif isinstance(index_or_collection, dict) and 'index' in index_or_collection:
            index = index_or_collection['index']
            metadata = index_or_collection['metadata']
            
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = index.search(query_embedding, top_k)
            
            similar_texts = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = scores[0][i]
                
                if score >= threshold and idx < len(metadata['texts']):
                    similar_texts.append({
                        'text': metadata['texts'][idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return similar_texts
        
        else:
            logging.error("Unsupported index type for similarity search")
            return []
            
    except Exception as e:
        logging.error(f"Similarity search failed: {e}")
        return []


def get_embedding_info() -> Dict[str, Any]:
    """
    Get information about current embedding configuration.
    
    Returns:
        Dict: Configuration info including model, dimensions, backend status
    """
    info = {
        'use_openai': USE_OPENAI_EMBEDDINGS,
        'openai_available': OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
        'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
        'chromadb_available': CHROMADB_AVAILABLE,
        'faiss_available': FAISS_AVAILABLE,
        'default_model': OPENAI_EMBEDDING_MODEL if USE_OPENAI_EMBEDDINGS else DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        'expected_dimensions': 1536 if USE_OPENAI_EMBEDDINGS else 384
    }
    
    return info


# Utility functions for common operations
def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts.
    
    Args:
        text1, text2: Texts to compare
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logging.error(f"Similarity computation failed: {e}")
        return 0.0
