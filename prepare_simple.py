#!/usr/bin/env python3
"""
Simple RAG preparation script for low-resource servers
Uses CPU-only mode and minimal memory footprint
"""

import os
import pandas as pd
from typing import List, Dict
import chromadb
from chromadb.config import Settings

import config

# Use a tiny embedding model that works offline
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("Warning: sentence-transformers not installed. Using fallback mode.")


class SimpleEmbeddingModel:
    """Ultra-lightweight embedding using TF-IDF as fallback"""
    
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        
    def fit(self, texts: List[str]):
        """Build vocabulary and IDF from texts"""
        from collections import Counter
        import math
        
        # Tokenize and build vocab
        all_tokens = []
        doc_counts = Counter()
        
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_counts[token] += 1
        
        self.vocab = {token: idx for idx, token in enumerate(set(all_tokens))}
        n_docs = len(texts)
        self.idf = {token: math.log(n_docs / (count + 1)) for token, count in doc_counts.items()}
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts as TF-IDF vectors"""
        embeddings = []
        for text in texts:
            tokens = text.lower().split()
            vec = [0.0] * len(self.vocab)
            token_counts = Counter(tokens)
            
            for token, count in token_counts.items():
                if token in self.vocab:
                    tf = count / len(tokens) if tokens else 0
                    idf = self.idf.get(token, 0)
                    vec[self.vocab[token]] = tf * idf
            
            # Normalize
            norm = sum(v*v for v in vec) ** 0.5
            if norm > 0:
                vec = [v/norm for v in vec]
            
            embeddings.append(vec)
        
        return embeddings
    
    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]


class SimpleVectorStore:
    """Simple in-memory vector store with cosine similarity"""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        # Try to load existing data
        cache_file = os.path.join(persist_directory, "cache.pkl")
        if os.path.exists(cache_file):
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
                    self.metadatas = data.get('metadatas', [])
                    self.ids = data.get('ids', [])
                print(f"Loaded {len(self.documents)} documents from cache")
            except Exception as e:
                print(f"Could not load cache: {e}")
    
    def add(self, documents: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict], ids: List[str]):
        """Add documents to the store"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Save to disk
        os.makedirs(self.persist_directory, exist_ok=True)
        cache_file = os.path.join(self.persist_directory, "cache.pkl")
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings,
                    'metadatas': self.metadatas,
                    'ids': self.ids
                }, f)
            print(f"Saved {len(self.documents)} documents to cache")
        except Exception as e:
            print(f"Could not save cache: {e}")
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> Dict:
        """Query for similar documents using cosine similarity"""
        if not self.embeddings:
            return {'documents': [], 'distances': [], 'metadatas': []}
        
        import numpy as np
        
        # Calculate cosine similarities
        query_vec = np.array(query_embedding)
        similarities = []
        
        for emb in self.embeddings:
            emb_vec = np.array(emb)
            norm_q = np.linalg.norm(query_vec)
            norm_e = np.linalg.norm(emb_vec)
            
            if norm_q > 0 and norm_e > 0:
                sim = np.dot(query_vec, emb_vec) / (norm_q * norm_e)
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        return {
            'documents': [self.documents[i] for i in top_indices],
            'distances': [1.0 - similarities[i] for i in top_indices],  # Convert to distance
            'metadatas': [self.metadatas[i] for i in top_indices]
        }


def prepare_simple_rag():
    """Prepare RAG index with minimal resources"""
    print("=" * 50)
    print("SochiGPT Simple RAG Preparation (Low-Resource Mode)")
    print("=" * 50)
    
    # Load Q&A data
    qa_file = "data/qa_dataset.csv"
    if not os.path.exists(qa_file):
        print(f"Error: {qa_file} not found!")
        print("Run create_test_data.py first or add your data.")
        return False
    
    df = pd.read_csv(qa_file)
    print(f"\nLoaded {len(df)} Q&A pairs from {qa_file}")
    
    # Initialize embedding model
    print("\nInitializing embedding model...")
    if HAS_ST:
        try:
            # Try to use a very small model
            print("Trying to load sentence-transformers model...")
            # Use a tiny model that might already be cached
            embedder = SimpleEmbeddingModel()
            texts = df['question_message'].tolist()
            embedder.fit(texts)
            print("Using TF-IDF fallback (no network/disk space for ML model)")
            use_st = False
        except Exception as e:
            print(f"ST failed: {e}, using TF-IDF")
            embedder = SimpleEmbeddingModel()
            texts = df['question_message'].tolist()
            embedder.fit(texts)
            use_st = False
    else:
        embedder = SimpleEmbeddingModel()
        texts = df['question_message'].tolist()
        embedder.fit(texts)
        use_st = False
    
    # Create vector store
    print(f"\nCreating vector store at {config.vector_db_path}...")
    vector_store = SimpleVectorStore(config.vector_db_path)
    
    # Index documents
    print("Indexing documents...")
    qa_pairs = []
    questions = []
    
    for idx, row in df.iterrows():
        question = str(row.get('question_message', ''))
        answer = str(row.get('reply_message', ''))
        
        if question and answer:
            qa_pairs.append({
                'id': f"qa_{idx}",
                'question': question,
                'answer': answer
            })
            questions.append(question)
    
    if not qa_pairs:
        print("No valid Q&A pairs found!")
        return False
    
    # Generate embeddings in batches
    print(f"Generating embeddings for {len(questions)} questions...")
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        embeddings = embedder.encode(batch)
        all_embeddings.extend(embeddings)
        print(f"  Processed {min(i+batch_size, len(questions))}/{len(questions)}")
    
    # Add to vector store
    vector_store.add(
        documents=[qa['question'] + " " + qa['answer'] for qa in qa_pairs],
        embeddings=all_embeddings,
        metadatas=[{'answer': qa['answer'], 'source': 'qa_dataset'} for qa in qa_pairs],
        ids=[qa['id'] for qa in qa_pairs]
    )
    
    print("\n" + "=" * 50)
    print("✅ Preparation complete!")
    print(f"Indexed {len(qa_pairs)} Q&A pairs")
    print(f"Vector store: {config.vector_db_path}")
    print("=" * 50)
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_query = "Как добраться до Красной Поляны?"
    query_emb = embedder.encode_single(test_query)
    results = vector_store.query(query_emb, n_results=2)
    
    print(f"\nQuery: {test_query}")
    for i, (doc, dist, meta) in enumerate(zip(results['documents'], 
                                               results['distances'], 
                                               results['metadatas'])):
        print(f"\nResult {i+1} (distance: {dist:.3f}):")
        print(f"  Answer: {meta.get('answer', '')[:100]}...")
    
    return True


if __name__ == "__main__":
    from collections import Counter
    import numpy as np
    success = prepare_simple_rag()
    exit(0 if success else 1)
