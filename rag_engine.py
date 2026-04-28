"""
RAG Engine for SochiGPT
Handles embedding generation, vector storage, and retrieval
"""

import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import config
import httpx

# Временное отключение системных прокси для загрузки моделей
# Сохраняем оригинальные значения
_original_http_proxy = os.environ.get('HTTP_PROXY')
_original_https_proxy = os.environ.get('HTTPS_PROXY')

class EmbeddingModel:
    """Lightweight multilingual embedding model"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        print(f"Loading embedding model: {self.model_name}")
        
        # Временно очищаем переменные окружения прокси для скачивания модели
        if _original_http_proxy:
            os.environ.pop('HTTP_PROXY', None)
        if _original_https_proxy:
            os.environ.pop('HTTPS_PROXY', None)
        
        try:
            # Модель загрузится напрямую, так как мы временно очистили переменные окружения прокси
            # Дополнительно указываем trust_env=False, чтобы игнорировать любые системные прокси
            import httpx
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=False,
                transport_kwargs={"trust_env": False}
            )
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        finally:
            # Восстанавливаем прокси обратно
            if _original_http_proxy:
                os.environ['HTTP_PROXY'] = _original_http_proxy
            if _original_https_proxy:
                os.environ['HTTPS_PROXY'] = _original_https_proxy
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """Encode a single text"""
        return self.encode([text])[0]


class VectorStore:
    """ChromaDB vector store for efficient similarity search"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or config.vector_db_path
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="sochi_qa",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the vector store"""
        # ChromaDB can handle batching internally
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, Dict, float]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                # Convert distance to similarity (cosine distance to similarity)
                similarity = 1.0 - distance
                formatted_results.append((doc, metadata, similarity))
        
        return formatted_results
    
    def get_count(self) -> int:
        """Get total number of documents in the store"""
        return self.collection.count()


class RAGEngine:
    """Main RAG engine combining embedding and retrieval"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        print("RAG Engine initialized")
    
    def index_documents(self, qa_pairs: List[Dict]):
        """
        Index question-answer pairs into the vector store
        
        Args:
            qa_pairs: List of dicts with 'question', 'answer', and optional metadata
        """
        documents = []
        metadatas = []
        ids = []
        
        for i, qa in enumerate(qa_pairs):
            # Combine question and answer for better retrieval
            doc_text = f"{qa['question']}\n{qa['answer']}"
            documents.append(doc_text)
            
            metadata = {
                'question': qa['question'],
                'answer': qa['answer'],
                'source': qa.get('source', 'unknown')
            }
            metadatas.append(metadata)
            
            # Create unique ID
            doc_id = qa.get('id', f"doc_{i}")
            ids.append(doc_id)
        
        # Add to vector store (with embeddings generated internally)
        embeddings = self.embedding_model.encode(documents)
        
        # Clear existing collection if needed
        try:
            self.vector_store.collection.delete(where={})
        except:
            pass
        
        self.vector_store.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata and similarity scores
        """
        top_k = top_k or config.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Filter by similarity threshold
        filtered_results = []
        for doc, metadata, similarity in results:
            if similarity >= config.min_similarity_threshold:
                filtered_results.append({
                    'document': doc,
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'similarity': similarity,
                    'source': metadata.get('source', 'unknown')
                })
        
        return filtered_results
    
    def query(self, query: str, top_k: int = None) -> str:
        """
        Simple query that returns the best matching answer
        
        Args:
            query: User query
            top_k: Number of candidates to consider
            
        Returns:
            Best matching answer or fallback message
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "Извините, я не нашел подходящего ответа в базе знаний."
        
        # Return the best match
        best_result = results[0]
        return best_result['answer']
