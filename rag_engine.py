"""
RAG Engine for SochiGPT
Handles embedding generation, vector storage, and retrieval
"""

import os
from typing import List, Dict, Tuple, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import config

class EmbeddingModel:
    """Lightweight multilingual embedding model"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        print(f"Loading embedding model: {self.model_name}")
        
        try:
            # Модель загрузится через прокси, указанную в переменных окружения
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=False
            )
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
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
        self._messages_df_cache: Dict[str, pd.DataFrame] = {}
        print("RAG Engine initialized")

    def _load_messages_shard_df(self, shard: str) -> Optional[pd.DataFrame]:
        shard = (shard or "").strip()
        if not shard:
            return None
        if shard in self._messages_df_cache:
            return self._messages_df_cache[shard]
        path = os.path.join(config.messages_path, f"{shard}.tsv")
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path, sep="\t", encoding="utf-8").reset_index(drop=True)
        self._messages_df_cache[shard] = df
        return df

    def _neighbor_window_text(
        self,
        shard: str,
        chat_name: str,
        message_id: str,
        before: int,
        after: int,
    ) -> str:
        df = self._load_messages_shard_df(shard)
        if df is None or df.empty:
            return ""

        mid = str(message_id).strip()
        if not mid:
            return ""

        m = df["message_id"].astype(str) == mid
        cn = str(chat_name or "").strip()
        if cn and "chat_name" in df.columns:
            m &= df["chat_name"].astype(str) == cn

        hit_rows = df.loc[m]
        if hit_rows.empty:
            return ""

        pos = int(hit_rows.index[0])
        lo = max(0, pos - max(0, before))
        hi = min(len(df), pos + max(0, after) + 1)

        lines: List[str] = []
        for i in range(lo, hi):
            row = df.iloc[i]
            rel = i - pos
            fn = str(row.get("from_name", "") or "").strip()
            txt = str(row.get("message", "") or "").strip()
            if not txt:
                continue
            tag = f"[{rel:+d}]"
            who = f"{fn}: " if fn else ""
            lines.append(f"{tag} {who}{txt}")
        return "\n".join(lines)

    def expand_documents_with_neighbors(self, docs: List[Dict]) -> List[Dict]:
        before = int(getattr(config, "neighbor_messages_before", 0))
        after = int(getattr(config, "neighbor_messages_after", 0))
        if before <= 0 and after <= 0:
            return docs

        out: List[Dict] = []
        for d in docs:
            shard = str(d.get("messages_shard") or "").strip()
            mid = str(d.get("message_id") or "").strip()
            chat_name = str(d.get("chat_name") or "").strip()
            if not shard or not mid:
                out.append(d)
                continue
            expanded = self._neighbor_window_text(shard, chat_name, mid, before, after)
            if not expanded:
                out.append(d)
                continue
            nd = dict(d)
            nd["document"] = expanded
            nd["answer"] = expanded
            out.append(nd)
        return out
    
    def index_documents(self, items: List[Dict]):
        """
        Index documents into the vector store.
        
        Args:
            items:
              - Q&A style: {'id','question','answer','source'?}
              - Text style: {'id','text','metadata'?,'source'?}
        """
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(items):
            if "text" in item:
                text = str(item.get("text", ""))
                doc_text = text
                metadata = dict(item.get("metadata") or {})
                # keep downstream shape: app expects question/answer fields in metadata
                metadata.setdefault("question", "")
                metadata.setdefault("answer", text)
                metadata["source"] = item.get("source", metadata.get("source", "unknown"))
            else:
                question = str(item.get("question", ""))
                answer = str(item.get("answer", ""))
                doc_text = f"{question}\n{answer}".strip()
                metadata = {
                    "question": question,
                    "answer": answer,
                    "source": item.get("source", "unknown"),
                }

            if not doc_text:
                continue

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(item.get("id", f"doc_{i}"))
        
        # Add to vector store (with embeddings generated internally)
        embeddings = self.embedding_model.encode(documents)
        
        # Clear existing collection if needed
        try:
            self.vector_store.collection.delete(where={})
        except:
            pass

        # Chroma has a max batch size (varies by version). Chunk to stay below it.
        max_batch_size = 5000
        for start in range(0, len(documents), max_batch_size):
            end = min(start + max_batch_size, len(documents))
            batch_embeddings = embeddings[start:end]
            batch_documents = documents[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = ids[start:end]

            self.vector_store.collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids,
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
                    'source': metadata.get('source', 'unknown'),
                    'messages_shard': str(metadata.get('messages_shard') or ''),
                    'chat_name': str(metadata.get('chat_name') or ''),
                    'message_id': str(metadata.get('message_id') or ''),
                })

        return self.expand_documents_with_neighbors(filtered_results)
    
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
