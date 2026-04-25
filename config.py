# RAG Configuration for SochiGPT

# Paths
chats_path = "data/ChatExport*"
question_reply_path = "data/question_reply.tsv"
messages_path = "data/messages"
token_path = "credentials/sochi.txt"
vector_db_path = "data/vector_db"
model_path = "models/phi-2.Q4_K_M.gguf"

# Embedding model (lightweight for weak servers with limited disk space)
# Using a small multilingual model that fits in ~100MB
embedding_model_name = "intfloat/multilingual-e5-small"
# Alternative models if you have more resources:
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (~470MB, better quality)

# LLM settings
llm_n_ctx = 2048  # Context window size
llm_n_threads = 2  # CPU threads (adjust based on your server)
llm_n_batch = 512  # Batch size for inference

# RAG settings
top_k_retrieval = 3  # Number of documents to retrieve
min_similarity_threshold = 0.5  # Minimum similarity score for retrieval

# Legacy settings (for compatibility)
word_markers = ["посовет", "подскаж"]
max_questions_tokens = 35
min_question_score = 100.0
min_cosin = 0.85
