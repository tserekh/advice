"""
Prepare and index Q&A data for SochiGPT RAG system
"""

import glob
import os
import pandas as pd

import config
from advice.marker import add_question_mark, get_reply_mapping
from advice.reader import resave_data
from advice.tokenizers import tokenize
from rag_engine import RAGEngine


def prepare_legacy_data():
    """Prepare data using the legacy pipeline (for backward compatibility)"""
    folders = glob.glob(f"{config.chats_path}/*")

    for i, folder in enumerate(folders):
        resave_data(i, folder)

    if os.path.exists(config.question_reply_path):
        question_reply = pd.read_csv(
            config.question_reply_path, encoding="utf-8", sep="\t"
        )
    else:
        question_reply = pd.DataFrame()
        folders = glob.glob(config.chats_path)
        for path in glob.glob(f"{config.messages_path}/*"):
            chat = pd.read_csv(path, sep="\t", encoding="utf-8")
            chat = add_question_mark(chat)
            question_reply = question_reply.append(get_reply_mapping(chat))

    question_reply["question_tokens"] = question_reply["question_message"].apply(
        tokenize
    )
    (
        question_reply.drop_duplicates(
            ["question_message", "question_message_id"]
        ).to_csv(config.question_reply_path, index=False, encoding="utf-8", sep="\t")
    )
    
    return question_reply


def index_for_rag(question_reply_df=None):
    """Index Q&A pairs into the vector store for RAG"""
    print("Initializing RAG Engine for indexing...")
    rag_engine = RAGEngine()
    
    # Load or use provided dataframe
    if question_reply_df is None:
        if not os.path.exists(config.question_reply_path):
            print(f"Error: {config.question_reply_path} not found.")
            print("Please run legacy preparation first or provide Q&A data.")
            return False
        
        question_reply_df = pd.read_csv(
            config.question_reply_path, encoding="utf-8", sep="\t"
        )
    
    # Prepare QA pairs for indexing
    qa_pairs = []
    for idx, row in question_reply_df.iterrows():
        question = row.get('question_message', '')
        answer = row.get('reply_message', '')
        
        if question and answer:
            qa_pairs.append({
                'id': f"qa_{idx}",
                'question': str(question),
                'answer': str(answer),
                'source': 'telegram_chat'
            })
    
    if not qa_pairs:
        print("No valid Q&A pairs found to index.")
        return False
    
    print(f"Found {len(qa_pairs)} Q&A pairs to index.")
    
    # Index documents
    rag_engine.index_documents(qa_pairs)
    
    print(f"Successfully indexed {len(qa_pairs)} documents into vector store.")
    print(f"Vector store location: {config.vector_db_path}")
    
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("SochiGPT Data Preparation & Indexing")
    print("=" * 50)
    
    # Step 1: Prepare legacy data (if needed)
    print("\nStep 1: Preparing Q&A data...")
    if os.path.exists(config.question_reply_path):
        print(f"Found existing {config.question_reply_path}, skipping legacy preparation.")
        question_reply = pd.read_csv(
            config.question_reply_path, encoding="utf-8", sep="\t"
        )
    else:
        print("Running legacy data preparation...")
        question_reply = prepare_legacy_data()
    
    # Step 2: Index for RAG
    print("\nStep 2: Indexing documents for RAG...")
    success = index_for_rag(question_reply)
    
    if success:
        print("\n" + "=" * 50)
        print("Preparation complete! You can now run app.py")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Preparation failed. Please check the errors above.")
        print("=" * 50)
