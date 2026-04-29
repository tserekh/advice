"""
Prepare and index Q&A data for SochiGPT RAG system
"""

import glob
import os
import pandas as pd
from pathlib import Path

import config
from advice.marker import add_question_mark, get_reply_mapping
from advice.reader import resave_data
from advice.tokenizers import tokenize
from rag_engine import RAGEngine


def check_local_chat_exports():
    """Check if chat export data exists in the local data folder"""
    data_dir = Path(config.chats_path)
    
    # Check if ChatExport folders already exist
    existing_exports = list(data_dir.glob("ChatExport_*"))
    if existing_exports:
        print(f"Found {len(existing_exports)} ChatExport folders in {config.chats_path}:")
        for exp in existing_exports:
            print(f"  - {exp.name}")
        return True
    
    print(f"No ChatExport folders found in {config.chats_path}")
    print("Please manually copy your Telegram ChatExport folders to the 'data' directory.")
    print("Expected format: data/ChatExport_<chat_name>/")
    return False


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
        for path in glob.glob(f"{config.messages_path}/*"):
            chat = pd.read_csv(path, sep="\t", encoding="utf-8")
            chat = add_question_mark(chat)
            question_reply = question_reply.append(get_reply_mapping(chat))

    if len(question_reply) == 0 or "question_message" not in question_reply.columns:
        print("Warning: No Q&A data found from legacy sources.")
        return pd.DataFrame()

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
    
    # Step 0: Check for local ChatExport data
    print("\nStep 0: Checking for ChatExport data in local folder...")
    check_local_chat_exports()
    
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
        
        # Fallback to test data if legacy preparation returned empty
        if len(question_reply) == 0 and os.path.exists("data/qa_dataset.csv"):
            print("\nNo legacy data found. Loading test dataset from data/qa_dataset.csv...")
            question_reply = pd.read_csv("data/qa_dataset.csv")
            question_reply.to_csv(config.question_reply_path, index=False, encoding="utf-8", sep="\t")
            print(f"Saved {len(question_reply)} Q&A pairs to {config.question_reply_path}")
    
    # Check if we have any data to index
    if len(question_reply) == 0:
        print("\nError: No Q&A data available for indexing.")
        print("Please add your data to data/qa_dataset.csv or provide chat exports.")
        exit(1)
    
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
