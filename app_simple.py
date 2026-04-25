#!/usr/bin/env python3
"""
SochiGPT Bot with Simple RAG (Low-Resource Mode)
Works with TF-IDF embeddings when ML models are not available
"""

import os
import sys
import telebot
from typing import List, Dict, Optional
import pickle

import config

# Import simple RAG components
try:
    from prepare_simple import SimpleEmbeddingModel, SimpleVectorStore
    HAS_SIMPLE_RAG = True
except ImportError:
    HAS_SIMPLE_RAG = False
    print("Warning: Simple RAG components not available")


class SimpleRAGBot:
    """Telegram bot with simple RAG capabilities"""
    
    def __init__(self, token: str):
        self.token = token
        self.bot = telebot.TeleBot(token)
        
        # Load RAG components
        if HAS_SIMPLE_RAG:
            self.embedder = SimpleEmbeddingModel()
            self.vector_store = SimpleVectorStore(config.vector_db_path)
            
            # Need to rebuild embedder vocab from cached data
            if self.vector_store.documents:
                # Extract questions from documents for vocab rebuilding
                questions = []
                for doc in self.vector_store.documents:
                    # Documents are stored as "question answer"
                    questions.append(doc.split('.')[0])
                self.embedder.fit(questions)
                print(f"RAG initialized with {len(self.vector_store.documents)} documents")
            else:
                print("Warning: No documents in vector store")
        else:
            self.embedder = None
            self.vector_store = None
        
        # Register handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup message handlers"""
        
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            welcome_text = (
                "👋 Привет! Я SochiGPT - ваш помощник по Сочи.\n\n"
                "Я могу ответить на вопросы о:\n"
                "• Транспорт и как добраться\n"
                "• Рестораны и где покушать\n"
                "• Достопримечательности\n"
                "• Отели и жилье\n"
                "• Погода и развлечения\n\n"
                "Просто задайте мне вопрос!"
            )
            self.bot.reply_to(message, welcome_text)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            user_question = message.text
            
            if not user_question or len(user_question.strip()) < 3:
                self.bot.reply_to(message, "Пожалуйста, задайте более подробный вопрос.")
                return
            
            # Try to get answer from RAG
            answer = self.get_rag_answer(user_question)
            
            if answer:
                self.bot.reply_to(message, answer)
            else:
                fallback_text = (
                    "😕 К сожалению, я не нашел точного ответа на ваш вопрос.\n\n"
                    "Попробуйте перефразировать вопрос или задать его проще.\n"
                    "Например:\n"
                    "• Как добраться до Красной Поляны?\n"
                    "• Где можно покушать в Сочи?\n"
                    "• Сколько стоит билет в дендрарий?"
                )
                self.bot.reply_to(message, fallback_text)
    
    def get_rag_answer(self, question: str) -> Optional[str]:
        """Get answer using RAG retrieval"""
        if not self.embedder or not self.vector_store:
            return None
        
        try:
            # Encode query
            query_embedding = self.embedder.encode_single(question)
            
            # Retrieve top results
            results = self.vector_store.query(query_embedding, n_results=3)
            
            if not results['documents']:
                return None
            
            # Check similarity threshold
            min_distance = min(results['distances']) if results['distances'] else 1.0
            
            # Distance < 0.5 means good similarity (cosine similarity > 0.5)
            if min_distance > 0.6:
                print(f"Low similarity (distance={min_distance:.3f}), skipping answer")
                return None
            
            # Get best answer
            best_idx = results['distances'].index(min_distance)
            best_metadata = results['metadatas'][best_idx]
            best_answer = best_metadata.get('answer', '')
            
            print(f"Found answer with distance {min_distance:.3f}")
            return best_answer
            
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return None
    
    def run(self):
        """Start the bot"""
        print("=" * 50)
        print("🌴 SochiGPT Bot with Simple RAG")
        print("=" * 50)
        print(f"Vector store: {config.vector_db_path}")
        print(f"Documents loaded: {len(self.vector_store.documents) if self.vector_store else 0}")
        print("\nBot is running... Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            self.bot.infinity_polling()
        except KeyboardInterrupt:
            print("\nBot stopped by user")


def main():
    """Main entry point"""
    # Load token
    token_file = config.token_path
    
    if not os.path.exists(token_file):
        print(f"Error: Token file not found at {token_file}")
        print("Please create the file with your Telegram bot token.")
        sys.exit(1)
    
    with open(token_file, 'r', encoding='utf-8') as f:
        token = f.read().strip()
    
    if not token:
        print("Error: Token file is empty")
        sys.exit(1)
    
    # Create and run bot
    bot = SimpleRAGBot(token)
    bot.run()


if __name__ == "__main__":
    main()
