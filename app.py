"""
SochiGPT - RAG-based Telegram Bot
Retrieval-Augmented Generation for Q&A support
"""

import os
import time
import logging
from dotenv import load_dotenv
import telebot
from telebot import apihelper
from typing import Optional

import config
from rag_engine import RAGEngine
from llm_generator import LLMGenerator
from advice.tokenizers import tokenize
from advice.marker import mark_question

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize bot
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден. Пожалуйста, создайте файл .env и укажите токен.")

# Настройка прокси (если есть)
http_proxy = os.getenv("HTTP_PROXY")
https_proxy = os.getenv("HTTPS_PROXY")

if http_proxy or https_proxy:
    # Заменяем socks5:// на socks5h:// для DNS через прокси
    proxy_url = (http_proxy or https_proxy).replace('socks5://', 'socks5h://')
    logger.info(f"Использование прокси: {proxy_url}")
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    apihelper.proxy = proxies
    # Увеличиваем таймауты для нестабильного соединения
    apihelper.CONNECT_TIMEOUT = 60
    apihelper.READ_TIMEOUT = 120
else:
    logger.info("Прокси не настроен, работа без прокси.")

# Функция для запуска бота с ретраями
def run_bot_with_retries(max_retries=5, delay=5):
    """Запуск бота с повторными попытками при ошибках соединения"""
    attempt = 0
    while attempt < max_retries:
        try:
            logger.info(f"Попытка запуска бота #{attempt + 1}")
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Не удалось запустить бота после {max_retries} попыток: {e}")
                raise
            logger.warning(f"Ошибка соединения: {e}. Повторная попытка через {delay} секунд...")
            time.sleep(delay)

# Initialize bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize RAG components
print("Initializing RAG Engine...")
rag_engine = RAGEngine()

print("Initializing LLM Generator...")
llm_generator = LLMGenerator()

# Check if we need to index documents
def ensure_indexed():
    """Check if vector store has documents, index if empty"""
    if rag_engine.vector_store.get_count() == 0:
        print("Vector store is empty. Please run prepare.py to index documents.")
        return False
    return True


@bot.message_handler(func=lambda message: mark_question(message.text, {}))
def handle_message(message):
    """Handle user questions with RAG pipeline"""
    user_query = message.text.strip()
    
    # Check if indexed
    if not ensure_indexed():
        bot.reply_to(
            message,
            "База знаний пуста. Пожалуйста, запустите подготовку данных через prepare.py"
        )
        return
    
    # Retrieve relevant documents
    retrieved_docs = rag_engine.retrieve(user_query, top_k=config.top_k_retrieval)
    
    if not retrieved_docs:
        # No relevant documents found
        reply = "Извините, я не нашел подходящего ответа в базе знаний.\nПопробуйте переформулировать вопрос."
        bot.reply_to(message, reply)
        return
    
    # Generate response using LLM (or fallback to retrieval-only)
    if llm_generator.is_available():
        # Use LLM to generate a contextual response
        response = llm_generator.generate_rag_response(
            query=user_query,
            retrieved_docs=retrieved_docs,
            max_tokens=300
        )
    else:
        # Retrieval-only mode
        best_match = retrieved_docs[0]
        similarity = best_match['similarity']
        response = best_match['answer']
        
        # Add metadata for transparency
        if similarity < config.min_cosin:
            response = f"Не уверен, но:\n{response}"
        response += f"\n\nпохожесть: {similarity:.2f}\nВопрос: {best_match['question']}"
    
    bot.reply_to(message, response)


def main():
    """Start the bot"""
    print("=" * 50)
    print("SochiGPT RAG Bot Started")
    print(f"LLM Available: {llm_generator.is_available()}")
    print(f"Documents in vector store: {rag_engine.vector_store.get_count()}")
    print("=" * 50)
    
    # Start polling with retries
    run_bot_with_retries(max_retries=10, delay=5)


if __name__ == "__main__":
    main()
