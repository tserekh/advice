"""
SochiGPT - RAG-based Telegram Bot
Retrieval-Augmented Generation for Q&A support
"""

import os
import time
import logging
import json
import re
from dotenv import load_dotenv
import telebot
from telebot import apihelper
from typing import Optional

import config
from rag_engine import RAGEngine
from llm_generator import LLMGenerator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG_LOG_PATH = "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log"

_RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"(?:(?:\+7|7|8)\s*[-(]?\s*\d{3}\s*[-)]?\s*\d{3}\s*[-]?\s*\d{2}\s*[-]?\s*\d{2})")
_RE_URL_CREDS = re.compile(r"(socks5h?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)


def _sanitize_text(text: str, limit: int = 12000) -> str:
    if not isinstance(text, str):
        return ""
    t = _RE_URL_CREDS.sub(r"\1<user>:<pass>@", text)
    t = _RE_EMAIL.sub("<email>", t)
    t = _RE_PHONE.sub("<phone>", t)
    # avoid logging bot token by accident
    bot_token = os.getenv("BOT_TOKEN")
    if bot_token and bot_token in t:
        t = t.replace(bot_token, "<bot_token>")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token in t:
        t = t.replace(hf_token, "<hf_token>")
    return t if len(t) <= limit else (t[:limit] + "\n<...truncated...>")


def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict):
    # region agent log
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": "6e8b83",
            "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Make logging failures visible during debugging
        try:
            print(f"[debug-log-write-failed] path={DEBUG_LOG_PATH}")
        except Exception:
            pass
    # endregion agent log

# Load environment variables from .env file
load_dotenv()

# Ensure HuggingFace libraries see the token (some read HF_TOKEN, others HUGGINGFACE_HUB_TOKEN)
_hf_token = os.getenv("HF_TOKEN")
if _hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = _hf_token
    os.environ["HF_TOKEN"] = _hf_token

# Initialize bot
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден. Пожалуйста, создайте файл .env и укажите токен.")

print(f"[debug] log_path={DEBUG_LOG_PATH}")
_dbg_log(
    "H0",
    "app.py:startup",
    "app started",
    {
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "has_hf_token": bool(os.getenv("HF_TOKEN")),
        "has_hf_hub_token": bool(os.getenv("HUGGINGFACE_HUB_TOKEN")),
        "config_file": getattr(config, "__file__", None),
        "model_repo": getattr(config, "model_repo", None),
        "model_filename": getattr(config, "model_filename", None),
        "model_path": getattr(config, "model_path", None),
    },
)

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


@bot.message_handler(content_types=["text"])
def handle_message(message):
    """Handle user questions with RAG pipeline"""
    if not getattr(message, "text", None):
        return

    user_query = message.text.strip()
    if not user_query:
        return

    # Ignore bot commands
    if user_query.startswith("/"):
        return
    _dbg_log(
        "H10",
        "app.py:handle_message:entry",
        "incoming user question",
        {"message_id": getattr(message, "message_id", None), "query": _sanitize_text(user_query, limit=2000)},
    )
    
    # Check if indexed
    if not ensure_indexed():
        bot.reply_to(
            message,
            "База знаний пуста. Пожалуйста, запустите подготовку данных через prepare.py"
        )
        return
    
    # Retrieve relevant documents
    retrieved_docs = rag_engine.retrieve(user_query, top_k=config.top_k_retrieval)

    _dbg_log(
        "H10",
        "app.py:handle_message:retrieved",
        "retrieved docs for query",
        {
            "top_k": config.top_k_retrieval,
            "results": [
                {
                    "similarity": float(d.get("similarity", 0.0)),
                    "question": _sanitize_text(d.get("question", "") or "", limit=4000),
                    "answer": _sanitize_text(d.get("answer", "") or "", limit=4000),
                    "source": d.get("source", ""),
                }
                for d in (retrieved_docs or [])
            ],
        },
    )
    
    if not retrieved_docs:
        # No relevant documents found
        reply = "Извините, я не нашел подходящего ответа в базе знаний.\nПопробуйте переформулировать вопрос."
        bot.reply_to(message, reply)
        return
    
    # Generate response using LLM (or fallback to retrieval-only)
    if llm_generator.is_available():
        # Use LLM to generate a contextual response
        _dbg_log(
            "H11",
            "app.py:handle_message:before_llm",
            "calling llm_generator.generate_rag_response",
            {"retrieved_docs_count": len(retrieved_docs), "max_tokens": 300},
        )
        response = llm_generator.generate_rag_response(
            query=user_query,
            retrieved_docs=retrieved_docs,
            max_tokens=300
        )
        _dbg_log(
            "H14",
            "app.py:handle_message:llm_prompt_snapshot",
            "snapshot of last prompt used by LLMGenerator",
            {
                "prompt_format": getattr(llm_generator, "last_prompt_format", None),
                "stop_sequences": getattr(llm_generator, "last_stop_sequences", None),
                "prompt_full": getattr(llm_generator, "last_prompt", None),
            },
        )
    else:
        # Retrieval-only mode
        best_match = retrieved_docs[0]
        similarity = best_match['similarity']
        response = best_match['answer']
        
        # Add metadata for transparency
        if similarity < config.min_cosin:
            response = f"Не уверен, но:\n{response}"
        response += f"\n\nпохожесть: {similarity:.2f}"
    
    _dbg_log(
        "H13",
        "app.py:handle_message:reply",
        "replying to user",
        {"message_id": getattr(message, "message_id", None), "response": _sanitize_text(response, limit=4000)},
    )
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
