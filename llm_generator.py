"""
Lightweight LLM Generator for SochiGPT
Uses llama-cpp-python for efficient CPU-based inference
"""

import os
import json
import time
import threading
import re
from typing import Optional, List, Dict
import config


class LLMGenerator:
    """
    Lightweight LLM generator using quantized models
    Optimized for weak servers with CPU-only inference
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.model_path
        self._lock = threading.Lock()
        self.last_prompt: Optional[str] = None
        self.last_prompt_format: Optional[str] = None
        self.last_stop_sequences: Optional[List[str]] = None

        # PII-safe sanitization for debug logs
        self._re_email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
        self._re_phone = re.compile(r"(?:(?:\+7|7|8)\s*[-(]?\s*\d{3}\s*[-)]?\s*\d{3}\s*[-]?\s*\d{2}\s*[-]?\s*\d{2})")
        self._re_url_creds = re.compile(r"(socks5h?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)
        # Model sometimes continues generation with fake RAG blocks mirroring the prompt.
        self._re_leaked_ctx_heading = re.compile(
            r"\r?\n\s*\[(?:Контекст|Фрагмент)\s+\d+\]"
        )

        def _sanitize_text(text: str, limit: int = 20000) -> str:
            if not isinstance(text, str):
                return ""
            t = self._re_url_creds.sub(r"\1<user>:<pass>@", text)
            t = self._re_email.sub("<email>", t)
            t = self._re_phone.sub("<phone>", t)
            bot_token = os.getenv("BOT_TOKEN")
            if bot_token and bot_token in t:
                t = t.replace(bot_token, "<bot_token>")
            hf_token = os.getenv("HF_TOKEN")
            if hf_token and hf_token in t:
                t = t.replace(hf_token, "<hf_token>")
            return t if len(t) <= limit else (t[:limit] + "\n<...truncated...>")

        self._sanitize_text = _sanitize_text
        self._system_prompt = None

        # region agent log
        def _dbg_log(hypothesis_id: str, location: str, message: str, data: Dict):
            try:
                payload = {
                    "sessionId": "6e8b83",
                    "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                }
                with open(
                    "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            except Exception:
                pass

        # endregion agent log
        
        def _ensure_model_present():
            # Download GGUF from Hugging Face if missing and config provides repo+filename
            if os.path.exists(self.model_path):
                return True

            repo_id = getattr(config, "model_repo", None)
            filename = getattr(config, "model_filename", None)
            if not repo_id or not filename:
                return False

            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            _dbg_log(
                "H15",
                "llm_generator.py:LLMGenerator.__init__:download_start",
                "model file missing; starting download",
                {
                    "config_file": getattr(config, "__file__", None),
                    "model_path": self.model_path,
                    "repo_id": repo_id,
                    "filename": filename,
                    "config_model_path": getattr(config, "model_path", None),
                },
            )
            try:
                from huggingface_hub import hf_hub_download

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=os.path.dirname(self.model_path) or ".",
                    local_dir_use_symlinks=False,
                )

                # Ensure it ends up exactly at model_path (some hubs return absolute path in cache)
                if os.path.abspath(downloaded_path) != os.path.abspath(self.model_path):
                    try:
                        import shutil
                        shutil.copy2(downloaded_path, self.model_path)
                    except Exception:
                        pass

                _dbg_log(
                    "H15",
                    "llm_generator.py:LLMGenerator.__init__:download_ok",
                    "model download finished",
                    {"downloaded_path": downloaded_path, "final_path": self.model_path, "exists": os.path.exists(self.model_path)},
                )
                return os.path.exists(self.model_path)
            except Exception as e:
                _dbg_log(
                    "H16",
                    "llm_generator.py:LLMGenerator.__init__:download_error",
                    "model download failed",
                    {"error_type": type(e).__name__, "error": str(e)[:300]},
                )
                return False

        # Check if model exists
        if not _ensure_model_present():
            print(f"Warning: Model not found at {self.model_path}")
            print("Please download a quantized GGUF model or configure auto-download in config.py.")
            print("The bot will work in retrieval-only mode until model is available.")
            self.model = None
            return
        
        print(f"Loading LLM model: {self.model_path}")
        
        # Lazy import to avoid dependency issues when model is not available
        try:
            from llama_cpp import Llama

            # Hypothesis H1: large n_batch triggers GGML_ASSERT on some builds/models.
            # Clamp n_batch to safer values while we confirm via runtime evidence.
            n_ctx = int(getattr(config, "llm_n_ctx", 2048))
            n_threads = int(getattr(config, "llm_n_threads", 2))
            n_batch_cfg = int(getattr(config, "llm_n_batch", 512))
            n_batch = min(n_batch_cfg, 128)

            _dbg_log(
                "H1",
                "llm_generator.py:LLMGenerator.__init__:llama_params",
                "initializing llama with params",
                {
                    "model_path": self.model_path,
                    "n_ctx": n_ctx,
                    "n_threads": n_threads,
                    "n_batch_cfg": n_batch_cfg,
                    "n_batch_effective": n_batch,
                    "n_gpu_layers": 0,
                },
            )
            
            # Load model with optimized settings for weak servers
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                n_gpu_layers=0,  # CPU-only mode
                verbose=False
            )
            
            print("LLM model loaded successfully")
        except ImportError:
            print("Warning: llama-cpp-python not installed. Running in retrieval-only mode.")
            print("Install with: pip install llama-cpp-python")
            self.model = None
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            _dbg_log(
                "H2",
                "llm_generator.py:LLMGenerator.__init__:load_error",
                "error loading llama model",
                {"error_type": type(e).__name__, "error": str(e)[:300]},
            )
            self.model = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response given a prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that will stop generation
            
        Returns:
            Generated text
        """
        if self.model is None:
            return None
        
        try:
            # Hypothesis H2: concurrent generation triggers llama.cpp assert (not thread-safe).
            with self._lock:
            # region agent log
                try:
                    with open(
                        "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "6e8b83",
                                    "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
                                    "hypothesisId": "H3",
                                    "location": "llm_generator.py:LLMGenerator.generate:entry",
                                    "message": "starting generation (locked)",
                                    "data": {
                                        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
                                        "max_tokens": max_tokens,
                                        "temperature": temperature,
                                        "top_p": top_p,
                                    },
                                    "timestamp": int(time.time() * 1000),
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # endregion agent log
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences,
                    echo=False
                )
            
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    def _strip_leaked_rag_markers(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        m = self._re_leaked_ctx_heading.search(text)
        if m:
            text = text[: m.start()]
        return text.strip()

    @staticmethod
    def _rag_doc_body(doc: Dict) -> str:
        raw = doc.get("document")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        q = str(doc.get("question") or "").strip()
        a = str(doc.get("answer") or "").strip()
        parts = [x for x in (q, a) if x]
        return "\n".join(parts).strip()

    def generate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_tokens: int = 300
    ) -> str:
        """
        Generate a response based on retrieved documents
        
        Args:
            query: User's question
            retrieved_docs: List of retrieved documents with context
            max_tokens: Maximum tokens for the response
            
        Returns:
            Generated answer
        """
        if self.model is None:
            # Fallback to retrieval-only mode
            if retrieved_docs:
                return retrieved_docs[0]['answer']
            return "Извините, я не нашел подходящего ответа в базе знаний."
        
        # Build context from retrieved documents (indexed text or legacy Q&A merged into document)
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):
            body = self._rag_doc_body(doc)
            if body:
                context_parts.append(f"[Фрагмент {i}]\n{body}")
        context = "\n\n".join(context_parts)
        
        # Create prompt for chat model
        system_prompt = """Ты житель Сочи ассистент, который отвечает на вопросы основываясь на предоставленном контексте.
Если в контексте нет нужной информации, скажи об этом честно. Если подходит несколько ответов, то ответь несколько вариантов.
Отвечай, одним приложением и по делу на русском языке максимально кратко."""
        self._system_prompt = system_prompt

        user_prompt = f"""{context}

Запрос пользователя:
{query}

Используя приведённые выше фрагменты, дай точный и полезный ответ."""

        model_path_lower = (self.model_path or "").lower()
        # Stop if the model starts echoing retrieved-context markup (common Qwen/RAG leak).
        ctx_echo_stops = ["\n[Контекст", "\n[Фрагмент"]
        if "qwen2.5" in model_path_lower or "qwen" in model_path_lower:
            # Qwen Instruct models work best with ChatML.
            prompt_format = "qwen_chatml"
            prompt = (
                "system\n"
                f"{system_prompt}\n"
                "user\n"
                f"{context}\n\nЗапрос пользователя:\n{query}\n"
                "assistant\n"
                "Ответ:"
            )
            stop_sequences = ["\nsystem\n", "\nuser\n", "\nassistant\n"] + ctx_echo_stops
        elif "phi-2" in model_path_lower or "phi2" in model_path_lower:
            # Phi-2 is not a Llama2-chat model; avoid [INST]/<<SYS>> format to reduce prompt leakage.
            prompt_format = "phi2_instruct"
            prompt = (
                f"Инструкция:\n{system_prompt}\n\n"
                f"Фрагменты:\n{context}\n\n"
                f"Запрос пользователя:\n{query}\n"
                f"Ответ:"
            )
            stop_sequences = ["\nЗапрос пользователя:", "\nФрагменты:", "\nИнструкция:"] + ctx_echo_stops
        else:
            prompt_format = "llama2_chat_inst"
            # Format for Llama-2 chat
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            stop_sequences = ["</s>", "[INST]", "Question:", "\nЗапрос пользователя:"] + ctx_echo_stops

        # Save last prompt for debugging (sanitized)
        self.last_prompt_format = prompt_format
        self.last_stop_sequences = stop_sequences
        self.last_prompt = self._sanitize_text(prompt, limit=20000)

        # region agent log
        try:
            with open(
                "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "6e8b83",
                            "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
                            "hypothesisId": "H4",
                            "location": "llm_generator.py:LLMGenerator.generate_rag_response:prompt_built",
                            "message": "built prompt for LLM",
                            "data": {
                                "model_path": self.model_path,
                                "query_preview": query[:120],
                                "context_docs": len(retrieved_docs),
                                "prompt_format": prompt_format,
                                "prompt_chars": len(prompt),
                                "prompt_full": self.last_prompt,
                                "stop_sequences": stop_sequences,
                            },
                            "timestamp": int(time.time() * 1000),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion agent log
        
        # Generate response
        response = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop_sequences=stop_sequences
        )

        # region agent log
        try:
            with open(
                "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "6e8b83",
                            "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
                            "hypothesisId": "H5",
                            "location": "llm_generator.py:LLMGenerator.generate_rag_response:llm_result",
                            "message": "got LLM response",
                            "data": {
                                "response_is_none": response is None,
                                "response_chars": len(response) if isinstance(response, str) else None,
                                "response_full": self._sanitize_text(response, limit=20000) if isinstance(response, str) else None,
                            },
                            "timestamp": int(time.time() * 1000),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # endregion agent log
        
        if response:
            # Clean up response
            response = response.strip()

            # If model leaked the instruction/system prompt, strip it deterministically.
            if self._system_prompt and self._system_prompt.strip() and self._system_prompt.strip() in response:
                response = response.replace(self._system_prompt.strip(), "").strip()
                # Also drop leading empty lines
                response = response.lstrip()

            response = self._strip_leaked_rag_markers(response)

            # If answer is just a URL (common with RAG), add a short intro.
            # NOTE: use \S (not \\S) to match non-whitespace.
            if re.fullmatch(r"https?://\S+", response):
                response = f"Вот ссылка: {response}"

            # region agent log
            try:
                with open(
                    "/Users/hq-dj3wx03jt2/PycharmProjects/advice/.cursor/debug-6e8b83.log",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "6e8b83",
                                "runId": os.environ.get("CURSOR_DEBUG_RUN_ID", "pre-fix"),
                                "hypothesisId": "H12",
                                "location": "llm_generator.py:LLMGenerator.generate_rag_response:final_response",
                                "message": "final response after postprocess",
                                "data": {
                                    "final_chars": len(response),
                                    "final_text": self._sanitize_text(response, limit=20000),
                                },
                                "timestamp": int(time.time() * 1000),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # endregion agent log

            # Remove any trailing incomplete sentences
            if len(response) > 10 and not response.endswith(('.', '!', '?', '"', ')')):
                # Try to find a good stopping point
                last_period = response.rfind('.')
                if last_period > len(response) // 2:
                    response = response[:last_period + 1]
            return response
        
        # Fallback to best retrieved answer
        if retrieved_docs:
            return retrieved_docs[0]['answer']
        
        return "Извините, я не смог сгенерировать ответ."
    
    def is_available(self) -> bool:
        """Check if LLM model is loaded and available"""
        return self.model is not None
