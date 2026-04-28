"""
Lightweight LLM Generator for SochiGPT
Uses llama-cpp-python for efficient CPU-based inference
"""

import os
from typing import Optional, List, Dict
from llama_cpp import Llama
import config


class LLMGenerator:
    """
    Lightweight LLM generator using quantized models
    Optimized for weak servers with CPU-only inference
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.model_path
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            print("Please download a quantized model (e.g., Llama-2-7B-chat Q4_K_M)")
            print("The bot will work in retrieval-only mode until model is available.")
            self.model = None
            return
        
        print(f"Loading LLM model: {self.model_path}")
        
        # Load model with optimized settings for weak servers
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=config.llm_n_ctx,
            n_threads=config.llm_n_threads,
            n_batch=config.llm_n_batch,
            n_gpu_layers=0,  # CPU-only mode
            verbose=False
        )
        
        print("LLM model loaded successfully")
    
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
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):
            context_parts.append(
                f"[Контекст {i}]\nВопрос: {doc['question']}\nОтвет: {doc['answer']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for chat model
        system_prompt = """Ты полезный ассистент, который отвечает на вопросы основываясь на предоставленном контексте.
Если в контексте нет нужной информации, скажи об этом честно.
Отвечай кратко и по делу на русском языке."""

        user_prompt = f"""{context}

Вопрос пользователя: {query}

Используя приведённый выше контекст, дай точный и полезный ответ на вопрос пользователя."""

        # Format for Llama-2 chat
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        
        # Generate response
        response = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop_sequences=["</s>", "[INST]", "Question:", "Вопрос:"]
        )
        
        if response:
            # Clean up response
            response = response.strip()
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
