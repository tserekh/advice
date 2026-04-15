#!/bin/bash
# Script to download a lightweight LLM model for SochiGPT

MODEL_DIR="models"
mkdir -p $MODEL_DIR

echo "Available models for weak servers:"
echo "1. Phi-2 (~2GB) - Smallest, fastest"
echo "2. TinyLlama-1.1B (~1GB) - Ultra-lightweight"
echo "3. Llama-2-7B-Chat Q4_K_M (~4GB) - Best quality/size ratio"
echo ""
read -p "Choose model (1-3, or press Enter for Llama-2-7B): " choice

case $choice in
    1)
        echo "Downloading Phi-2..."
        wget -O $MODEL_DIR/phi-2.Q4_K_M.gguf \
            "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
        ;;
    2)
        echo "Downloading TinyLlama-1.1B..."
        wget -O $MODEL_DIR/tinyllama-1.1b-chat.Q4_K_M.gguf \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ;;
    3|*)
        echo "Downloading Llama-2-7B-Chat..."
        wget -O $MODEL_DIR/llama-2-7b-chat.Q4_K_M.gguf \
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        ;;
esac

echo ""
echo "Model downloaded to $MODEL_DIR/"
echo "Update config.py if you chose a different model:"
echo '  model_path = "models/your-model-name.gguf"'
