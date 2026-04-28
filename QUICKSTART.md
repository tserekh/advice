# Быстрый старт SochiGPT RAG

## 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 2. Подготовка данных

```bash
python prepare.py
```

## 3. (Опционально) Скачать LLM модель

Для слабого сервера выберите легковесную модель:

```bash
./scripts/download_model.sh
```

Или вручную:
```bash
mkdir -p models
cd models
# Phi-2 (~2GB) - самый быстрый вариант
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
```

## 4. Запуск бота

```bash
python app.py
```

## Проверка работы

Бот должен вывести:
```
==================================================
SochiGPT RAG Bot Started
LLM Available: True/False
Documents in vector store: XXX
==================================================
```

- **LLM Available: True** - модель загружена, будет генерация ответов
- **LLM Available: False** - режим только retrieval (тоже работает!)
- **Documents > 0** - данные проиндексированы

## Настройка для слабого сервера

В `config.py` уменьшите параметры:

```python
llm_n_ctx = 1024      # было 2048
llm_n_threads = 1     # было 2
llm_n_batch = 256     # было 512
```

## Если нет данных для индексирования

Создайте тестовые данные:

```python
import pandas as pd

data = {
    'question_message': [
        'Какой отель выбрать в Сочи?',
        'Где поесть в Адлере?'
    ],
    'reply_message': [
        'Рекомендуем отель "Жемчужина" у моря',
        'Кафе "У моря" на набережной'
    ]
}

df = pd.DataFrame(data)
df.to_csv('data/question_reply.tsv', sep='\t', index=False)
```

Затем снова запустите `python prepare.py`
