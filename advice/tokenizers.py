from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List

def tokenize(text: str) -> List[str]:
    stop_words = ['за', 'там', 'так', 'то', 'все', 'если', '-',
                  ':', 'но', 'как', 'по', 'у', 'есть', 'это', 'я', 'с', 'а', 'не', ')', 'и', '.', 'в', ',']
    tokens = word_tokenize(text.lower(), language="russian")
    tokens = list(filter(lambda x: 'посовет' not in x, tokens))
    tokens = list(filter(lambda x: x not in stop_words, tokens))

    return list(set(tokens))

def mapping_tokenizer()