from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List


def tokenize(text: str) -> List[str]:
    tokens = word_tokenize(text.lower(), language="russian")
    return list(set(tokens))
