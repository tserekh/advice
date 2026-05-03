import re
from typing import List


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = ""

    lowered = text.lower()
    tokens = re.findall(r"\w+", lowered, flags=re.UNICODE)

    # normalize: unique tokens, drop empty
    uniq = {t for t in tokens if t}
    return list(uniq)
