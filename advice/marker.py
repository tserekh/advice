import pandas as pd
import io
from typing import List
import numpy as np
def add_question_mark(df: pd.DataFrame)-> pd.DataFrame:
    target = "question"
    target_list = []
    for message in df["message"].values:
        target_list.append(int("посовет" in message))
    df[target] = target_list
    return df

def get_reply_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """ "message", "question", "go_to_message_id", "message_id"]] """
    df_question = (
        df[df["question"]==1]
        .rename(columns={"message": "question_message", "message_id": "question_message_id"})
        [["question_message", "question_message_id"]]
    )
    df_reply = (
        df[df["go_to_message_id"].notnull()]
        .rename(
            columns={
                "message": "reply_message",
                "message_id": "reply_message_id",
                "go_to_message_id": "question_message_id"
            }
        )[["reply_message", "reply_message_id", "question_message_id"]]
    )
    return pd.merge(df_question, df_reply, on="question_message_id")

class Similarity:
    pass

class FastText(Similarity):
    def load_vectors(self, vectors_path: str, use_tokens: List[str]):
        fin = io.open(vectors_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in use_tokens:
                data[tokens[0]] = list(map(float, tokens[1:]))
        self.embeddings = data

    def __init__(self, vectors_name: str,  use_tokens: List[str]):
        self.vectors_name = vectors_name
        self.use_tokens = use_tokens
        self.load_vectors(self.vectors_name, self.use_tokens)

    def tokenizer(self):
        pass

    def cosin(self, tokens1: List[str], tokens2: List[str]):
        question_embedding = np.zeros(300)
        message_embedding = np.zeros(300)
        for token in tokens1:
            if token in self.embeddings:
                question_embedding += np.array(self.embeddings[token])
        for token in tokens2:
            if token in self.embeddings:
                message_embedding += np.array(self.embeddings[token])
        cos_sim = np.dot(question_embedding, message_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(message_embedding))
        return cos_sim
