import io
import json
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from advice.tokenizers import tokenize


def get_question_score(tokens, coefs) -> float:
    score = 0
    for token in tokens:
        if token in coefs:
            score+=coefs[token]
    return score



def single_message_mark(message):
    with open("data/models/coefs.json") as f:
        coefs = json.loads(f.read())
        text = message.text.lower()
        tokens = tokenize(text)
        question_score = get_question_score(tokens, coefs)
        model_mark = (question_score>config.min_question_score) and (len(tokens)<config.max_questions_tokens)
        return int(("посовет" in text) or model_mark)


def add_question_mark(data: pd.DataFrame)-> pd.DataFrame:
    target_name = "question"
    with open("data/models/coefs.json") as f:
        coefs = json.loads(f.read())
    target_list = []
    for message in data["message"].values:
        tokens = tokenize(message)
        question_score = get_question_score(tokens, coefs)
        model_mark = (question_score>config.min_question_score) and (len(tokens)<config.max_questions_tokens)
        target = int(("посовет" in message) or model_mark)
        target_list.append(target)
    data[target_name] = target_list
    return data

def get_reply_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """ "message", "question", "go_to_message_id", "message_id"]] """
    df_question = (
        df[df["question"]==1]
        .rename(columns={"message": "question_message", "message_id": "question_message_id"})
        [["question_message", "question_message_id"]]

    )
    df_question["question_message_id"] = df_question["question_message_id"].astype(str)
    df["go_to_message_id"] = df["go_to_message_id"].astype(str)
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
    def load_vectors(self, vectors_path: str, use_tokens: Optional[List[str]]):
        fin = io.open(vectors_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        data = {}
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            if use_tokens and (tokens[0] in use_tokens):
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
        if cos_sim:
            return cos_sim
        else:
            return 0
