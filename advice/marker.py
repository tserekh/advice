import io
import json
from typing import Optional, List, Dict, Iterable, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import telebot
import fasttext
import config
from advice.tokenizers import tokenize


def get_question_score(tokens: Iterable[str], coefs: Dict[str, float]) -> float:
    score = 0
    for token in tokens:
        if token in coefs:
            score += coefs[token]
    return score


def mark_question(text: str, coefs: Dict[str, float]) -> int:
    text = text.lower()
    mark = False
    for word_marker in config.word_markers:
        mark |= word_marker in text
    return mark


def single_message_mark(message: telebot.types.Message) -> int:
    #     with open("data/models/coefs.json") as f:
    #         coefs = json.loads(f.read())
    return mark_question(message.text, {})


def add_question_mark(data: pd.DataFrame) -> pd.DataFrame:
    target_name = "question"
    # with open(config.mark_model_coefs_path) as f:
    #     coefs = json.loads(f.read())
    data[target_name] = data["message"].apply(lambda x: mark_question(x, {}))
    return data


def get_reply_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """ "message", "question", "go_to_message_id", "message_id"]]"""
    df_question = df[df["question"] == 1].rename(
        columns={"message": "question_message", "message_id": "question_message_id"}
    )[["question_message", "question_message_id"]]
    df_question["question_message_id"] = df_question["question_message_id"].astype(str)
    df["go_to_message_id"] = df["go_to_message_id"].astype(str)
    df_reply = df[df["go_to_message_id"].notnull()].rename(
        columns={
            "message": "reply_message",
            "message_id": "reply_message_id",
            "go_to_message_id": "question_message_id",
        }
    )[["reply_message", "reply_message_id", "question_message_id"]]
    return pd.merge(df_question, df_reply, on="question_message_id")


class FastTextOld:
    def load_vectors(self, vectors_path: str, use_tokens: Optional[Set[str]]):
        if not use_tokens:
            use_tokens = set()
        fin = io.open(
            vectors_path, "r", encoding="utf-8", newline="\n", errors="ignore"
        )
        data = {}
        for line in tqdm(fin):
            tokens = line.rstrip().split(" ")
            if use_tokens and (tokens[0] in use_tokens):
                data[tokens[0]] = list(map(float, tokens[1:]))
        return data

    def __init__(self, vectors_name: str, use_tokens: Optional[Set[str]] = None):
        self.vectors_name = vectors_name
        if use_tokens:
            self.use_tokens = use_tokens
        else:
            self.use_tokens = set()
        self.embeddings = self.load_vectors(self.vectors_name, self.use_tokens)

    def cosin(self, tokens1: List[str], tokens2: List[str]) -> float:
        question_embedding = np.zeros(300)
        message_embedding = np.zeros(300)
        for token in tokens1:
            if token in self.embeddings:
                question_embedding += np.array(self.embeddings[token])
        for token in tokens2:
            if token in self.embeddings:
                message_embedding += np.array(self.embeddings[token])
        cos_sim = np.dot(question_embedding, message_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(message_embedding)
        )
        if cos_sim:
            return cos_sim
        else:
            return 0.0


class FastText:
    def __init__(self, use_tokens: Optional[Set[str]] = None):
        if use_tokens:
            self.use_tokens = use_tokens
        else:
            self.use_tokens = set()
        self.ft = fasttext.load_model("/root/cc.ru.300.bin")

    def cosin(self, tokens1: List[str], tokens2: List[str]) -> float:
        question_embedding = np.zeros(300)
        message_embedding = np.zeros(300)
        for token in tokens1:
            question_embedding += np.array(self.ft.get_word_vector(token))
        for token in tokens2:
            message_embedding += np.array(self.ft.get_word_vector(token))
        cos_sim = np.dot(question_embedding, message_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(message_embedding)
        )
        if cos_sim:
            return cos_sim
        else:
            return 0.0


def get_locations(message: str) -> set:
    locations = set()
    for location in config.locations:
        if location in message.lower():
            locations.add(location)
    if locations:
        return locations
    else:
        return {"no_location"}
