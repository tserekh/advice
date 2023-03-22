import glob
import os

import pandas as pd

import config
from advice.marker import add_question_mark, get_reply_mapping
from advice.reader import resave_data
from advice.tokenizers import tokenize

if __name__ == "__main__":
    folders = glob.glob(f"{config.chats_path}/*")

    for i, folder in enumerate(folders):
        resave_data(i, folder)

    if os.path.exists(config.question_reply_path):
        question_reply = pd.read_csv(
            config.question_reply_path, encoding="utf-8", sep="\t"
        )
    else:
        question_reply = pd.DataFrame()
        folders = glob.glob(config.chats_path)
        for path in glob.glob(f"{config.messages_path}/*"):
            chat = pd.read_csv(path, sep="\t", encoding="utf-8")
            chat = add_question_mark(chat)
            question_reply = question_reply.append(get_reply_mapping(chat))

    question_reply["question_tokens"] = question_reply["question_message"].apply(
        tokenize
    )
    (
        question_reply.drop_duplicates(
            ["question_message", "question_message_id"]
        ).to_csv(config.question_reply_path, index=False, encoding="utf-8", sep="\t")
    )
