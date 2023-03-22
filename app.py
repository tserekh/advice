from logging import getLogger

import pandas as pd
import telebot
from nltk.tokenize import word_tokenize

import config
from advice.marker import FastText
from advice.marker import single_message_mark

logger = getLogger()


def tokenize(text):
    tokens = word_tokenize(text.lower(), language="russian")
    return list(set(tokens))


if __name__ == "__main__":
    question_reply = pd.read_csv(config.question_reply_path, sep="\t", encoding="utf-8")
    all_tokens = []
    for question_tokens in question_reply.drop_duplicates("question_message")[
        "question_tokens"
    ].values:
        all_tokens += question_tokens
    vc = pd.Series(all_tokens).value_counts()
    use_tokens = set(vc.iloc[50:].index)
    fast_text = FastText(config.vectors_path, use_tokens)
    with open(config.token_path) as f:
        BOT_TOKEN = f.read()
    bot = telebot.TeleBot(BOT_TOKEN)

    @bot.message_handler(func=single_message_mark)
    def handle_message(message):
        tokens = tokenize(message.text)
        logger.info(f"Input message {message.text}")
        question_reply["cosin"] = question_reply["question_tokens"].apply(
            lambda tokens2: fast_text.cosin(tokens, tokens2)
        )
        df_notnull = question_reply[question_reply["cosin"].notnull()].sort_values(
            "cosin"
        )
        if len(df_notnull) > 0:
            row = df_notnull.iloc[-1]
        else:
            return
        reply_message = row["reply_message"]
        cosin = row["cosin"]
        question_message = row["question_message"]
        reply = f"{reply_message}\nпохожесть: {cosin}\nВопрос: {question_message}"
        logger.info(f"Reply {reply}")
        bot.reply_to(message, reply)

    bot.polling()
