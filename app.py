import ast

import pandas as pd
import telebot
import logging

import config
from advice.marker import FastText
from advice.marker import single_message_mark
from advice.tokenizers import tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.log_path, mode="w+", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

question_reply = pd.read_csv(config.question_reply_path, sep="\t", encoding="utf-8")
question_reply["question_tokens"] = question_reply["question_tokens"].apply(
    ast.literal_eval
)
all_tokens = []
for question_tokens in question_reply.drop_duplicates("question_message")[
    "question_tokens"
].values:
    all_tokens += question_tokens
vc = pd.Series(all_tokens).value_counts()
use_tokens = set(vc.iloc[50:].index)
fast_text = FastText(config.vectors_path, set(use_tokens))
with open(config.token_path) as f:
    BOT_TOKEN = f.read()
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(func=single_message_mark)
def handle_message(message):
    logger.info(f"Received message: {message.text}")
    tokens = tokenize(message.text)
    question_reply["cosin"] = (
        question_reply["question_tokens"]
        .apply(lambda tokens2: fast_text.cosin(tokens, tokens2))
        .fillna(0)
    )
    df_notnull = question_reply.sort_values("cosin")
    if len(df_notnull) > 0:
        row = df_notnull.iloc[-1]
        cosin = row["cosin"]
        reply_message = row["reply_message"]
        question_message = row["question_message"]
        reply = f"{reply_message}\nпохожесть: {cosin}\nВопрос: {question_message}"
        if cosin < config.min_cosin:
            reply = f"не уверен, но:\n{reply}"
    else:
        reply = "не знаю"
    bot.reply_to(message, reply)
    logger.info(f"Sent reply: {reply}")


try:
    bot.polling()
except Exception as e:
    logger.exception(e)
