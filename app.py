from advice.reader import read_chats_data
from advice.marker import add_question_mark, get_reply_mapping
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from advice.marker import FastText
import telebot
import os
from logging import getLogger
import pickle

logger = getLogger()
question_reply_path = "data/question_reply.csv"
def tokenize(text):
    tokens = word_tokenize(text.lower(), language="russian")
    return list(set(tokens))

if os.path.exists(question_reply_path):
    question_reply = pd.read_csv(question_reply_path)
else:
    question_reply = pd.DataFrame()
    chats = read_chats_data()
    for chat in chats:
        chat_w_mark = add_question_mark(chat)
        question_reply = question_reply.append(get_reply_mapping(chat_w_mark))
    questions_w_replies = question_reply[['question_message', 'question_message_id']].drop_duplicates()
    question_reply["question_tokens"] = question_reply['question_message'].apply(tokenize)
    question_reply.to_csv(question_reply_path, index=False)
all_tokens = []
for question_tokens in question_reply.drop_duplicates('question_message')['question_tokens'].values:
    all_tokens += question_tokens

vc = pd.Series(all_tokens).value_counts()
use_tokens = set(vc.iloc[50:].index)
# if os.path.exists("data/fast_text.pkl"):
#     fast_text = pickle.load("data/fast_text.pkl")
# else:
#
#     pickle.dump(fast_text, "data/fast_text.pkl")
fast_text = FastText("D:/Downloads/cc.ru.300.vec/cc.ru.300.vec", use_tokens)


with open("C:/Users/artem/Documents/sochi.txt") as f:
    TOKEN = f.read()

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(func=lambda message: "посовет" in message.text.lower())
def handle_message(message):
    tokens = tokenize(message.text)
    question_reply['cosin'] = question_reply['question_tokens'].apply(lambda tokens2: fast_text.cosin(tokens, tokens2))
    row = question_reply[question_reply['cosin'].notnull()].sort_values('cosin').iloc[-1]
    reply_message = row['reply_message']
    cosin = row['cosin']
    bot.reply_to(message, f"{reply_message}\nпохожесть:{cosin}")
print("start bot")
# Запуск бота
bot.polling()