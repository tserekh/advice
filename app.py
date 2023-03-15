import glob
import os
from logging import getLogger

import pandas as pd
import telebot
from nltk.tokenize import word_tokenize

import config
from advice.marker import FastText
from advice.marker import add_question_mark, get_reply_mapping
from advice.marker import single_message_mark

logger = getLogger()

def tokenize(text):
    tokens = word_tokenize(text.lower(), language="russian")
    return list(set(tokens))

# if os.path.exists(question_reply_path):
#     question_reply = pd.read_csv(question_reply_path)
# else:
question_reply = pd.DataFrame()
folders = glob.glob(config.chats_path)
for path in glob.glob(config.messages_path):
    chat = pd.read_csv(path, sep='\t', encoding='utf-8')
    chat = add_question_mark(chat)
    question_reply = question_reply.append(get_reply_mapping(chat))
print(question_reply)
questions_w_replies = question_reply[['question_message', 'question_message_id']].drop_duplicates()
question_reply["question_tokens"] = question_reply['question_message'].apply(tokenize)
question_reply.drop_duplicates().to_csv(config.question_reply_path, index=False, encoding='utf-8')
all_tokens = []
for question_tokens in question_reply.drop_duplicates('question_message')['question_tokens'].values:
    all_tokens += question_tokens

vc = pd.Series(all_tokens).value_counts()
use_tokens = set(vc.iloc[50:].index)
# use_tokens = None
fast_text = FastText(config.vectors_path, use_tokens)
with open(config.token_path) as f:
    TOKEN = f.read()

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(func=single_message_mark)
def handle_message(message):
    tokens = tokenize(message.text)
    question_reply['cosin'] = question_reply['question_tokens'].apply(lambda tokens2: fast_text.cosin(tokens, tokens2))
    df_notnull = question_reply[question_reply['cosin'].notnull()].sort_values('cosin')
    if len(df_notnull)>0:
        row = df_notnull.iloc[-1]
    else:
        return
    reply_message = row['reply_message']
    cosin = row['cosin']
    question_message = row['question_message']
    bot.reply_to(message, f"{reply_message}\nпохожесть: {cosin}\nВопрос: {question_message}")
print("start bot")
# Запуск бота
bot.polling()
