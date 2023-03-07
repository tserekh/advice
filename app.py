from advice.reader import read_chats_data
from advice.marker import add_question_mark, get_reply_mapping
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from advice.marker import FastText


question_reply = pd.DataFrame()
chats = read_chats_data()
for chat in chats:
    chat_w_mark = add_question_mark(chat)
    question_reply = question_reply.append(get_reply_mapping(chat_w_mark))

questions_w_replies = question_reply[['question_message', 'question_message_id']].drop_duplicates()



def tokenize(text):
    tokens = word_tokenize(text.lower(), language="russian")
    return list(set(tokens))

question_reply["question_tokens"] = question_reply['question_message'].apply(tokenize)

# questions_w_replies.drop_duplicates('question_message')

all_tokens = []

for question_tokens in question_reply.drop_duplicates('question_message')['question_tokens'].values:
    all_tokens+=question_tokens

vc = pd.Series(all_tokens).value_counts()

use_tokens = list(vc.iloc[50:].index)


fast_text = FastText("D:/Downloads/cc.ru.300.vec/cc.ru.300.vec", use_tokens)

text = "Кто-нибудь может посоветовать инструктора по сноуборду?"

tokens = tokenize(text)

question_reply['cosin'] = question_reply['question_tokens'].apply(lambda tokens2: fast_text.cosin(tokens, tokens2))

row = question_reply[question_reply['cosin'].notnull()].sort_values('cosin').iloc[-1]

reply_message = row['reply_message']