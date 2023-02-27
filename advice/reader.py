import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score


def get_message(mes):
    dic = {}
    dic['message'] = mes.text.split('\n\n')[-2]
    dic['message_id'] = mes.get('id').replace('message', '')
    dic['from_name'] = mes.find(attrs={'class': 'from_name'})
    reply = mes.find(attrs={'class': 'reply_to details'})

    # print(reply)
    if reply:
        dic['go_to_message_id'] = reply.find('a').get('href').replace('#go_to_message', '')
    return dic

# папки с ChatExport
sub_list = []
folders = glob.glob("C:/Users/artem/Downloads/Telegram Desktop/ChatExport/ChatExport*")
files = []
for folder in folders:
    files = glob.glob(f"{folder}/*.html")
    messages_all = []
    for file in tqdm(files):
        with open(file, encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            messages = soup.body.find_all(attrs = {'class':'message'})#.text#.split('\n\n')[-2].lower()
        messages_all+=messages
    chat_name = soup.find(attrs = {'class', "text bold"}).text
    df_sub = pd.DataFrame(list(map(get_message, messages_all)))
    df_sub["chat_name"] = chat_name
    sub_list.append(df_sub)
df = pd.concat(sub_list)