import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm
from typing import Dict, Iterable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import config
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score


def get_message(message: str)->Dict:
    dic = {}
    dic['message'] = message.text.split('\n\n')[-2]
    dic['message_id'] = message.get('id').replace('message', '')
    dic['from_name'] = message.find(attrs={'class': 'from_name'})
    reply = message.find(attrs={'class': 'reply_to details'})
    if reply:
        dic['go_to_message_id'] = reply.find('a').get('href').replace('#go_to_message', '')
    return dic

# папки с ChatExport
sub_list = []
def read_chats_data() -> Iterable[pd.DataFrame]:
    folders = glob.glob(config.chats_path)
    for folder in folders:
        files = glob.glob(f"{folder}/*.html")
        messages = []
        for file in tqdm(files):
            with open(file, encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                messages_part = soup.body.find_all(attrs = {'class':'message'})
            messages+=messages_part
        chat_name = soup.find(attrs = {'class', "text bold"}).text
        df = pd.DataFrame(list(map(get_message, messages)))
        df["chat_name"] = chat_name
        yield df