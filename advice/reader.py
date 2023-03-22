import glob
from typing import Dict, Union
import config
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import telebot


def get_message(message: telebot.types.Message) -> Dict:
    dic = {}
    dic["message"] = message.text.split("\n\n")[-2]
    dic["message_id"] = message.get("id").replace("message", "")
    from_name = message.find(attrs={"class": "from_name"})
    if from_name:
        dic["from_name"] = from_name.text
    reply = message.find(attrs={"class": "reply_to details"})
    if reply:
        dic["go_to_message_id"] = (
            reply.find("a").get("href").replace("#go_to_message", "")
        )
    return dic


def resave_data(i, folder):
    files = glob.glob(f"{folder}/*.html")
    messages = []
    for file in tqdm(files[::-1]):
        with open(file, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            messages_part = soup.body.find_all(attrs={"class": "message"})
        messages += messages_part
    chat_name = soup.find(attrs={"class", "text bold"}).text
    df = pd.DataFrame(list(map(get_message, messages)))
    df["chat_name"] = chat_name

    df["message"] = df["message"].astype(str)
    df = df[df["message"].apply(len) > 0]
    df["message_id"] = df["message_id"].astype(str)
    df["go_to_message_id"] = df["go_to_message_id"].fillna("").astype(str)
    df["chat_name"] = df["chat_name"].fillna("").astype(str)
    df.to_csv(
        f"{config.messages_path}/{i}.tsv", index=False, encoding="utf-8", sep="\t"
    )
