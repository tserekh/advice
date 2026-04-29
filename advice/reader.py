import glob
from typing import Dict, Union
import config
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_message(message_tag) -> Dict:
    dic = {}
    # Извлекаем ID сообщения
    msg_id = message_tag.get("id", "")
    if msg_id.startswith("message"):
        dic["message_id"] = msg_id.replace("message", "")
    else:
        dic["message_id"] = ""
    
    # Пропускаем сервисные сообщения
    classes = message_tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    if "service" in classes:
        return None
    
    from_name = message_tag.find(attrs={"class": "from_name"})
    if from_name:
        dic["from_name"] = from_name.get_text(strip=True)
    
    reply = message_tag.find(attrs={"class": "reply_to details"})
    if reply:
        href = reply.find("a")
        if href and href.get("href"):
            dic["go_to_message_id"] = href.get("href").replace("#go_to_message", "")
    
    # Извлекаем текст сообщения из div class="text"
    text_div = message_tag.find(attrs={"class": "text"})
    if text_div:
        # Получаем текст, сохраняя переносы строк как \n
        text = text_div.get_text(separator="\n", strip=True)
        # Заменяем <br> на переносы строк (на случай если они не обработались)
        dic["message"] = text
    else:
        dic["message"] = ""
    
    return dic


def resave_data(i, folder):
    files = glob.glob(f"{folder}/*.html")
    messages = []
    chat_name = None
    
    for file in tqdm(files[::-1]):
        with open(file, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Находим все сообщения с классом "message"
            # Telegram export использует классы: "message default", "message service"
            all_messages = soup.body.find_all(attrs={"class": lambda x: x and isinstance(x, list) and "message" in x})
            
            # Извлекаем имя чата из первого файла
            if chat_name is None:
                name_tag = soup.find(attrs={"class": "text bold"})
                if name_tag:
                    chat_name = name_tag.get_text(strip=True)
            
            # Фильтруем только обычные сообщения (не service)
            for msg in all_messages:
                classes = msg.get("class", [])
                if isinstance(classes, str):
                    classes = [classes]
                if "service" not in classes and "default" in classes:
                    messages.append(msg)
    
    if not messages:
        print(f"Warning: No messages found in {folder}, skipping...")
        return
    
    if chat_name is None:
        chat_name = "Unknown_Chat"
    
    df = pd.DataFrame(list(map(get_message, messages)))
    # Удаляем None (сервисные сообщения, которые могли просочиться)
    df = df.dropna(subset=["message"])
    df["chat_name"] = chat_name

    df["message"] = df["message"].astype(str)
    df = df[df["message"].apply(len) > 0]
    df["message_id"] = df["message_id"].astype(str)
    df["go_to_message_id"] = df["go_to_message_id"].fillna("").astype(str)
    df["chat_name"] = df["chat_name"].fillna("").astype(str)
    df.to_csv(
        f"{config.messages_path}/{i}.tsv", index=False, encoding="utf-8", sep="\t"
    )
