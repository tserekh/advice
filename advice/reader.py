import glob
import os
from typing import Dict, Optional
import config
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_message(message_tag) -> Optional[Dict[str, str]]:
    """
    Parse a single Telegram HTML-export message block.

    Telegram export uses:
    - service messages: <div class="message service" ...>
    - normal messages:  <div class="message default clearfix" ...>
    """
    # Пропускаем сервисные сообщения
    classes = message_tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    if "service" in classes:
        return None

    dic: Dict[str, str] = {
        "message_id": "",
        "from_name": "",
        "go_to_message_id": "",
        "message": "",
    }

    # Извлекаем ID сообщения
    msg_id = message_tag.get("id", "") or ""
    if msg_id.startswith("message"):
        dic["message_id"] = msg_id.replace("message", "")

    from_name = message_tag.find(attrs={"class": "from_name"})
    if from_name:
        dic["from_name"] = from_name.get_text(strip=True)

    reply = message_tag.find(attrs={"class": "reply_to details"})
    if reply:
        href = reply.find("a")
        if href and href.get("href"):
            dic["go_to_message_id"] = href.get("href").replace("#go_to_message", "")

    # Извлекаем текст сообщения/подпись из div class="text"
    text_div = message_tag.find(attrs={"class": "text"})
    if text_div:
        dic["message"] = text_div.get_text(separator="\n", strip=True)

    return dic


def resave_data(i, folder):
    folder_str = str(folder)
    if folder_str.lower().endswith(".html"):
        files = [folder_str]
    else:
        files = glob.glob(f"{folder_str}/*.html")
    messages = []
    chat_name = None
    
    for file in tqdm(files[::-1]):
        with open(file, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Telegram export использует:
            # - "message service"
            # - "message default clearfix"
            all_messages = soup.select("div.message")
            
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

    parsed_messages = [m for m in (get_message(m) for m in messages) if m is not None]
    df = pd.DataFrame(parsed_messages)

    if df.empty:
        print(f"Warning: No parsed messages found in {folder}, skipping...")
        return

    df["chat_name"] = chat_name

    df["message"] = df["message"].astype(str)
    df = df[df["message"].apply(len) > 0]
    df["message_id"] = df["message_id"].astype(str)
    df["go_to_message_id"] = df["go_to_message_id"].fillna("").astype(str)
    df["chat_name"] = df["chat_name"].fillna("").astype(str)
    os.makedirs(config.messages_path, exist_ok=True)
    df.to_csv(
        f"{config.messages_path}/{i}.tsv", index=False, encoding="utf-8", sep="\t"
    )
