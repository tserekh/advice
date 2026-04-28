import os
from dotenv import load_dotenv
import httpx

# Загружаем переменные из .env
load_dotenv()

http_proxy = os.getenv("HTTP_PROXY")
https_proxy = os.getenv("HTTPS_PROXY")

print(f"HTTP_PROXY: {http_proxy}")
print(f"HTTPS_PROXY: {https_proxy}")

if not http_proxy:
    print("Ошибка: HTTP_PROXY не найден в .env файле!")
    exit(1)

# Пробуем сделать запрос через прокси
try:
    print(f"\nПопытка подключения к Google через прокси {http_proxy}...")
    
    # Явно указываем прокси для клиента
    with httpx.Client(proxy=http_proxy, timeout=10.0) as client:
        response = client.get("https://www.google.com")
        print(f"Статус код: {response.status_code}")
        print("Успех! Прокси работает.")
        
except Exception as e:
    print(f"Ошибка подключения: {e}")
    print("\nВозможные причины:")
    print("1. Неверный формат прокси (должен быть http://user:pass@ip:port)")
    print("2. Прокси сервер недоступен или требует другой тип авторизации")
    print("3. Порт 443 может не поддерживать HTTP CONNECT туннелирование, попробуй другой порт если есть")
