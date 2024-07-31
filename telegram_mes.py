
import os
import requests
import json

# Set environment variables with your actual bot token and chat ID
os.environ['BOT_TOKEN'] = "7391245801:AAGW5uZnJE9n3hBUpepqigahNDPfGiGger0"
os.environ['CHAT_ID'] = "7436946463"  # Replace this with your actual chat ID

# Get the environment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

def send_telegram_message(message):
    url_message = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }
    payload_message = {
        "chat_id": CHAT_ID,
        "text": "This is the view from your cam. "+message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "disable_notification": True,
    }
    response_message = requests.post(url_message, headers=headers, data=json.dumps(payload_message))
    print(response_message.status_code)
    print(response_message.json())

def send_telegram_photo(photo_path):
    url_photo = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    payload_photo = {
        "chat_id": CHAT_ID,
    }
    with open(photo_path, "rb") as photo:
        files = {
            "photo": photo
        }
        response_photo = requests.post(url_photo, data=payload_photo, files=files)
    print(response_photo.status_code)
    print(response_photo.json())
