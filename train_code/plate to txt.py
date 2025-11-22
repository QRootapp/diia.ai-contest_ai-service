import sys
import io
import base64
from pathlib import Path
from openai import AzureOpenAI
import time
import re
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dotenv import load_dotenv

load_dotenv()

# --- Налаштування ---

# API ключі для Azure OpenAI
API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = "https://codemie.lab.epam.com/llms"
API_VERSION = "2024-02-01"
MODEL_DEPLOYMENT_NAME = "gpt-4.1"  # Назва моделі на сервері

# Папка з вирізаними номерами
SOURCE_DIR = r'D:\JOB\HAKATON\parking-violations-detector\car_plates'

# Вихідний файл для розмітки
OUTPUT_FILE = r'D:\JOB\HAKATON\parking-violations-detector\label.txt'
# --------------------

# Регулярний вираз для очищення
VALID_CHARS_RE = re.compile(r'[^A-Z0-9АВІКМНОРСТХ]')


def clean_plate_text(text):
    """Очищує текст, отриманий від AI, до стандартного формату."""
    text = text.strip().upper()
    text = text.replace(' ', '').replace('-', '').replace(':', '').replace('.', '')
    text = VALID_CHARS_RE.sub('', text)
    return text


def auto_label_images_azure():
    """
    Проходить по папці, відправляє кожне фото в AzureOpenAI
    і записує результат у label.txt.
    """

    try:
        client = AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
    except Exception as e:
        print(f"Помилка конфігурації Azure OpenAI: {e}")
        return

    source_path = Path(SOURCE_DIR)
    label_path = Path(OUTPUT_FILE)

    image_files = sorted(list(source_path.glob('*.png')))

    if not image_files:
        print(f"Помилка: Не знайдено файлів .png у папці '{SOURCE_DIR}'.")
        return

    print(f"Запуск авто-розмітки (Azure {MODEL_DEPLOYMENT_NAME})... Знайдено {len(image_files)} зображень.")

    with open(label_path, 'a', encoding='utf-8') as f:
        for i, img_path in enumerate(image_files):
            print(f"--- Обробка [ {i + 1}/{len(image_files)} ]: {img_path.name} ---")
            try:
                # 1. Кодуємо зображення в base64
                with open(img_path, "rb") as image_file:
                    b64_image = base64.b64encode(image_file.read()).decode('utf-8')

                # 2. Створюємо простий запит для нашої задачі
                response = client.chat.completions.create(
                    model=MODEL_DEPLOYMENT_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Розпізнай текст на цьому українському номерному знаку. Відповідай ТІЛЬКИ текстом номерного знаку, без жодних пояснень. Якщо номер нечитабельний, напиши 'BAD_IMAGE'."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=30  # Номерний знак короткий
                )

                # 3. Отримуємо та очищуємо текст
                raw_text = response.choices[0].message.content
                cleaned_text = clean_plate_text(raw_text)

                # 4. Записуємо результат
                if "BAD_IMAGE" in raw_text.upper() or len(cleaned_text) < 4:
                    print(f"   Пропущено (нечитабельно): '{raw_text}'")
                else:
                    relative_path = f"{source_path.name}/{img_path.name}"
                    f.write(f"{relative_path}\t{cleaned_text}\n")
                    print(f"   Збережено: {cleaned_text} (оригінал: '{raw_text}')")

                # Затримка для дотримання лімітів API
                time.sleep(1)  # Можна збільшити при виникненні помилок "rate limit"

            except Exception as e:
                print(f"   Помилка API (файл пропущено): {e}")
                time.sleep(5)

    print("\n" + "=" * 40)
    print(f"Автоматичну розмітку завершено.")
    print(f"   Результати збережено у файлі: {OUTPUT_FILE}")
    print("   Рекомендується вручну перевірити файл та виправити помилки.")
    print("=" * 40)


if __name__ == "__main__":
    auto_label_images_azure()