import sys
import io
import base64
import json
import shutil
import time
import re
import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Налаштування ---

# API ключі для Azure OpenAI
API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = "https://codemie.lab.epam.com/llms"
API_VERSION = "2024-02-01"
MODEL_DEPLOYMENT_NAME = "gpt-4.1"  # Модель повинна підтримувати Vision (gpt-4o або gpt-4-turbo)

# Папка з вихідними фото машин
SOURCE_DIR = "raw_car_photos"

# Папка для збереження відібраних фото
# Скрипт створить її автоматично, якщо вона не існує
DEST_DIR = "car_plates"

# Файл з розміткою
LABEL_FILE = "label.txt"
# --------------------------------

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def get_next_index(dest_dir):
    """
    Перевіряє папку призначення і знаходить наступний вільний номер файлу.
    Наприклад, якщо є 0000005.png, поверне 6.
    """
    path = Path(dest_dir)
    if not path.exists():
        return 1

    existing_files = list(path.glob('*.png'))
    if not existing_files:
        return 1

    max_idx = 0
    for f in existing_files:
        try:
            # Витягуємо число з назви файлу (наприклад "0000005" -> 5)
            idx = int(f.stem)
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1


def clean_plate_text(text):
    """Видаляє пробіли та зайві символи з номера."""
    if text:
        return text.upper().replace(' ', '').replace('-', '').replace('.', '')
    return ""


def analyze_image_with_gpt(client, image_path):
    """
    Відправляє фото в GPT-4 Vision з промптом-фільтром.
    Повертає словник {valid: bool, plate: str, reason: str}
    """
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Системний промпт: жорсткі правила фільтрації
    system_prompt = """
    You are a dataset preparation assistant for Ukrainian License Plate Recognition.
    Your task is to analyze the image and decide if it is suitable for training.

    CRITERIA FOR "VALID":
    1. A Ukrainian license plate is CLEARLY visible.
    2. The plate is STANDARD: White background, Black text (e.g., AA1234BB).
    3. The image is sharp, not blurred, not pixelated.

    CRITERIA FOR "INVALID" (Reject immediately):
    1. Yellow plates (Bus/Taxi).
    2. Blue plates (Police).
    3. Black plates (Military).
    4. Red plates (Transit/Diplomatic).
    5. "Dummy" plates (e.g., "VIP", "BOSS", car model names like "X5", "Ranger").
    6. Plate is obscured, covered by snow/dirt, or Photoshop-blurred.
    7. The car is too far away to read the text reliably.

    OUTPUT FORMAT:
    Return ONLY a JSON object with these fields:
    - "valid": boolean (true if it meets all criteria, false otherwise).
    - "plate": string (the text on the plate, e.g., "AA1234BB". Empty if invalid).
    - "reason": string (short explanation why valid or invalid).
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_DEPLOYMENT_NAME,
            response_format={"type": "json_object"},  # Важливо для стабільності
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this car image based on the criteria."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                }
            ],
            max_tokens=150,
            temperature=0  # Максимальна точність
        )

        result_text = response.choices[0].message.content
        return json.loads(result_text)

    except Exception as e:
        print(f"   Помилка запиту до API: {e}")
        return None


def main():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)
    label_path = Path(LABEL_FILE)

    # Створення папки виводу, якщо немає
    dest_path.mkdir(parents=True, exist_ok=True)

    # Ініціалізація клієнта
    try:
        client = AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION
        )
    except Exception as e:
        print(f"Помилка ініціалізації клієнта: {e}")
        return

    # Отримуємо список всіх зображень
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in source_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not images:
        print(f"У папці {SOURCE_DIR} не знайдено зображень.")
        return

    print(f"Початок обробки. Знайдено {len(images)} файлів.")

    # Визначаємо, з якого номера починати (щоб не перезаписати старі)
    current_idx = get_next_index(DEST_DIR)
    print(f"Наступний файл буде збережено як: {current_idx:07d}.png")

    valid_count = 0

    for i, img_file in enumerate(images):
        print(f"[{i + 1}/{len(images)}] Аналіз: {img_file.name}...", end=" ", flush=True)

        result = analyze_image_with_gpt(client, img_file)

        if not result:
            print("Пропущено через помилку API.")
            continue

        is_valid = result.get("valid", False)
        plate_text = clean_plate_text(result.get("plate", ""))
        reason = result.get("reason", "No reason provided")

        if is_valid and len(plate_text) >= 4:
            # 1. Формуємо нове ім'я
            new_filename = f"{current_idx:07d}.png"
            new_file_path = dest_path / new_filename

            # 2. Копіюємо та конвертуємо в PNG (якщо треба, просто копіюємо)
            # Для надійності просто копіюємо файл
            shutil.copy2(img_file, new_file_path)

            # 3. Записуємо в файл
            # Формат: car_plates/0000001.png	AA1234BB
            relative_path = f"{dest_path.name}/{new_filename}"

            with open(label_path, "a", encoding="utf-8") as f:
                f.write(f"{relative_path}\t{plate_text}\n")

            print(f"ОК! ({plate_text}) -> {new_filename}")
            current_idx += 1
            valid_count += 1
        else:
            print(f"ВІДХИЛЕНО. Причина: {reason} ({plate_text})")

        # Пауза щоб не спамити API
        time.sleep(1.5)

    print("\n" + "=" * 40)
    print(f"Роботу завершено.")
    print(f"   Всього оброблено: {len(images)}")
    print(f"   Відібрано та збережено: {valid_count}")
    print(f"   Результати в: {LABEL_FILE}")


if __name__ == "__main__":
    main()