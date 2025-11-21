import sys
import io
import base64
import json
import shutil
import time
import re
import os
import threading
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from PIL import Image  # Потрібно встановити: pip install Pillow
import os
from dotenv import load_dotenv

load_dotenv()

# --- НАЛАШТУВАННЯ КОРИСТУВАЧА ---

# 1. Ваші API ключі
API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = "https://codemie.lab.epam.com/llms"
API_VERSION = "2024-02-01"
MODEL_DEPLOYMENT_NAME = "gpt-4.1"

# 2. Папки
SOURCE_DIR = "raw_car_photos"
DEST_DIR = "car_plates"
LABEL_FILE = "label.txt"

# Кількість потоків для паралельної обробки
MAX_WORKERS = 8
# --------------------------------

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

file_write_lock = threading.Lock()
print_lock = threading.Lock()


def get_next_index(dest_dir):
    """Знаходить наступний вільний номер файлу."""
    path = Path(dest_dir)
    if not path.exists():
        return 1
    existing_files = list(path.glob('*.png'))
    if not existing_files:
        return 1
    max_idx = 0
    for f in existing_files:
        try:
            idx = int(f.stem)
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1


def clean_plate_text(text):
    if text:
        # Залишаємо тільки букви і цифри
        return text.upper().replace(' ', '').replace('-', '').replace('.', '').strip()
    return ""


def compress_image(image_path, max_size=1024, quality=70):
    """
    Зменшує розмір зображення та стискає його в пам'яті.
    Повертає байти JPEG.
    """
    try:
        with Image.open(image_path) as img:
            # Конвертуємо в RGB (щоб уникнути проблем з PNG transparency)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Зменшуємо розмір, якщо він великий
            img.thumbnail((max_size, max_size))

            # Зберігаємо в буфер (в пам'ять)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None


def analyze_image_with_retry(client, image_path, max_retries=5):
    """
    Відправляє запит з автоматичним повтором при помилках.
    """
    # ВИКОРИСТОВУЄМО СТИСНЕННЯ ЗАМІСТ ПРЯМОГО ЧИТАННЯ
    image_data = compress_image(image_path)

    if not image_data:
        return None  # Файл пошкоджений

    b64_image = base64.b64encode(image_data).decode('utf-8')

    system_prompt = """
    You are a dataset preparation assistant for Ukrainian License Plate Recognition.

    CRITERIA FOR "VALID":
    1. A Ukrainian license plate is CLEARLY visible.
    2. The plate is STANDARD: White background, Black text (e.g., AA1234BB).
    3. The image is sharp.

    CRITERIA FOR "INVALID":
    1. Yellow (Bus), Blue (Police), Black (Military), Red (Transit) plates.
    2. "Dummy" plates (names, brands like "Q7", "BMW").
    3. Obscured, dirty, blurry, or too far away.
    4. Foreign plates (not UA).

    OUTPUT FORMAT (JSON ONLY):
    { "valid": boolean, "plate": "string", "reason": "string" }
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_DEPLOYMENT_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze this car image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]}
                ],
                max_tokens=150,
                temperature=0
            )
            result_text = response.choices[0].message.content
            return json.loads(result_text)

        except Exception as e:
            error_msg = str(e)
            # Ловимо 429 (Rate Limit), 5xx (Server Error) і тепер 413 (хоча з компресією його не має бути)
            if "429" in error_msg or "500" in error_msg or "502" in error_msg or "413" in error_msg:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                with print_lock:
                    print(f"   API зайнятий/помилка ({error_msg[:30]}...). Повтор через {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                with print_lock:
                    print(f"   Критична помилка для {image_path.name}: {e}")
                return None
    return None


def process_single_image(client, img_file, dest_path, label_path, current_idx_container):
    result = analyze_image_with_retry(client, img_file)

    if not result:
        return False, img_file.name, "API/File Error"

    is_valid = result.get("valid", False)
    plate_text = clean_plate_text(result.get("plate", ""))
    reason = result.get("reason", "No reason")

    # Додаткова перевірка на довжину номера (стандартні UA номери зазвичай 8 символів)
    if is_valid and len(plate_text) >= 4:
        with file_write_lock:
            idx = current_idx_container[0]

            new_filename = f"{idx:07d}.png"
            new_file_path = dest_path / new_filename

            shutil.copy2(img_file, new_file_path)

            relative_path = f"{dest_path.name}/{new_filename}"
            with open(label_path, "a", encoding="utf-8") as f:
                f.write(f"{relative_path}\t{plate_text}\n")

            current_idx_container[0] += 1

        return True, plate_text, new_filename
    else:
        return False, plate_text, reason


def main():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)
    label_path = Path(LABEL_FILE)
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        client = AzureOpenAI(api_key=API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=API_VERSION)
    except Exception as e:
        print(f"Помилка ініціалізації: {e}")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in source_path.iterdir() if f.suffix.lower() in valid_extensions]

    if not images:
        print(f"Не знайдено зображень у {SOURCE_DIR}")
        return

    start_idx = get_next_index(DEST_DIR)
    current_idx_container = [start_idx]

    print(f"Запуск багатопотокової обробки (зі стисненням).")
    print(f"   Зображень: {len(images)}")
    print(f"   Потоків: {MAX_WORKERS}")
    print("=" * 50)

    valid_count = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_image, client, img, dest_path, label_path, current_idx_container): img.name
            for img in images
        }

        for future in as_completed(futures):
            processed_count += 1
            img_name = futures[future]
            try:
                success, text_or_reason, filename_or_reason = future.result()

                with print_lock:
                    if success:
                        print(f"[{processed_count}/{len(images)}] ЗБЕРЕЖЕНО: {text_or_reason} -> {filename_or_reason}")
                        valid_count += 1
                    else:
                        # Можна розкоментувати для детального логування відхилених
                        # print(f"[{processed_count}/{len(images)}] ПРОПУЩЕНО: {filename_or_reason} ({text_or_reason})")
                        pass

            except Exception as e:
                with print_lock:
                    print(f"[{processed_count}/{len(images)}] ЗБІЙ на {img_name}: {e}")

    print("\n" + "=" * 40)
    print(f"Завершено.")
    print(f"   Оброблено: {len(images)}")
    print(f"   Збережено: {valid_count}")


if __name__ == "__main__":
    main()