import sys
import io
import json
import cv2
import numpy as np
import re
from ultralytics import YOLO
from pathlib import Path
import subprocess
import os

# ============================================================================
# НАБІР СИМВОЛІВ ДЛЯ РОЗПІЗНАВАННЯ (СИМВОЛИ З UA_DICT.TXT)
# ============================================================================
# Цей набір включає всі символи, які модель OCR навчена розпізнавати.
#
# -- Літери (15) --
# A, B, C, E, H, I, K, M, O, P, T, X, D, U, Y
#
# -- Цифри (10) --
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# ============================================================================

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ініціалізація
yolo_model = YOLO('train_models/YOLO/my_YOLO_detection_car_plates.pt')


def preprocess_plate_image(plate_crop):
    """М'який препроцесинг - тільки збільшення розміру"""
    # Конвертуємо в градації сірого
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Збільшуємо розмір якщо номер маленький
    height, width = gray.shape
    if height < 80:
        scale = 2  # Менше збільшення = менше артефактів
        gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

    # М'яке підвищення контрасту
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # БЕЗ бінаризації - вона створює артефакти!
    return gray


def smart_ocr_correction(raw_text):
    """
    Спрощене розумне виправлення OCR помилок.
    Виправляє ЛИШЕ логічні помилки (O/0, I/1, B/8) на основі контексту,
    оскільки модель вже обмежена словником ua_dict.txt.
    """
    # Очищення: видаляємо лише пробіли та дефіси.
    text = raw_text.replace(' ', '').replace('-', '').upper()

    # Якщо текст порожній
    if not text or len(text) < 3:
        return ""

    # Перетворюємо в список для зручної заміни
    chars = list(text)

    # === ГОЛОВНА ЛОГІКА ===
    # Контекстне виправлення O/0, I/1, B/8.
    # Це найважливіша частина, оскільки ці символи є у словнику.
    if len(chars) == 8:
        # Перші 2 символи - мають бути літери
        for i in [0, 1]:
            if chars[i] == '0':
                chars[i] = 'O'
            elif chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '8':
                chars[i] = 'B'

        # Середні 4 символи (позиції 2-5) - мають бути цифри
        for i in range(2, 6):
            if chars[i] == 'O':
                chars[i] = '0'
            elif chars[i] == 'I':
                chars[i] = '1'
            elif chars[i] == 'B':
                chars[i] = '8'

        # Останні 2 символи (позиції 6-7) - мають бути літери
        for i in [6, 7]:
            if chars[i] == '0':
                chars[i] = 'O'
            elif chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '8':
                chars[i] = 'B'

    text = ''.join(chars)

    # === ВАЛІДАЦІЯ ФОРМАТУ ===
    # Використовуємо твій набір з 15 літер
    allowed_letters = 'ABCEHIKMOPTXDUY'

    # Перевірка стандартного формату AA1234BB
    standard_pattern = fr'^[{allowed_letters}]{{2}}\d{{4}}[{allowed_letters}]{{2}}$'
    if re.match(standard_pattern, text):
        return text
    # Повертаємо "як є", якщо нічого не підійшло (але пройшло поріг довжини)
    return text if len(text) >= 5 else ""


def detect_pure_yolo(image_path):
    if not Path(image_path).exists():
        return {"error": "File not found"}

    # YOLO детекція НОМЕРНИХ ЗНАКІВ (натренована модель!)
    results = yolo_model(image_path, verbose=False, iou=0.5, conf=0.3)

    img = cv2.imread(image_path)
    detected_cars = []

    for result in results:
        boxes = result.boxes

        for i, box in enumerate(boxes):
            # Вирізаємо область номерного знаку
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]

            # ЗБЕРІГАЄМО оригінал вирізки
            original_crop_path = f"yolo_crop_{i}.jpg"
            cv2.imwrite(original_crop_path, plate_crop)

            # ПРЕПРОЦЕСИНГ для кращого розпізнавання
            processed_plate = preprocess_plate_image(plate_crop)

            # Зберігаємо препроцесинг
            processed_crop_path = f"yolo_crop_{i}_processed.jpg"
            # Отримуємо абсолютний шлях, оскільки subprocess буде запущений з іншої директорії
            abs_processed_crop_path = os.path.abspath(processed_crop_path)
            cv2.imwrite(abs_processed_crop_path, processed_plate)

            paddle_ocr_dir = '../PaddleOCR-main'
            predict_script_path = 'tools/infer/predict_rec.py'
            model_dir_relative = './inference/ua_model'
            dict_path_relative = 'ua_dict.txt'

            command = [
                sys.executable,
                predict_script_path,
                '--image_dir', abs_processed_crop_path,
                '--rec_model_dir', model_dir_relative,
                '--rec_image_shape', '3, 48, 320',
                '--rec_char_dict_path', dict_path_relative,
                '--use_gpu', 'False',
                '--show_log', 'False'
            ]

            ocr_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=paddle_ocr_dir
            )

            all_text = []

            if ocr_process.returncode != 0:
                print(f"    PADDLE SUBPROCESS FAILED (return code {ocr_process.returncode}):", file=sys.stderr)
                print(f"   --- STDOUT ---", file=sys.stderr)
                print(ocr_process.stdout, file=sys.stderr)
                print(f"   --- STDERR ---", file=sys.stderr)
                print(ocr_process.stderr, file=sys.stderr)

            # Парсимо стандартний вивід
            raw_paddle_text = ""
            paddle_confidence = 0.0
            found_prediction = False

            for line in ocr_process.stdout.splitlines():
                if "Predicts of" in line:
                    match = re.search(r":\s*\('([^']*)',\s*([0-9.]*)\)", line)
                    if match:
                        raw_paddle_text = match.group(1)
                        paddle_confidence = float(match.group(2))

                        # ФІЛЬТРУЄМО: беремо тільки впевнені результати (>30%)
                        if paddle_confidence > 0.3:
                            all_text.append({
                                "text": raw_paddle_text,
                                "confidence": paddle_confidence
                            })
                        found_prediction = True
                        break  # Знайшли, далі не шукаємо

            if not found_prediction and ocr_process.returncode == 0:
                print(f"   Paddle process ran successfully, but no prediction was found for crop {i}.",
                      file=sys.stderr)
            if all_text:
                # Об'єднуємо тільки якісні фрагменти
                raw_combined = " ".join([t["text"] for t in all_text])
                avg_conf = sum([t["confidence"] for t in all_text]) / len(all_text)

                # ВИПРАВЛЯЄМО помилки OCR
                corrected_plate = smart_ocr_correction(raw_combined)

                # Додаємо тільки якщо номер не порожній після очищення
                if corrected_plate and len(corrected_plate) >= 5:
                    detected_cars.append({
                        "plate": corrected_plate,
                        "raw_text": raw_combined,
                        "confidence": round(avg_conf * 100, 1),
                        "fragments": all_text,
                        "has_disabled_badge": False
                    })

    return {
        "cars": detected_cars,
        "has_disabled_parking_sign": False
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_yolo_ocr_clean.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_to_process = sys.argv[1]

    if not os.path.isabs(image_to_process):
        image_to_process = os.path.abspath(image_to_process)

    result = detect_pure_yolo(image_to_process)

    # === ФІНАЛЬНИЙ РЕЗУЛЬТАТ ===
    print(json.dumps(result, ensure_ascii=False, indent=2))