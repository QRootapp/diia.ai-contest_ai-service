import sys
import json
import cv2
import re
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time  # Додано для унікальних імен файлів


def preprocess_plate_image(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 80:
        gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def correct_plate_text(text):
    allowed_letters = 'ABCEHIKMOPTXDUY'
    standard_pattern = fr'^[{allowed_letters}]{{2}}\d{{4}}[{allowed_letters}]{{2}}$'
    text = text.replace(' ', '').replace('-', '').upper()
    if not text or len(text) < 3:
        return ""
    chars = list(text)
    # Спроба виправити, якщо довжина схожа на стандартну (8 символів)
    if len(chars) == 8:
        for i in [0, 1, 6, 7]:  # літери
            if chars[i] == '0': chars[i] = 'O'
            if chars[i] == '1': chars[i] = 'I'
            if chars[i] == '8': chars[i] = 'B'
        for i in range(2, 6):  # цифри
            if chars[i] == 'O': chars[i] = '0'
            if chars[i] == 'I': chars[i] = '1'
            if chars[i] == 'B': chars[i] = '8'

    text = ''.join(chars)

    if re.match(standard_pattern, text):
        return text

    # --- УВАГА: Ось тут була потенційна проблема ---
    # Якщо текст не пройшов RegEx, ви повертали все сміття, якщо воно довше 5 символів.
    # Я залишив це як є, але додав коментар.
    return text if len(text) >= 5 else ""


def detect_license_plate(image_path, yolo, ocr):
    if not Path(image_path).exists():
        return {"error": "File not found"}

    # Створення папки для дебагу
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)

    # Очистка папки перед запуском (опціонально, щоб не накопичувати тисячі файлів)
    # for file in debug_dir.glob("*"):
    #     file.unlink()

    img = cv2.imread(image_path)
    detected_cars = []

    # Зменшив conf для YOLO, щоб точно щось знайти, якщо модель слабка
    results = yolo(img, verbose=False, iou=0.5, conf=0.3)

    timestamp = int(time.time())

    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 1. Вирізаємо
            crop = img[y1:y2, x1:x2]

            # --- ЗБЕРЕЖЕННЯ RAW (те що бачить YOLO) ---
            raw_filename = debug_dir / f"{timestamp}_box{j}_raw.jpg"
            cv2.imwrite(str(raw_filename), crop)

            # 2. Обробляємо
            processed_crop = preprocess_plate_image(crop)

            # --- ЗБЕРЕЖЕННЯ PROCESSED (те що йде в OCR) ---
            proc_filename = debug_dir / f"{timestamp}_box{j}_processed.jpg"
            cv2.imwrite(str(proc_filename), processed_crop)

            ocr_out = ocr.predict(processed_crop)

            fragments = []
            if ocr_out and isinstance(ocr_out, list):
                # PaddleOCR повертає структуру [ [ [points], (text, conf) ], ... ]
                # Іноді структура відрізняється залежно від версії (v3/v4 vs v2)
                # У вашому коді використовується rec.get('rec_texts'), що схоже на PP-Structure або новішу версію.
                # Якщо це стандартний PaddleOCR, output зазвичай: result[0] -> list of lines

                # Логування сирого виходу OCR для розуміння структури
                # print(f"[DEBUG OCR RAW] {ocr_out}")

                rec = ocr_out[0]
                # Адаптація під різні формати відповіді Paddle
                if isinstance(rec, dict):
                    texts = rec.get('rec_texts', [])
                    scores = rec.get('rec_scores', [])
                elif isinstance(rec, list):
                    # Стандартний формат PaddleOCR: [[[[x,y],...], ("text", 0.9)], ...]
                    texts = [line[1][0] for line in rec]
                    scores = [line[1][1] for line in rec]
                else:
                    texts = []
                    scores = []

                for txt, score in zip(texts, scores):
                    if txt and score > 0.3:  # Поріг впевненості OCR
                        fragments.append({"text": txt, "confidence": score})

            if fragments:
                raw_text = " ".join(f["text"] for f in fragments)
                confidence = sum(f["confidence"] for f in fragments) / len(fragments)
                corrected = correct_plate_text(raw_text)

                # Додаємо інфо про файл дебагу у вивід, щоб ви знали куди дивитись
                if corrected and len(corrected) >= 5:
                    detected_cars.append({
                        "plate": corrected,
                        "raw_text": raw_text,
                        "confidence": round(confidence * 100, 1),
                        "debug_file": str(raw_filename)
                    })

    return {"cars": detected_cars}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path>", file=sys.stderr)
        sys.exit(1)

    # Ініціалізація YOLO
    yolo_model = YOLO('train_models/YOLO/my_YOLO_detection_car_plates.pt')

    # ВИПРАВЛЕНО: Прибрано show_log та lang, які викликали помилку та warning
    # use_angle_cls=False залишаємо, щоб не перевертало текст (важливо для горизонтальних номерів)
    paddle_ocr = PaddleOCR(text_recognition_model_dir='train_models/OCR', use_angle_cls=False)

    image_path = sys.argv[1]
    if not Path(image_path).is_absolute():
        image_path = str(Path(image_path).resolve())

    result = detect_license_plate(image_path, yolo_model, paddle_ocr)
    print(json.dumps(result, ensure_ascii=False, indent=2))