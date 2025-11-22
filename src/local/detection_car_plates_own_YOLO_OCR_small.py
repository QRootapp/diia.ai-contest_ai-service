import sys
import json
import cv2
import re
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

def preprocess_plate_image(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 80:
        gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)
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
    return text if len(text) >= 5 else ""

def detect_license_plate(image_path, yolo, ocr):
    #print(f"[DEBUG] Очікую файл: {image_path}. Існує: {Path(image_path).exists()}")
    if not Path(image_path).exists():
        return {"error": "File not found"}

    img = cv2.imread(image_path)
    #print("cv2.imread result:", img is not None)
    detected_cars = []
    results = yolo(img, verbose=False, iou=0.5, conf=0.3)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crop = preprocess_plate_image(crop)
            ocr_out = ocr.predict(crop)
            fragments = []
            if ocr_out and isinstance(ocr_out, list):
                rec = ocr_out[0]
                texts = rec.get('rec_texts', [])
                scores = rec.get('rec_scores', [])
                for txt, score in zip(texts, scores):
                    if txt and score > 0.3:
                        fragments.append({"text": txt, "confidence": score})

            if fragments:
                raw_text = " ".join(f["text"] for f in fragments)
                confidence = sum(f["confidence"] for f in fragments) / len(fragments)
                corrected = correct_plate_text(raw_text)
                if corrected and len(corrected) >= 5:
                    detected_cars.append({
                        "plate": corrected,
                        "raw_text": raw_text,
                        "confidence": round(confidence * 100, 1)
                    })
    return {"cars": detected_cars}
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path>", file=sys.stderr)
        sys.exit(1)

    yolo_model = YOLO('train_models/YOLO/my_YOLO_detection_car_plates.pt')
    paddle_ocr = PaddleOCR(text_recognition_model_dir='train_models/OCR')

    image_path = sys.argv[1]
    if not Path(image_path).is_absolute():
        image_path = str(Path(image_path).resolve())

    result = detect_license_plate(image_path, yolo_model, paddle_ocr)
    print(json.dumps(result, ensure_ascii=False, indent=2))