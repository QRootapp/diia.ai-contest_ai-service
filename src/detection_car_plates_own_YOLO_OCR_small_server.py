import uvicorn
import cv2
import numpy as np
import re
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- ГЛОБАЛЬНІ ЗМІННІ ДЛЯ МОДЕЛЕЙ ---
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ПРИ СТАРТІ ---
    print("Завантаження моделей...")
    try:
        models["yolo"] = YOLO('train_models/YOLO/my_YOLO_detection_car_plates.pt')
        models["ocr"] = PaddleOCR(text_recognition_model_dir='train_models/OCR')
        print("Моделі успішно завантажено.")
    except Exception as e:
        print(f"Помилка завантаження моделей: {e}")

    yield

    # Очищення ресурсів при вимкненні
    models.clear()


app = FastAPI(lifespan=lifespan)

# Налаштування CORS для інтеграції з фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # У production вкажіть конкретні домени
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ДОПОМІЖНІ ФУНКЦІЇ (ІДЕНТИЧНІ ДО ОРИГІНАЛЬНОГО ФАЙЛУ) ---

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


def detect_license_plate(img, yolo, ocr):
    """
    Основна логіка детекції (ідентична до оригінального файлу).
    img: numpy array (зображення).
    """
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


# --- API ЕНДПОІНТИ ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """Проста HTML сторінка для тестування API"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Детекція номерних знаків</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            input[type="file"] { margin: 20px 0; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Детекція номерних знаків</h1>
        <form id="uploadForm">
            <input type="file" id="imageFile" accept="image/*" required>
            <br>
            <button type="submit">Відправити на обробку</button>
        </form>
        <div id="result"></div>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('imageFile');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    resultDiv.textContent = 'Будь ласка, виберіть файл';
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.textContent = 'Обробка...';
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    resultDiv.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultDiv.textContent = 'Помилка: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Перевірка стану сервера"""
    return {"status": "ok", "message": "Сервер працює"}


@app.post("/detect")
async def detect_license_plate_endpoint(file: UploadFile = File(...)):
    """
    Ендпоінт для обробки зображення та детекції номерних знаків.
    Отримує файл через HTTP POST, обробляє його та повертає JSON.
    
    Файл передається через multipart/form-data (form data), а не raw binary.
    FastAPI автоматично парсить multipart/form-data і надає доступ через UploadFile.
    """
    # Перевірка формату
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл має бути зображенням")

    try:
        # Читання бінарних даних з form data (multipart/form-data)
        # file.read() повертає bytes - бінарні дані файлу
        contents = await file.read()
        # Конвертуємо bytes в numpy array для OpenCV
        nparr = np.frombuffer(contents, np.uint8)

        # Декодування в OpenCV формат
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Не вдалося декодувати зображення")

        # Обробка зображення (використовуємо ідентичну логіку)
        result = detect_license_plate(img, models["yolo"], models["ocr"])
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
