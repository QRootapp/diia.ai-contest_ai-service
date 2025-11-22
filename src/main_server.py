import uvicorn
import cv2
import numpy as np
import re
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# URLs мікросервісів
YOLO_SERVICE_URL = "http://localhost:8001/detect_plates"
OCR_SERVICE_URL = "http://localhost:8002/recognize_text"

app = FastAPI()

# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def correct_plate_text(text):
    allowed_letters = 'ABCEHIKMOPTXDUY'
    standard_pattern = fr'^\[{allowed_letters}\]{{2}}\\d{{4}}\[{allowed_letters}\]{{2}}$'
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

# --- API ЕНДПОІНТ ---

@app.post("/detect")
async def detect_license_plate_endpoint(file: UploadFile = File(...)):
    """
    Основний ендпоінт для обробки зображення.
    Координує роботу YOLO та OCR сервісів.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл має бути зображенням")
    
    try:
        # Читання файлу
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Не вдалося декодувати зображення")
        
        # Кодування зображення для відправки
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        
        # 1. Відправка в YOLO сервіс
        async with httpx.AsyncClient(timeout=30.0) as client:
            yolo_response = await client.post(
                YOLO_SERVICE_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")}
            )
            
            if yolo_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Помилка YOLO сервісу")
            
            yolo_data = yolo_response.json()
            plate_crops = yolo_data.get("plate_crops", [])
        
        detected_cars = []
        
        # 2. Відправка кожного crop в OCR сервіс
        async with httpx.AsyncClient(timeout=30.0) as client:
            for crop_data in plate_crops:
                # Декодування crop з base64
                import base64
                crop_bytes = base64.b64decode(crop_data["image"])
                
                # Відправка в OCR
                ocr_response = await client.post(
                    OCR_SERVICE_URL,
                    files={"file": ("crop.jpg", crop_bytes, "image/jpeg")}
                )
                
                if ocr_response.status_code == 200:
                    ocr_data = ocr_response.json()
                    fragments = ocr_data.get("fragments", [])
                    
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
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
