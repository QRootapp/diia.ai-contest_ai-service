import uvicorn
import cv2
import numpy as np
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# --- ГЛОБАЛЬНІ ЗМІННІ ---
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Завантаження YOLO моделі...")
    try:
        models["yolo"] = YOLO('train_models/YOLO/my_YOLO_detection_car_plates.pt')
        print("YOLO модель успішно завантажено.")
    except Exception as e:
        print(f"Помилка завантаження YOLO: {e}")
    
    yield
    
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/detect_plates")
async def detect_plates(file: UploadFile = File(...)):
    """
    Детекція номерних знаків на зображенні.
    Повертає координати та crop зображення номерів.
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
        
        # YOLO детекція
        results = models["yolo"](img, verbose=False, iou=0.5, conf=0.3)
        
        plate_crops = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                
                # Препроцесинг
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if gray.shape[0] < 80:
                    gray = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), 
                                    interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                crop_processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Кодування crop в base64
                _, buffer = cv2.imencode('.jpg', crop_processed)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                plate_crops.append({
                    "bbox": [x1, y1, x2, y2],
                    "image": crop_base64
                })
        
        return {"plate_crops": plate_crops}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка YOLO: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
