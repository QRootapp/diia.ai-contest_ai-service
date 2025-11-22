import uvicorn
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR

# --- ГЛОБАЛЬНІ ЗМІННІ ---
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Завантаження OCR моделі...")
    try:
        models["ocr"] = PaddleOCR(text_recognition_model_dir='train_models/OCR')
        print("OCR модель успішно завантажено.")
    except Exception as e:
        print(f"Помилка завантаження OCR: {e}")
    
    yield
    
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/recognize_text")
async def recognize_text(file: UploadFile = File(...)):
    """
    Розпізнавання тексту на зображенні номерного знаку.
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
        
        # OCR розпізнавання
        ocr_out = models["ocr"].predict(img)
        
        fragments = []
        if ocr_out and isinstance(ocr_out, list):
            rec = ocr_out[0]
            texts = rec.get('rec_texts', [])
            scores = rec.get('rec_scores', [])
            for txt, score in zip(texts, scores):
                if txt and score > 0.3:
                    fragments.append({"text": txt, "confidence": score})
        
        return {"fragments": fragments}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка OCR: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
