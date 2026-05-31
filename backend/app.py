import io
import os
import time
import uuid
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from schemas import ImagePayload
from utils import ensure_directories, validate_image_filename, load_image_from_bytes, decode_base64_image
from models import ModelManager

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

ensure_directories(str(BASE_DIR))

app = FastAPI(
    title="Lemon Disease Detection API",
    description="FastAPI backend for plant disease classification and YOLOv10 object detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_manager = ModelManager(models_dir=str(MODEL_DIR))

class PredictResponse(BaseModel):
    disease: str
    confidence: float
    bounding_boxes: list
    model_used: str
    inference_time: str
    uploaded_image: str


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
async def predict_image(
    image_file: UploadFile | None = File(default=None),
    payload: ImagePayload | None = Body(default=None),
):
    if image_file is None and payload is None:
        raise HTTPException(status_code=400, detail="No image was provided")

    if image_file is not None:
        if not validate_image_filename(image_file.filename):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        contents = await image_file.read()
        filename = f"{uuid.uuid4().hex}_{image_file.filename}"
    else:
        contents = decode_base64_image(payload.image)
        filename = f"{uuid.uuid4().hex}.png"

    upload_path = UPLOAD_DIR / filename
    upload_path.write_bytes(contents)

    image = load_image_from_bytes(contents)

    start_time = time.time()
    detections = model_manager.detect(image)
    label_index, confidence, raw_predictions = model_manager.classify(image)
    disease_name = model_manager.get_class_label(label_index)

    inference_time = f"{time.time() - start_time:.2f} sec"
    response = {
        "disease": disease_name,
        "confidence": round(confidence, 4),
        "bounding_boxes": detections,
        "model_used": "DenseNet121.keras + YOLOv10",
        "inference_time": inference_time,
        "uploaded_image": f"/uploads/{filename}",
    }
    return JSONResponse(response)


@app.get("/capture", response_model=PredictResponse)
async def capture_image():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise HTTPException(status_code=500, detail="Unable to open camera")

    ret, frame = capture.read()
    capture.release()
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Camera capture failed")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    contents = buffer.getvalue()
    filename = f"camera_{uuid.uuid4().hex}.png"

    upload_path = UPLOAD_DIR / filename
    upload_path.write_bytes(contents)

    start_time = time.time()
    detections = model_manager.detect(image)
    label_index, confidence, raw_predictions = model_manager.classify(image)
    disease_name = model_manager.get_class_label(label_index)

    inference_time = f"{time.time() - start_time:.2f} sec"
    response = {
        "disease": disease_name,
        "confidence": round(confidence, 4),
        "bounding_boxes": detections,
        "model_used": "DenseNet121.keras + YOLOv10",
        "inference_time": inference_time,
        "uploaded_image": f"/uploads/{filename}",
    }
    return JSONResponse(response)


@app.get("/uploads/{image_name}")
async def get_uploaded_image(image_name: str):
    file_path = UPLOAD_DIR / image_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded image not found")
    return FileResponse(file_path)


