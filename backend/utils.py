import os
import io
import base64
from PIL import Image
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def ensure_directories(base_path: str):
    for sub in ("uploads", "results", "models"):
        path = os.path.join(base_path, sub)
        os.makedirs(path, exist_ok=True)


def validate_image_filename(filename: str) -> bool:
    extension = filename.rsplit(".", 1)[-1].lower()
    return extension in ALLOWED_EXTENSIONS


def load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


def decode_base64_image(data_str: str) -> bytes:
    try:
        if data_str.startswith("data:"):
            data_str = data_str.split(",", 1)[1]
        return base64.b64decode(data_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode base64 image: {exc}")
