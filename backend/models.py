import os
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

from keras.models import load_model
from keras.applications.densenet import preprocess_input


class ModelManager:

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

        self.classification_model = self._load_classification_model()
        self.object_detector = self._load_object_detector()

    def _load_classification_model(self):

        model_path = os.path.join(
            self.models_dir,
            "DenseNet121.keras"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Classification model not found at {model_path}"
            )

        return load_model(
            model_path,
            custom_objects={
                "preprocess_input": preprocess_input
            },
            compile=False,
            safe_mode=False
        )

    def _load_object_detector(self):

        model_path = os.path.join(
            self.models_dir,
            "best.pt"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}"
            )

        return YOLO(model_path)

    def preprocess_for_classification(
        self,
        image: Image.Image
    ) -> np.ndarray:

        image = image.convert("RGB")

        image = image.resize((224, 224))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    def classify(
        self,
        image: Image.Image
    ):

        input_tensor = self.preprocess_for_classification(
            image
        )

        predictions = self.classification_model.predict(
            input_tensor,
            verbose=0
        )

        prediction = predictions[0]

        top_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[top_index]
        )

        return (
            top_index,
            confidence,
            predictions.tolist()
        )

    def detect(
        self,
        image: Image.Image
    ):

        image = image.convert("RGB")

        results = self.object_detector(
            image,
            imgsz=640,
            verbose=False
        )

        detections = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            conf = float(
                box.conf.item()
            )

            cls = int(
                box.cls.item()
            )

            class_name = self.object_detector.names.get(
                cls,
                f"Class {cls}"
            )

            detections.append(
                {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "xmin": round(x1, 2),
                    "ymin": round(y1, 2),
                    "xmax": round(x2, 2),
                    "ymax": round(y2, 2)
                }
            )

        return detections

    def draw_disease_boxes(
        self,
        image: Image.Image
    ):

        image = image.convert("RGB")

        detections = self.detect(image)

        output_image = image.copy()

        draw = ImageDraw.Draw(output_image)

        for det in detections:

            x1 = det["xmin"]
            y1 = det["ymin"]
            x2 = det["xmax"]
            y2 = det["ymax"]

            class_id = det["class_id"]

            confidence = det["confidence"]

            class_name = self.object_detector.names.get(
                class_id,
                f"Class {class_id}"
            )

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=5
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            draw.text(
                (x1, max(0, y1 - 20)),
                label,
                fill="red"
            )

        return output_image

    def get_class_label(
        self,
        label_index: int
    ) -> str:

        labels = [
            "Black Spot",
            "Leaf Rust",
            "Powdery Mildew",
            "Healthy",
            "Bacterial Blight",
            "Nutrient Deficiency"
        ]

        if 0 <= label_index < len(labels):
            return labels[label_index]

        return f"Class {label_index}"