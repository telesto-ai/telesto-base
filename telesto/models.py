import io
from typing import List
import base64
import random

import numpy as np
from PIL import Image


class ClassificationModelBase:
    """
    Base class for the model to be served.

    Attributes:
        classes: list, contains the labels
        model_path: str, path to the model file
        model: the object representing the model, to be loaded with _load_model()
    """
    def __init__(self, classes: List[str], model_path: str):
        print(model_path)
        self.classes: List[str] = classes
        self.class_n: int = len(classes)
        self.model_path: str = model_path
        self.model = self._load_model(model_path=model_path)

    def _load_model(self, model_path: str):
        raise NotImplemented

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        raise NotImplemented

    def _preprocess(self, doc):
        input_list = []
        for image_doc in doc["images"]:
            image_bytes = base64.b64decode(image_doc["content"])
            image = Image.open(io.BytesIO(image_bytes))

            array = np.array(image)
            assert array.ndim in [2, 3], f"Wrong number of dimensions: {array.ndim}"

            input_list.append(array)
        return input_list

    def _postprocess(self, pred_array):
        predictions = []
        for pred in pred_array:
            class_probs = {self.classes[i]: float(prob) for i, prob in enumerate(pred)}
            class_prediction = self.classes[pred.argmax()]
            predictions.append({"probs": class_probs, "prediction": class_prediction})
        return {"predictions": predictions}

    def __call__(self, in_doc):
        input_list = self._preprocess(in_doc)
        if not (0 < len(input_list) <= 32):
            raise ValueError(f"Wrong number of images: {len(input_list)}")

        pred_array = self.predict(input_list)
        out_doc = self._postprocess(pred_array)
        return out_doc


class RandomModel(ClassificationModelBase):
    def __init__(self):
        super().__init__(classes=["cat", "dog"], model_path='')

    def _load_model(self, model_path: str):
        pass

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        batch_size = len(input_list)
        return np.array([[random.random() for _ in range(self.class_n)] for _ in range(batch_size)])
