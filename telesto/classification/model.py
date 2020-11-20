from typing import List
import random

import numpy as np


class ClassificationModelBase:
    """
    Base class for the model to be served.

    Attributes:
        classes: list, contains the labels
        model_path: str, path to the model file
        model: the object representing the model, to be loaded with _load_model()
    """
    def __init__(self, classes: List[str], model_path: str):
        self.classes: List[str] = classes
        self.class_n: int = len(classes)
        self.model_path: str = model_path
        self.model = self._load_model(model_path=model_path)

    def _load_model(self, model_path: str) -> object:
        raise NotImplemented

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        raise NotImplemented

    def __call__(self, input_list: List[np.ndarray]) -> np.ndarray:
        if not (0 < len(input_list) <= 32):
            raise ValueError(f"Wrong number of images: {len(input_list)}")

        return self.predict(input_list)


class RandomClassificationModel(ClassificationModelBase):
    def __init__(self):
        super().__init__(classes=["cat", "dog"], model_path='')

    def _load_model(self, model_path: str):
        pass

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        batch_size = len(input_list)
        predictions = []
        for _ in range(batch_size):
            p = random.random()
            predictions.append([p, 1 - p])
        return np.array(predictions)
