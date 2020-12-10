from typing import List
import random

import numpy as np


class ClassificationModelBase:
    """Base class for a classification model wrapper.

    Attributes:
        classes (list): contains the labels
        model: the object representing the model, to be loaded with _load_model()
    """

    def __init__(self, classes: List[str], model_path: str):
        self.classes: List[str] = classes
        self.model = self._load_model(model_path=model_path)

    def _load_model(self, model_path: str) -> object:
        """Load model weights from path, create and return a model object."""

        raise NotImplemented

    def predict(self, input_list: List[np.ndarray]) -> np.ndarray:
        """Classify a list of input images and return an array of class probabilities.

        Args:
            input_list: list of input images, each image is a 3D array
                with a number of channels

        Returns:
            2D array of class probabilities, 0 dim - images, 1 dim - classes
        """
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
