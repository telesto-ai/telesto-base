from typing import List

import numpy as np

from telesto.object_detection import DetectionObject
from telesto.utils import BBox


class ObjectDetectionModelBase:
    """Base class for an instance segmentation model wrapper.

    Attributes:
        classes (list): contains the labels
        model: the object representing the model, to be loaded with _load_model()
    """

    def __init__(self, classes: List[str], model_path: str):
        self.model = self._load_model(model_path=model_path)
        self.classes = classes

    def _load_model(self, model_path: str) -> object:
        """Load model weights from path, create and return a model object."""

        raise NotImplemented

    def predict(self, input: List[np.ndarray]) -> List[List[DetectionObject]]:
        """Segment input image and return a list of found objects.

        Args:
            input: list of image arrays, 1 or 3 channels

        Returns:
            list of results for all input arrays
        """
        raise NotImplemented


class DummyObjectDetectionModel(ObjectDetectionModelBase):
    def __init__(self):
        super().__init__(classes=[], model_path="")

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: List[np.ndarray]) -> List[List[DetectionObject]]:
        return [[DetectionObject(BBox(0, 0, 9, 9))] for _ in input]
