from __future__ import annotations
from typing import List

import numpy as np

from telesto.instance_segmentation import DataStorage, SegmentationObject


class SegmentationModelBase:
    """Base class for an instance segmentation model wrapper.

    Attributes:
        classes (list): contains the labels
        model: the object representing the model, to be loaded with _load_model()
    """

    def __init__(self, classes: List[str], model_path: str, storage: DataStorage):
        self._storage = storage
        self.model = self._load_model(model_path=model_path)
        self.classes = classes

    def _load_model(self, model_path: str) -> object:
        """Load model weights from path, create and return a model object."""

        raise NotImplemented

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        """Segment input image and return a list of found objects.

        Args:
            input: input image, 3D array with 1 or 3 image channels

        Returns:
            list of found objects
        """
        raise NotImplemented

    def __call__(self, job_id: str):
        image = self._storage.load(job_id, output=False)
        objects = self.predict(np.asarray(image))
        self._storage.save(job_id, objects, output=True)


class DummySegmentationModel(SegmentationModelBase):
    def __init__(self, storage: DataStorage):
        super().__init__(classes=["fg", "bg"], model_path="", storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        return [SegmentationObject(class_i=1, x=10, y=10, w=2, h=2, mask=[[0, 1], [0, 1]])]
