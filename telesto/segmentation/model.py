from __future__ import annotations
from typing import List

import numpy as np

from telesto.segmentation import DataStorage, SegmentationObject


class SegmentationModelBase:

    def __init__(self, classes: List[str], model_path: str, storage: DataStorage):
        self._storage = storage
        self.model = self._load_model(model_path=model_path)
        self.classes = classes
        self.class_n = len(classes)

    def _load_model(self, model_path: str) -> object:
        raise NotImplemented

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        raise NotImplemented

    def __call__(self, job_id: str):
        image = self._storage.load(job_id, output=False)
        objects = self.predict(np.asarray(image))
        self._storage.save(job_id, objects, output=True)


class DummySegmentationModel(SegmentationModelBase):
    def __init__(self, storage: DataStorage):
        super().__init__(classes=["fg", "bg"], model_path='', storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        return [SegmentationObject(class_i=1, x=10, y=10, w=2, h=2, data=[[0, 1], [0, 1]])]
