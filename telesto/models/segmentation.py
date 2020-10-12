from __future__ import annotations
from typing import List
from typing import TYPE_CHECKING

from PIL import Image
import numpy as np

if TYPE_CHECKING:
    from apps.segmentation import ImageStorage


class SegmentationModelBase:

    def __init__(self, classes: List[str], model_path: str, storage: ImageStorage):
        self._storage = storage
        self._model = self._load_model(model_path=model_path)
        self.classes = classes
        self.class_n = len(classes)

    def _load_model(self, model_path: str):
        raise NotImplemented

    def predict(self, input: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def __call__(self, job_id: str):
        image = self._storage.load(job_id, is_mask=False)

        mask_array = self.predict(np.asarray(image))
        mask = Image.fromarray(mask_array, mode="L")

        self._storage.save(job_id, mask, is_mask=True)


class DummySegmentationModel(SegmentationModelBase):
    def __init__(self, storage: ImageStorage):
        super().__init__(classes=["fg", "bg"], model_path='', storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 3:
            input = input[:, :, 0]
        return input
