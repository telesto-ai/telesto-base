from typing import List

import numpy as np

from telesto.instance_segmentation import SegmentationObject


class SegmentationModelBase:
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

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        """Segment input image and return a list of found objects.

        Args:
            input: input image, 3D array with 1 or 3 image channels

        Returns:
            list of found objects
        """
        raise NotImplemented


class DummySegmentationModel(SegmentationModelBase):
    def __init__(self):
        super().__init__(classes=["fg", "bg"], model_path="")

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[SegmentationObject]:
        return [SegmentationObject(coords=[(0, 0), (1, 1)])]
