from enum import Enum

from .classification import ClassificationModelBase, RandomClassificationModel
from .segmentation import SegmentationModelBase, DummySegmentationModel


class ModelTypes(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
