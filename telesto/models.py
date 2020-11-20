from enum import Enum

from .classification.model import ClassificationModelBase, RandomClassificationModel
from .segmentation.model import SegmentationModelBase, DummySegmentationModel


class ModelTypes(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
