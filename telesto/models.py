from enum import Enum

from telesto.classification.model import ClassificationModelBase, RandomClassificationModel
from telesto.segmentation.model import SegmentationModelBase, DummySegmentationModel


class ModelTypes(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
