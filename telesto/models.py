from enum import Enum

from telesto.classification.model import ClassificationModelBase, RandomClassificationModel
from telesto.instance_segmentation.model import SegmentationModelBase, DummySegmentationModel


class ModelType(str, Enum):
    CLASSIFICATION = "CLASSIFICATION"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
