from typing import List

import PIL.Image
import numpy as np
import pytest

from telesto.instance_segmentation import DataStorage, DetectionObject
from telesto.instance_segmentation.model import SegmentationModelBase


@pytest.fixture(scope="session")
def storage():
    storage = DataStorage()
    yield storage
    storage.clean()


test_object = DetectionObject(coords=[(0, 0)])


class SegmentationModelTest(SegmentationModelBase):
    def __init__(self, storage: DataStorage):
        super().__init__(classes=["fg", "bg"], model_path='', storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> List[DetectionObject]:
        return [test_object]


def test_segmentation_model_base_call(storage: DataStorage):
    model = SegmentationModelTest(storage)

    job_id = "abc"
    array = np.array([[0, 0], [1, 1], [2, 2]])
    image = PIL.Image.fromarray(array.astype(np.uint8))
    storage.save(job_id, image, output=False)

    model(job_id)

    objects = storage.load(job_id, output=True)
    assert objects == [test_object]
