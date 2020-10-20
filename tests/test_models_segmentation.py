import PIL.Image
import numpy as np
import pytest

from apps.segmentation import ImageStorage
from models import SegmentationModelBase


@pytest.fixture(scope="session")
def storage():
    storage = ImageStorage()
    yield storage
    storage.clean()


class SegmentationModelTest(SegmentationModelBase):
    def __init__(self, storage: ImageStorage):
        super().__init__(classes=["fg", "bg"], model_path='', storage=storage)

    def _load_model(self, model_path: str):
        pass

    def predict(self, input: np.ndarray) -> np.ndarray:
        return np.array([[0, 0], [1, 1], [2, 2]])


def test_segmentation_model_base_call(storage: ImageStorage):
    model = SegmentationModelTest(storage)

    job_id = "abc"
    array = np.array([[0, 0], [1, 1], [2, 2]])
    image = PIL.Image.fromarray(array.astype(np.uint8))
    storage.save(job_id, image, is_mask=False)

    model(job_id)

    mask = storage.load(job_id, is_mask=True)
    assert np.all(np.asarray(mask) == np.array([[0, 0], [1, 1], [2, 2]]))
