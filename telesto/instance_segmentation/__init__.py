import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any


class DataStorage:

    def __init__(self, base_path: str = "./data/storage"):
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def clean(self):
        shutil.rmtree(self._base_path, ignore_errors=True)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _data_path(self, gid: str, output: bool) -> Path:
        type_ = "output" if output else "input"
        return self._base_path / f"{gid}-{type_}.pickle"

    def save(self, gid: str, obj: Any, output: bool):
        pickle.dump(obj, self._data_path(gid, output).open("wb"))

    def load(self, gid: str, output: bool) -> Any:
        image_path = self._data_path(gid, output)
        if image_path.exists():
            return pickle.load(image_path.open("rb"))


@dataclass
class SegmentationObject:
    """Instance segmentation object.

    Attributes:
        class_i: class index
        x: X coordinate of top left corner
        y: Y coordinate of top left corner
        w: object width
        h: object height
        mask: 2D mask of object pixels
    """

    class_i: int
    x: int
    y: int
    w: int
    h: int
    mask: List[List[int]]
