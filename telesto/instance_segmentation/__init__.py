import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Dict, Tuple

import numpy as np


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
class BBox:
    """Bounding box dataclass.

    (x1, y1) - top left corner, (x2, y2) - bottom right one
    """

    x1: int
    y1: int
    x2: int
    y2: int


class DetectionObject:
    """
    Attributes:
        coords: list of (x, y) coordinates
        bbox: object bounding box
    """

    def __init__(self, coords: List[Tuple[int, int]]):
        """
        Args:
            coords: list of (x, y) coordinates where object mask == 1
        """
        if not coords:
            raise ValueError("'coords' argument cannot be empty")

        self.coords = coords
        xs, ys = zip(*coords)
        self.bbox = BBox(min(xs), min(ys), max(xs), max(ys))

    def __repr__(self):
        return f"<DetectionObject: bbox={self.bbox} coord_n={len(self.coords)}>"

    def __eq__(self, other: "DetectionObject"):
        return other.coords == self.coords and other.bbox == self.bbox


def rle_encode(coords: List[Tuple[int, int]], image_size: Tuple[int, int]) -> str:
    if not coords:
        return ""

    w, h = image_size
    image = np.zeros((h, w))
    xs, ys = zip(*coords)
    image[ys, xs] = 1
    image = image.flatten()
    image = np.insert(image, [0, len(image)], [0, 0])

    starts = ((image[:-1] == 0) & (image[1:] == 1)).nonzero()[0]
    ends = ((image[:-1] == 1) & (image[1:] == 0)).nonzero()[0]
    lengths = ends - starts

    return " ".join(f"{st} {l}" for st, l in zip(starts, lengths))


def segmentation_object_asdict(obj: DetectionObject, image_size: Tuple[int, int]) -> Dict:
    dic = {
        "x": obj.bbox.x1,
        "y": obj.bbox.y1,
        "w": obj.bbox.x2 - obj.bbox.x1 + 1,
        "h": obj.bbox.y2 - obj.bbox.y1 + 1,
        "mask": rle_encode(obj.coords, image_size)
    }
    return dic
