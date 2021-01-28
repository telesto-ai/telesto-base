import pickle
import shutil
from pathlib import Path
from typing import List, Any, Dict, Tuple

import numpy as np

from telesto.utils import BBox, BaseObject


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


class SegmentationObject(BaseObject):
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
        return f"<SegmentationObject: bbox={self.bbox} coord_n={len(self.coords)}>"

    def __eq__(self, other: "SegmentationObject"):
        return other.coords == self.coords and other.bbox == self.bbox

    def asdict(self, image_size: Tuple[int, int]) -> Dict:
        dic = {
            "x": self.bbox.x1,
            "y": self.bbox.y1,
            "w": self.bbox.x2 - self.bbox.x1 + 1,
            "h": self.bbox.y2 - self.bbox.y1 + 1,
            "mask": rle_encode(self.coords, image_size)
        }
        return dic


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
