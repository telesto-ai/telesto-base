from dataclasses import dataclass
from typing import Dict


@dataclass
class BBox:
    """Bounding box dataclass.

    (x1, y1) - top left corner, (x2, y2) - bottom right one
    """

    x1: int
    y1: int
    x2: int
    y2: int


class BaseObject:

    def asdict(self, *args, **kwargs) -> Dict:
        raise NotImplementedError
