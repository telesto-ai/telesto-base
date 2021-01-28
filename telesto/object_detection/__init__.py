from typing import Dict

from telesto.utils import BaseObject, BBox


class DetectionObject(BaseObject):
    """
    Attributes:
        bbox: object bounding box
    """

    def __init__(self, bbox: BBox):
        """
        Args:
            bbox: object bounding box
        """
        self.bbox = bbox

    def __repr__(self):
        return f"<DetectionObject: bbox={self.bbox}>"

    def asdict(self) -> Dict:
        dic = {
            "x": self.bbox.x1,
            "y": self.bbox.y1,
            "w": self.bbox.x2 - self.bbox.x1 + 1,
            "h": self.bbox.y2 - self.bbox.y1 + 1,
        }
        return dic
