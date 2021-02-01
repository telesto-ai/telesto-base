from dataclasses import dataclass
from typing import Dict, List
import base64
import io

import PIL.Image
import numpy as np


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


def convert_base64_images_to_arrays(doc: Dict) -> List[np.ndarray]:
    input_list = []
    for image_doc in doc["images"]:
        image_bytes = base64.b64decode(image_doc["content"])
        image = PIL.Image.open(io.BytesIO(image_bytes))
        expected_modes = ["RGB", "L"]
        if image.mode not in expected_modes:
            raise ValueError(f"Wrong image mode: {image.mode}. Expected: {expected_modes}")

        input_list.append(np.asarray(image))
    return input_list


def preprocess_base64_images(doc: Dict, max_images: int = 32) -> List[np.ndarray]:
    input_list = convert_base64_images_to_arrays(doc)
    if not (0 < len(input_list) <= max_images):
        raise ValueError(f"Wrong number of images: {len(input_list)}. Expected: 1 - {max_images}")

    return input_list
