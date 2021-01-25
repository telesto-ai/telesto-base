import base64
import io
import json
from typing import List, Dict

import falcon
import numpy as np
import PIL.Image

from telesto.config import config
from telesto.resources.simple import RootResource, DocsResource
from telesto.object_detection.model import DummyObjectDetectionModel
from telesto.object_detection import DetectionObject


PNG_RGB_BASE64_FORMAT = {
    "type": "png",
    "palette": "RGB24",
    "encoding": "base64",
    "max_size": "5120",
}

PREDICT_ENDPOINT_DOCS = {
    "path": "/",
    "method": "POST",
    "name": "Predict endpoint",
    "request_body": {
        "image": "<str>",
    },
    "image_format": {
        **PNG_RGB_BASE64_FORMAT,
    },
    "response_body": {
        "objects": [
            {
                "x": "<int>",
                "y": "<int>",
                "w": "<int>",
                "h": "<int>",
                "mask": "<str>"
            }
        ],
    },
    "classes": config.get("common", "classes").split(","),
},


def preprocess(doc: Dict) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(doc["image"])
        image = PIL.Image.open(io.BytesIO(image_bytes))
        assert image.mode == "RGB", f"Wrong image mode: {image.mode}. Expected: 'RGB'"
    except Exception as e:
        raise ValueError(e)
    return np.asarray(image)


def postprocess(objects: List[DetectionObject]) -> Dict[str, List[Dict]]:
    return {"objects": [(obj.asdict()) for obj in objects]}


def post_handler(model_wrapper, req: falcon.Request, resp: falcon.Response):
    req_doc = json.load(req.bounded_stream)
    input_data = preprocess(req_doc)

    pred_data = model_wrapper.predict(input_data)

    resp_doc = postprocess(pred_data)
    resp.body = json.dumps(resp_doc)


def add_routes(api: falcon.API):
    api.add_route(
        "/", RootResource(post_handler, "ObjectDetectionModel", DummyObjectDetectionModel)
    )
    api.add_route("/docs", DocsResource(PREDICT_ENDPOINT_DOCS))
