import json
from typing import List, Dict

import falcon

from telesto.config import config
from telesto.resources.simple import RootResource, DocsResource
from telesto.object_detection.model import DummyObjectDetectionModel
from telesto.object_detection import DetectionObject
from telesto.utils import preprocess_base64_images

PNG_RGB_GRAY8_BASE64_FORMAT = {
    "type": "png",
    "palette": "RGB24 or GRAY8",
    "encoding": "base64",
    "max_size": "5120",
}

PREDICT_ENDPOINT_DOCS = {
    "path": "/",
    "method": "POST",
    "name": "Predict endpoint",
    "request_body": {
        "images": [
            {
                "content": "<str>"
            }
        ]
    },
    "image_format": {
        **PNG_RGB_GRAY8_BASE64_FORMAT,
    },
    "response_body": {
        "predictions": [
            {
                "objects": [
                    {
                        "x": "<int>",
                        "y": "<int>",
                        "w": "<int>",
                        "h": "<int>",
                    }
                ]
            }
        ]
    },
    # "classes": config.get("common", "classes").split(","),
}


def postprocess(predictions: List[List[DetectionObject]]) -> Dict[str, List[Dict]]:
    return {
        "predictions": [
            {
                "objects": [obj.asdict() for obj in objects]
            }
            for objects in predictions
        ]
    }


def post_handler(model_wrapper, req: falcon.Request, resp: falcon.Response):
    req_doc = json.load(req.bounded_stream)
    input_data = preprocess_base64_images(req_doc)

    pred_data = model_wrapper.predict(input_data)

    resp_doc = postprocess(pred_data)
    resp.body = json.dumps(resp_doc)


def add_routes(api: falcon.API):
    api.add_route(
        "/", RootResource(post_handler, "ObjectDetectionModel", DummyObjectDetectionModel)
    )
    api.add_route("/docs", DocsResource(PREDICT_ENDPOINT_DOCS))
