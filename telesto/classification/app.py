import base64
import io
import json
from typing import List, Dict

import falcon
import numpy as np
import PIL.Image

from telesto.config import config
from telesto.resources.simple import RootResource, DocsResource
from telesto.models import RandomClassificationModel

PNG_RGB_BASE64_FORMAT = {
    "type": "png",
    "palette": "RGB24 or GRAY8",
    "encoding": "base64",
    "max_size": "1024",
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
        ],
    },
    "image_content_format": {
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


def preprocess(doc: Dict) -> List[np.ndarray]:
    input_list = []
    for image_doc in doc["images"]:
        image_bytes = base64.b64decode(image_doc["content"])
        image = PIL.Image.open(io.BytesIO(image_bytes))
        array = np.array(image)
        assert array.ndim in [2, 3], f"Wrong number of dimensions: {array.ndim}"

        input_list.append(array)

    if not (0 < len(input_list) <= 32):
        raise ValueError(f"Wrong number of images: {len(input_list)}: expected 1 - 32")
    return input_list


def postprocess(pred_array: np.ndarray, classes: List[str]) -> Dict:
    predictions = []
    for pred in pred_array:
        class_probs = {classes[i]: round(float(prob), 5) for i, prob in enumerate(pred)}
        class_prediction = classes[pred.argmax()]
        predictions.append({"probs": class_probs, "prediction": class_prediction})
    return {"predictions": predictions}


def post_handler(model_wrapper, req: falcon.Request, resp: falcon.Response):
    req_doc = json.load(req.bounded_stream)
    input_data = preprocess(req_doc)

    pred_data = model_wrapper.predict(input_data)

    resp_doc = postprocess(pred_data, model_wrapper.classes)
    resp.body = json.dumps(resp_doc)


def add_routes(api: falcon.API):
    api.add_route("/", RootResource(post_handler, "ClassificationModel", RandomClassificationModel))
    api.add_route("/docs", DocsResource(PREDICT_ENDPOINT_DOCS))
