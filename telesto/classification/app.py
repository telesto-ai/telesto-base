import json
from typing import List, Dict

import falcon
import numpy as np

from telesto.config import config
from telesto.resources.simple import RootResource, DocsResource
from telesto.models import DummyClassificationModel
from telesto.utils import convert_base64_images_to_arrays

PNG_RGB_GRAY8_BASE64_FORMAT = {
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
        **PNG_RGB_GRAY8_BASE64_FORMAT,
    },
    "response_body": {
        "predictions": [
            {
                "probs": [
                    {
                        "class": "<str>",
                        "prob": "<float>",
                    }
                ],
                "class": "<str>",
            }
        ],
    },
    "classes": config.get("common", "classes").split(","),
},


def preprocess(doc: Dict) -> List[np.ndarray]:
    input_list = convert_base64_images_to_arrays(doc)
    if not (0 < len(input_list) <= 32):
        raise ValueError(f"Wrong number of images: {len(input_list)}. Expected: 1 - 32")

    return input_list


def postprocess(pred_array: np.ndarray, classes: List[str]) -> Dict:
    predictions = []
    for pred in pred_array:
        class_probs = [
            {"class": classes[i], "prob": round(float(prob), 5)} for i, prob in enumerate(pred)
        ]
        predictions.append({"probs": class_probs, "class": classes[pred.argmax()]})
    return {"predictions": predictions}


def post_handler(model_wrapper, req: falcon.Request, resp: falcon.Response):
    req_doc = json.load(req.bounded_stream)
    input_data = preprocess(req_doc)

    pred_data = model_wrapper.predict(input_data)

    resp_doc = postprocess(pred_data, model_wrapper.classes)
    resp.body = json.dumps(resp_doc)


def add_routes(api: falcon.API):
    api.add_route("/", RootResource(post_handler, "ClassificationModel", DummyClassificationModel))
    api.add_route("/docs", DocsResource(PREDICT_ENDPOINT_DOCS))
