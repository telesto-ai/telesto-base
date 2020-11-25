import base64
import io
import os
import json
import socket
from importlib import import_module
from typing import List

import falcon
import numpy as np
import PIL.Image

from telesto.logger import logger
from telesto.models import RandomClassificationModel


def preprocess(doc: dict) -> List[np.ndarray]:
    input_list = []
    for image_doc in doc["images"]:
        image_bytes = base64.b64decode(image_doc["content"])
        image = PIL.Image.open(io.BytesIO(image_bytes))
        array = np.array(image)
        assert array.ndim in [2, 3], f"Wrong number of dimensions: {array.ndim}"

        input_list.append(array)
    return input_list


def postprocess(pred_array: np.ndarray, classes: List[str]) -> dict:
    predictions = []
    for pred in pred_array:
        class_probs = {classes[i]: round(float(prob), 5) for i, prob in enumerate(pred)}
        class_prediction = classes[pred.argmax()]
        predictions.append({"probs": class_probs, "prediction": class_prediction})
    return {"predictions": predictions}


class ClassificationResource:
    def __init__(self):
        try:
            self.model_wrapper = self._load_model()
        except ModuleNotFoundError as e:
            if int(os.environ.get("USE_FALLBACK_MODEL", 0)):
                logger.warning(
                    "No model module found. Using fallback model 'RandomClassificationModel'"
                )
                self.model_wrapper = RandomClassificationModel()
            else:
                raise e

    @staticmethod
    def _load_model():
        module = import_module("model")
        model_class = getattr(module, "ClassificationModel")
        return model_class()

    def on_get(self, req: falcon.Request, resp: falcon.Response):
        doc = {"status": "ok", "host": socket.getfqdn(), "worker.pid": os.getpid()}
        resp.body = json.dumps(doc, ensure_ascii=False)

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        try:
            req_doc = json.load(req.bounded_stream)
            input_list = preprocess(req_doc)
            pred_array = self.model_wrapper(input_list)
            resp_doc = postprocess(pred_array, self.model_wrapper.classes)
            resp.body = json.dumps(resp_doc)
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


def add_routes(api: falcon.API):
    api.add_route("/", ClassificationResource())
