import os
import json
import socket
from importlib import import_module

import falcon

from telesto.logger import logger
from telesto.models import RandomClassificationModel


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
            resp_doc = self.model_wrapper(req_doc)
            resp.body = json.dumps(resp_doc)
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


def add_routes(api: falcon.API):
    api.add_route("/", ClassificationResource())
