import os
import socket
from importlib import import_module
import json
import copy

import falcon

from telesto.config import config
from telesto.logger import logger


API_DOCS = {
    "name": config.get("common", "name"),
    "description": config.get("common", "desc"),
    "authentication": {
        "header": "Authorization",
        "schema": "Bearer <API_KEY>"
    },
    "endpoints": [
        {
            "path": "/",
            "method": "GET",
            "name": "Status endpoint",
        },

        {
            "path": "/docs",
            "method": "GET",
            "name": "Documentation endpoint",
        }
    ]
}


class RootResource:
    def __init__(self, post_handler, model_class, FallbackModelClass):
        self.post_handler = post_handler
        try:
            module = import_module("model")
            self.model_wrapper = getattr(module, model_class)()
        except ModuleNotFoundError as e:
            if int(os.environ.get("USE_FALLBACK_MODEL", 0)):
                logger.warning(
                    f"No model module found. Using fallback model '{FallbackModelClass}'"
                )
                self.model_wrapper = FallbackModelClass()
            else:
                raise e

    def on_get(self, req: falcon.Request, resp: falcon.Response):
        doc = {"status": "ok", "host": socket.getfqdn(), "worker.pid": os.getpid()}
        resp.body = json.dumps(doc, ensure_ascii=False)

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        try:
            self.post_handler(self.model_wrapper, req, resp)
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


class DocsResource:
    def __init__(self, *extra_endpoint_docs):
        self._api_docs = copy.deepcopy(API_DOCS)
        self._api_docs["endpoints"].extend(extra_endpoint_docs)

    def on_get(self, req, resp):
        resp.body = json.dumps(self._api_docs, ensure_ascii=False)
