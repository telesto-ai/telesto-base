import os
import json
import socket
import logging
from importlib import import_module

import falcon

logger = logging.getLogger("telesto")

gunicorn_logger = logging.getLogger("gunicorn")
if gunicorn_logger.handlers:
    logger.handlers = gunicorn_logger.handlers
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class PredictResource:
    def __init__(self):
        self.model_wrapper = self._load_model()

    @staticmethod
    def _load_model():
        module_name = "model"
        module = import_module(module_name)
        class_name = "ClassificationModel"
        model_class = getattr(module, class_name)
        return model_class()

    def on_get(self, req, resp):
        doc = {"status": "ok", "host": socket.getfqdn(), "worker.pid": os.getpid()}
        resp.body = json.dumps(doc, ensure_ascii=False)

    def on_post(self, req, resp):
        try:
            req_doc = json.load(req.bounded_stream)
            resp_doc = self.model_wrapper(req_doc)
            resp.body = json.dumps(resp_doc)
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


def get_app():
    api = falcon.API()
    api.add_route("/", PredictResource())
    return api


app = get_app()

if __name__ == "__main__":
    logger.info("Starting API server...")

    from wsgiref import simple_server

    httpd = simple_server.make_server("0.0.0.0", 9876, app)
    logger.info("API server started")
    httpd.serve_forever()
