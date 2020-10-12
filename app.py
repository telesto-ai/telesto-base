import sys

import falcon
from falcon.http_status import HTTPStatus

from telesto.logger import logger
from telesto.models import ModelTypes


class HandleCORS(object):
    def process_request(self, req, resp):
        resp.set_header("Access-Control-Allow-Origin", "*")
        resp.set_header("Access-Control-Allow-Methods", "*")
        resp.set_header("Access-Control-Allow-Headers", "*")
        resp.set_header("Access-Control-Max-Age", 1728000)  # 20 days
        if req.method == "OPTIONS":
            raise HTTPStatus(falcon.HTTP_200, body="\n")


def get_app(model_type: ModelTypes = ModelTypes.CLASSIFICATION):
    api = falcon.API(middleware=[HandleCORS()])

    if model_type == ModelTypes.SEGMENTATION:
        from telesto.apps.segmentation import add_routes
    elif model_type == ModelTypes.CLASSIFICATION:
        from telesto.apps.classification import add_routes
    else:
        raise Exception(f"Wrong model type: {model_type}")

    add_routes(api)
    return api


if __name__ == "__main__":
    logger.info("Starting dev API server...")

    from wsgiref import simple_server

    model_type = ModelTypes(sys.argv[1])
    httpd = simple_server.make_server("0.0.0.0", 9876, get_app(model_type))
    logger.info("Dev API server started")
    httpd.serve_forever()
