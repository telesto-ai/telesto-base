import falcon
from falcon.http_status import HTTPStatus

from telesto.logger import logger
from telesto.config import config
from telesto.models import ModelType


class HandleCORS(object):
    def process_request(self, req, resp):
        resp.set_header("Access-Control-Allow-Origin", "*")
        resp.set_header("Access-Control-Allow-Methods", "*")
        resp.set_header("Access-Control-Allow-Headers", "*")
        resp.set_header("Access-Control-Max-Age", 1728000)  # 20 days
        if req.method == "OPTIONS":
            raise HTTPStatus(falcon.HTTP_200, body="\n")


class AuthMiddleware(object):

    @staticmethod
    def _api_key_is_valid(auth):
        try:
            _, api_key = auth.split(" ")
            assert config.get("common", "api_key") == api_key, "API key mismatch"
        except Exception as e:
            logger.warning(f"Auth error: {e}")
            return False

        return True

    def process_request(self, req, resp):
        challenges = ["Bearer realm='Access to protected endpoint'"]
        auth = req.get_header("Authorization")

        if auth is None:
            description = "Please provide an API key as part of the request."
            raise falcon.HTTPUnauthorized("API key required", description, challenges)

        if not self._api_key_is_valid(auth):
            description = (
                "The provided API key is not valid. Please request a API key and try again."
            )
            raise falcon.HTTPUnauthorized("Authentication required", description, challenges)


def get_app():
    middleware = [HandleCORS()]
    if config.get("common", "api_key"):
        middleware.append(AuthMiddleware())
    api = falcon.API(middleware=middleware)

    model_type = config.get("common", "model_type")
    if model_type == ModelType.INSTANCE_SEGMENTATION:
        from telesto.instance_segmentation.app import add_routes
    elif model_type == ModelType.CLASSIFICATION:
        from telesto.classification.app import add_routes
    else:
        raise Exception(f"Wrong model type: {model_type}")

    add_routes(api)
    return api


if __name__ == "__main__":
    logger.info("Starting dev API server...")

    from wsgiref import simple_server

    httpd = simple_server.make_server("0.0.0.0", 9876, get_app())
    logger.info("Dev API server started")
    httpd.serve_forever()
