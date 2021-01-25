import os

import falcon
from falcon import testing

from telesto.config import config
from telesto.app import get_app

os.environ["USE_FALLBACK_MODEL"] = "1"


def test_api_key_auth_error():
    config["common"]["api_key"] = "API_KEY"

    client = testing.TestClient(get_app())
    resp = client.simulate_get("/")

    assert resp.status == falcon.HTTP_401


def test_api_key_auth_ok():
    config["common"]["api_key"] = "API_KEY"

    client = testing.TestClient(get_app())
    resp = client.simulate_get("/", headers={"Authorization": "Bearer API_KEY"})

    assert resp.status == falcon.HTTP_OK
