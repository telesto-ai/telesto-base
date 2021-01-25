import base64
import io
import json

import falcon
import pytest
from falcon import testing
import numpy as np
import PIL.Image

from telesto.config import config
from telesto.utils import BBox
from telesto.models import ModelType
from telesto.object_detection import DetectionObject


@pytest.fixture(scope="session", autouse=True)
def config_fixture():
    config["common"]["model_type"] = ModelType.OBJECT_DETECTION.value
    config["common"]["api_key"] = ""


def make_test_image(rgb=False):
    array = np.array([[0, 0], [1, 1], [2, 2]])
    if rgb:
        array = np.stack([array] * 3, axis=2)
    image = PIL.Image.fromarray(array.astype(np.uint8))
    return image


def test_root_get(client: testing.TestClient):
    resp = client.simulate_get("/")

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert resp_doc["status"] == "ok"


def test_docs_get(client: testing.TestClient):
    resp = client.simulate_get("/docs")

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert "endpoints" in resp_doc and resp_doc["endpoints"]


def test_root_post(client: testing.TestClient):
    test_image = make_test_image(rgb=True)
    fp = io.BytesIO()
    test_image.save(fp, format="PNG")
    fp.seek(0)
    req_doc = {"image": base64.b64encode(fp.read()).decode()}

    resp = client.simulate_post(
        "/", body=json.dumps(req_doc), headers={"content-type": "application/json"}
    )

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert resp_doc["objects"] == [DetectionObject(BBox(0, 0, 9, 9)).asdict()]
