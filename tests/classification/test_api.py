import base64
import io
import json

import falcon
import pytest
from falcon import testing
import numpy as np
import PIL.Image

from telesto.config import config
from telesto.models import ModelType


@pytest.fixture(scope="session", autouse=True)
def config_fixture():
    config["common"]["model_type"] = ModelType.CLASSIFICATION.value
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
    req_doc = {"images": [{"content": base64.b64encode(fp.read()).decode()}]}

    resp = client.simulate_post(
        "/", body=json.dumps(req_doc), headers={"content-type": "application/json"}
    )

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert "predictions" in resp_doc

    for pred_doc in resp_doc["predictions"]:
        assert isinstance(pred_doc["class"], str)

        for prob_doc in pred_doc["probs"]:
            assert isinstance(prob_doc["class"], str)
            assert 0 <= prob_doc["prob"] <= 1
