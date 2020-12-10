import os
import base64
import io
import json
import time

import falcon
from falcon import testing
import pytest
import numpy as np
import PIL.Image

from telesto.app import get_app
from telesto.config import config
from telesto.models import ModelType
from telesto.instance_segmentation import DataStorage, SegmentationObject

os.environ["USE_FALLBACK_MODEL"] = "1"


@pytest.fixture(scope="session", autouse=True)
def config_fixture():
    config["common"]["model_type"] = ModelType.INSTANCE_SEGMENTATION.value
    config["common"]["api_key"] = ""


@pytest.fixture
def client():
    return testing.TestClient(get_app())


@pytest.fixture(scope="session")
def storage():
    storage = DataStorage()
    yield storage
    storage.clean()


def make_test_image(rgb=False):
    array = np.array([[0, 0], [1, 1], [2, 2]])
    if rgb:
        array = np.stack([array] * 3, axis=2)
    image = PIL.Image.fromarray(array.astype(np.uint8))
    return image


def test_image_storage_save_load(storage: DataStorage):
    image = make_test_image()

    gid = "abc"
    storage.save(gid, image, output=False)
    restored_image = storage.load(gid, output=False)

    assert np.all(np.asarray(image) == np.asarray(restored_image))


def test_image_storage_not_found(storage: DataStorage):
    restored_image = storage.load(gid="xyz", output=False)

    assert restored_image is None


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


def test_segm_get(client: testing.TestClient, storage: DataStorage):
    job_id = "bcd"
    test_segm_object = SegmentationObject(class_i=1, x=0, y=0, w=1, h=1, mask=[[]])
    storage.save(job_id, [test_segm_object], output=True)

    resp = client.simulate_get(f"/jobs/{job_id}")

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert SegmentationObject(**resp_doc["objects"][0]) == test_segm_object


def test_segm_post_get(client: testing.TestClient):
    image = make_test_image(rgb=True)
    fp = io.BytesIO()
    image.save(fp, format="PNG")
    fp.seek(0)
    req_doc = {"image": base64.b64encode(fp.read()).decode()}

    resp = client.simulate_post(
        "/jobs", body=json.dumps(req_doc), headers={"content-type": "application/json"}
    )

    assert resp.status == falcon.HTTP_CREATED, resp.text

    resp_doc = json.loads(resp.content)
    assert resp_doc["job_id"], resp_doc

    resp = client.simulate_get(f"/jobs/{resp_doc['job_id']}")
    assert resp.status == falcon.HTTP_404, resp.text

    time.sleep(1)

    resp = client.simulate_get(f"/jobs/{resp_doc['job_id']}")
    assert resp.status == falcon.HTTP_200, resp.text

    resp_doc = json.loads(resp.content)
    assert "objects" in resp_doc, resp_doc


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
