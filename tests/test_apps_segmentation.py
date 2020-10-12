import os
import base64
import io
import json
import time

import falcon
from falcon import testing
import pytest
import numpy as np
from PIL import Image

from app import get_app
from telesto.apps.segmentation import ImageStorage, image_to_base64

os.environ["USE_FALLBACK_MODEL"] = "1"


@pytest.fixture
def client():
    return testing.TestClient(get_app())


@pytest.fixture(scope="module")
def storage():
    storage = ImageStorage()
    yield storage
    storage.clean()


def test_image_storage_save_load(storage: ImageStorage):
    array = np.array([[0, 0], [1, 1]])
    image = Image.fromarray(array, mode="L")

    gid = "abc"
    storage.save(gid, image, is_mask=False)
    restored_image = storage.load(gid, is_mask=False)

    assert np.all(np.asarray(image) == np.asarray(restored_image))


def test_image_storage_not_found(storage: ImageStorage):
    restored_image = storage.load(gid="xyz", is_mask=False)

    assert restored_image is None


def test_root_get(client: testing.TestClient):
    resp = client.simulate_get('/')

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    assert resp_doc["status"] == "ok"


def test_segm_get(client: testing.TestClient, storage: ImageStorage):
    array = np.array([[0, 0], [1, 1]])
    image = Image.fromarray(array, mode="L")
    job_id = "abc"
    storage.save(job_id, image, is_mask=True)

    resp = client.simulate_get(f"/jobs/{job_id}")

    assert resp.status == falcon.HTTP_OK

    resp_doc = json.loads(resp.content)
    image_bytes = base64.b64decode(resp_doc["mask"]["content"])
    resp_image = Image.open(io.BytesIO(image_bytes))
    assert np.all(np.asarray(image) == np.asarray(resp_image))


def test_segm_post_get(client: testing.TestClient):
    array = np.array([[0, 0], [1, 1]])
    image = Image.fromarray(array, mode="L")
    image_b64 = image_to_base64(image)
    req_doc = {"image": {"content": image_b64.decode()}}

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
    assert "mask" in resp_doc and "content" in resp_doc["mask"], resp_doc
