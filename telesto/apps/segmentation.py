import base64
import io
import os
import socket
import threading
import time
from importlib import import_module
from pathlib import Path
from typing import Type, Deque
from uuid import uuid4
import shutil
import json
from collections import deque

import PIL
from PIL.Image import Image
import falcon

from telesto.logger import logger
from telesto.models.segmentation import DummySegmentationModel, SegmentationModelBase


def preprocess(doc: dict) -> Image:
    image_bytes = base64.b64decode(doc["image"]["content"])
    image = PIL.Image.open(io.BytesIO(image_bytes))
    return image


def image_to_base64(image: Image) -> bytes:
    fp = io.BytesIO()
    image.save(fp, format="PNG")
    fp.seek(0)
    return base64.b64encode(fp.read())


def postprocess(image: Image) -> dict:
    image_b64 = image_to_base64(image)
    return {"mask": {"content": image_b64.decode()}}


class SegmentationBase:
    def on_get(self, req, resp):
        doc = {"status": "ok", "host": socket.getfqdn(), "worker.pid": os.getpid()}
        resp.body = json.dumps(doc, ensure_ascii=False)


class ImageStorage:

    def __init__(self, base_path: str = "./data/images"):
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def clean(self):
        shutil.rmtree(self._base_path, ignore_errors=True)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _image_path(self, gid: str, is_mask: bool) -> Path:
        image_type = "mask" if is_mask else "image"
        return self._base_path / f"{gid}-{image_type}.png"

    def save(self, gid: str, image: Image, is_mask: bool):
        image.save(self._image_path(gid, is_mask), format="PNG")

    def load(self, gid: str, is_mask: bool) -> Image:
        image_path = self._image_path(gid, is_mask)
        if image_path.exists():
            return PIL.Image.open(image_path)


class SegmentationJobs:

    def __init__(self, storage: ImageStorage, job_queue: Deque):
        self._storage = storage
        self._job_queue = job_queue

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        try:
            req_doc = json.load(req.bounded_stream)
            assert "image" in req_doc and "content" in req_doc["image"]

            image = preprocess(req_doc)
            job_id = uuid4().hex
            self._storage.save(job_id, image, is_mask=False)
            self._job_queue.appendleft(job_id)

            resp.status = falcon.HTTP_CREATED
            resp.body = json.dumps({"job_id": job_id})
        except (ValueError, AssertionError) as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


class SegmentationJob:

    def __init__(self, storage: ImageStorage):
        self._storage = storage

    def on_get(self, req: falcon.Request, resp: falcon.Response, job_id: str):
        try:
            image = self._storage.load(job_id, is_mask=True)
            assert image is not None

            resp_doc = postprocess(image)
            resp.body = json.dumps(resp_doc)
        except AssertionError as e:
            raise falcon.HTTPError(falcon.HTTP_404, description="No image found")
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


def load_model(storage: ImageStorage) -> SegmentationModelBase:
    try:
        module = import_module("model")
        model_class = getattr(module, "SegmentationModel")
        return model_class(storage)
    except ModuleNotFoundError as e:
        if int(os.environ.get("USE_FALLBACK_MODEL", 0)):
            logger.warning(
                "No 'model' module found. Using fallback model 'DummySegmentationModel'"
            )
            return DummySegmentationModel(storage)
        else:
            raise e


def start_worker(storage: ImageStorage, job_queue: Deque):

    def thread_function():
        logger.info("Starting worker thread")
        model_wrapper = load_model(storage)
        logger.info("Worker thread started")

        while True:
            if job_queue:
                job_id = job_queue.pop()
                logger.info(f"Processing task {job_id}")
                model_wrapper(job_id)
                logger.info(f"Finished task {job_id}")
            else:
                time.sleep(1)

    thread = threading.Thread(target=thread_function, args=(), daemon=True)
    thread.start()


def add_routes(api: falcon.API):
    storage = ImageStorage()
    job_queue = deque()
    start_worker(storage, job_queue)

    api.add_route("/", SegmentationBase())
    # Note: falcon internally strips trailing slashes from requests
    api.add_route("/jobs", SegmentationJobs(storage, job_queue))
    api.add_route("/jobs/{job_id}", SegmentationJob(storage))
