import base64
import io
import os
import socket
import threading
import time
from dataclasses import asdict
from importlib import import_module
from typing import Deque, List
from uuid import uuid4
import json
from collections import deque

import PIL
from PIL.Image import Image
import falcon

from telesto.logger import logger
from telesto.config import config
from telesto.instance_segmentation import SegmentationObject, DataStorage
from telesto.instance_segmentation.model import DummySegmentationModel, SegmentationModelBase


INPUT_IMAGE_FORMAT = {
    "type": "png",
    "palette": "RGB24",
    "encoding": "base64",
    "max_size": "5120",
}

OUTPUT_OBJECT_MASK_FORMAT = {
    "type": "json",
    "palette": "GREY8",
    "encoding": "plain",
}

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
            "description": "Returns status of the API",
        },
        {
            "path": "/docs",
            "method": "GET",
            "name": "Documentation endpoint",
            "description": "Returns this information",
        },
        {
            "path": "/jobs/",
            "method": "POST",
            "request_body": {
                "image": "<str>",
            },
            "image_format": {
                **INPUT_IMAGE_FORMAT,
            },
            "response_body": {
                "job_id": "<UUID>"
            }
        },
        {
            "path": "/jobs/<job_id>",
            "method": "GET",
            "response_body": {
                "objects": [
                    {
                        "class_i": "<int>",
                        "x": "<int>",
                        "y": "<int>",
                        "w": "<int>",
                        "h": "<int>",
                        "mask": "<str>"
                    }
                ],
            },
            "object_mask_format": {
                **OUTPUT_OBJECT_MASK_FORMAT,
            },
            "classes": config.get("common", "classes").split(","),
        }
    ]
}


def preprocess(doc: dict) -> Image:
    try:
        image_bytes = base64.b64decode(doc["image"])
        image = PIL.Image.open(io.BytesIO(image_bytes))
        assert image.mode == "RGB", f"Wrong image mode: {image.mode} != 'RGB'"
    except Exception as e:
        raise ValueError(e)
    return image


def postprocess(objects: List[SegmentationObject]) -> dict:
    return {"objects": [asdict(obj) for obj in objects]}


class SegmentationBase:
    def on_get(self, req, resp):
        doc = {"status": "ok", "host": socket.getfqdn(), "worker.pid": os.getpid()}
        resp.body = json.dumps(doc, ensure_ascii=False)


class SegmentationDocs:
    def on_get(self, req, resp):
        resp.body = json.dumps(API_DOCS, ensure_ascii=False)


class SegmentationJobs:

    def __init__(self, storage: DataStorage, job_queue: Deque):
        self._storage = storage
        self._job_queue = job_queue

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        try:
            req_doc = json.load(req.bounded_stream)
            assert "image" in req_doc, f"'image' not found in {req_doc}"

            job_id = uuid4().hex
            image = preprocess(req_doc)
            self._storage.save(job_id, image, output=False)
            self._job_queue.appendleft(job_id)

            resp.status = falcon.HTTP_CREATED
            resp.body = json.dumps({"job_id": job_id})
        except (ValueError, AssertionError) as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


class SegmentationJob:

    def __init__(self, storage: DataStorage):
        self._storage = storage

    def on_get(self, req: falcon.Request, resp: falcon.Response, job_id: str):
        try:
            objects = self._storage.load(job_id, output=True)
            assert objects is not None, "No data found"

            resp_doc = postprocess(objects)
            resp.body = json.dumps(resp_doc)
        except AssertionError as e:
            raise falcon.HTTPError(falcon.HTTP_404, description=str(e))
        except ValueError as e:
            raise falcon.HTTPError(falcon.HTTP_400, description=str(e))
        except Exception as e:
            logger.error(e, exc_info=True)
            raise falcon.HTTPError(falcon.HTTP_500)


def load_model(storage: DataStorage) -> SegmentationModelBase:
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


def start_worker(storage: DataStorage, job_queue: Deque):

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
    storage = DataStorage()
    job_queue = deque()
    start_worker(storage, job_queue)

    api.add_route("/", SegmentationBase())
    api.add_route("/docs", SegmentationDocs())
    # Note: falcon internally strips trailing slashes from requests
    api.add_route("/jobs", SegmentationJobs(storage, job_queue))
    api.add_route("/jobs/{job_id}", SegmentationJob(storage))
