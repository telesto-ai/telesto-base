import os

import pytest
from falcon import testing

from telesto.app import get_app


@pytest.fixture
def client():
    return testing.TestClient(get_app())


@pytest.fixture(scope="session", autouse=True)
def use_fallback_model():
    os.environ["USE_FALLBACK_MODEL"] = "1"
