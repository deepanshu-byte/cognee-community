"""Keep unit tests independent of remote model pricing and dlt telemetry."""

import os

import httpx
import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
os.environ["RUNTIME__DLTHUB_TELEMETRY"] = "false"


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests must use mocked HTTP transports.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
