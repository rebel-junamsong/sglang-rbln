# SPDX-License-Identifier: Apache-2.0
import base64
import io
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from PIL import Image

pytestmark = pytest.mark.e2e

SMOKE_MODEL_ID = "hf-internal-testing/tiny-sdxl-pipe"
SERVER_READY_TIMEOUT_SEC = 600
REQUEST_TIMEOUT_SEC = 600


def _free_port() -> int:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


@pytest.fixture(scope="module")
def serve_url(npu_name):
    port = _free_port()
    serve_env = {**os.environ, "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln"}
    serve_env.setdefault("RBLN_FORCE_NPU_NAME", npu_name)
    serve_cmd = [
        str(Path(sys.executable).with_name("sglang")),
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        SMOKE_MODEL_ID,
        "--backend",
        "diffusers",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    server = subprocess.Popen(
        serve_cmd,
        env=serve_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_SEC
    try:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                server_log = server.stdout.read()[-4000:]
                raise RuntimeError(f"server exited early:\n{server_log}")
            try:
                if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server did not become healthy in time")
        yield base_url
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)


def test_server_reports_the_smoke_model(serve_url):
    served_models = requests.get(f"{serve_url}/v1/models", timeout=30).json()
    assert [entry["id"] for entry in served_models["data"]] == [SMOKE_MODEL_ID]


def test_image_generation_round_trip(serve_url):
    response = requests.post(
        f"{serve_url}/v1/images/generations",
        json={
            "prompt": "a red square",
            "n": 1,
            "size": "64x64",
            "num_inference_steps": 2,
            "response_format": "b64_json",
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    assert response.status_code == 200, response.text[-2000:]
    (generated,) = response.json()["data"]
    image = Image.open(io.BytesIO(base64.b64decode(generated["b64_json"])))
    assert image.size == (64, 64)
