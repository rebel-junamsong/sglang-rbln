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
SERVER_READY_TIMEOUT_SEC = 900
HEALTH_PROBE_TIMEOUT_SEC = 30
REQUEST_TIMEOUT_SEC = 900
SHUTDOWN_TIMEOUT_SEC = 60
# These tests carry no npu marker, and the host's NPUs belong to other sessions:
# the server runs in host mode and sees no NPU (there is no NPU 99), with or
# without torch-rbln in the environment.
HIDDEN_NPUS = "99"


def _free_port() -> int:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


def _descendants(pid: int) -> list[int]:
    """Every live process under ``pid``, wherever its process group.

    The scheduler starts its own session, so killing the server's group misses it.
    """
    found, pending = [], [pid]
    while pending:
        parent = pending.pop()
        for task in Path(f"/proc/{parent}/task").glob("*"):
            try:
                children = (task / "children").read_text().split()
            except OSError:  # the task exited while we looked
                continue
            found += map(int, children)
            pending += map(int, children)
    return found


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _stop(server: subprocess.Popen) -> None:
    """Stop the server and everything it started, then check nothing is left."""
    children = _descendants(server.pid)
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=SHUTDOWN_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    for child in children:
        if _alive(child):
            os.kill(child, signal.SIGKILL)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SEC
    while (left := [c for c in children if _alive(c)]) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not left, f"server children outlived the server: {left}"


@pytest.fixture(scope="module")
def serve_url(npu_name, tmp_path_factory):
    port = _free_port()
    serve_env = {
        **os.environ,
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
        "RBLN_VISIBLE_DEVICES": HIDDEN_NPUS,
        "SGLANG_RBLN_EXECUTION_MODE": "host",
    }
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
    # A file, not a pipe: nobody reads a pipe while the tests run, and a server
    # whose pipe fills blocks on its next log line.
    log_path = tmp_path_factory.mktemp("serve") / "server.log"
    with log_path.open("w") as server_log:
        server = subprocess.Popen(
            serve_cmd,
            env=serve_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_SEC
    try:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                tail = log_path.read_text(errors="replace")[-4000:]
                raise RuntimeError(f"server exited early:\n{tail}")
            try:
                health = requests.get(
                    f"{base_url}/health", timeout=HEALTH_PROBE_TIMEOUT_SEC
                )
                if health.status_code == 200:
                    break
            except requests.RequestException:  # not listening yet, or busy loading
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server did not become healthy in time")
        yield base_url
    finally:
        _stop(server)


def test_server_reports_the_smoke_model(serve_url):
    served_models = requests.get(
        f"{serve_url}/v1/models", timeout=HEALTH_PROBE_TIMEOUT_SEC
    ).json()
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
