# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from sglang_rbln.patches import ffmpeg_spawn

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="process_group needs Python 3.11"
)


@pytest.fixture
def patched(monkeypatch):
    from imageio_ffmpeg import _io, _utils

    for module in (_utils, _io):
        monkeypatch.setattr(module, "_popen_kwargs", _utils._popen_kwargs)
    assert ffmpeg_spawn.register() == ffmpeg_spawn.PATCHED_MODULES
    return _utils, _io


def test_writer_spawns_ffmpeg_in_its_own_group_without_preexec_fn(patched):
    for module in patched:
        popen = module._popen_kwargs(prevent_sigint=True)
        assert popen["preexec_fn"] is None
        assert popen["process_group"] == 0


def test_other_spawns_are_unchanged(patched):
    utils, _ = patched
    assert "process_group" not in utils._popen_kwargs()


def test_register_twice_wraps_once(patched):
    utils, _ = patched
    wrapped = utils._popen_kwargs
    ffmpeg_spawn.register()
    assert utils._popen_kwargs is wrapped


def test_child_gets_its_own_process_group(patched):
    utils, _ = patched
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpgrp())"],
        capture_output=True,
        text=True,
        check=True,
        **{
            name: value
            for name, value in utils._popen_kwargs(prevent_sigint=True).items()
            if name in ("preexec_fn", "process_group")
        },
    )
    assert int(child.stdout) != os.getpgrp()
