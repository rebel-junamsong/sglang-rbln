# SPDX-License-Identifier: Apache-2.0
import pytest
from sglang.multimodal_gen.runtime import platforms


@pytest.fixture
def fresh_platform(monkeypatch):
    for attr in ("_current_platform", "_current_platform_selection"):
        monkeypatch.setattr(platforms, attr, None)
    monkeypatch.setattr(platforms, "_backend_init_done", False)
    yield
