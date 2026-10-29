# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

from sglang_rbln import patches


def resolve(target: str) -> object:
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for name in parts[split:]:
            owner = getattr(owner, name)
        return owner
    raise ImportError(target)


@pytest.fixture
def registered_targets(monkeypatch: pytest.MonkeyPatch, rbln_platform) -> list[str]:
    targets: list[str] = []
    monkeypatch.setattr(
        HookRegistry,
        "register",
        classmethod(lambda cls, target, *args, **kwargs: targets.append(target)),
    )
    patches.register()
    assert targets
    return targets


def test_every_hook_target_exists_in_the_installed_sglang(registered_targets):
    missing = []
    for target in registered_targets:
        try:
            resolve(target)
        except (ImportError, AttributeError):
            missing.append(target)
    assert not missing
