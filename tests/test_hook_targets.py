# SPDX-License-Identifier: Apache-2.0

import importlib
from collections import Counter

import pytest
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

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
def registered_hooks(
    monkeypatch: pytest.MonkeyPatch, rbln_platform
) -> list[tuple[str, HookType]]:
    hooks: list[tuple[str, HookType]] = []
    monkeypatch.setattr(
        HookRegistry,
        "register",
        classmethod(
            lambda cls, target, hook, hook_type=HookType.AFTER: hooks.append(
                (target, hook_type)
            )
        ),
    )
    patches.register()
    assert hooks
    return hooks


def test_every_hook_target_exists_in_the_installed_sglang(registered_hooks):
    missing = []
    for target, _ in registered_hooks:
        try:
            resolve(target)
        except (ImportError, AttributeError):
            missing.append(target)
    assert not missing


def test_each_target_gets_at_most_one_around_hook(registered_hooks):
    arounds = Counter(
        target for target, kind in registered_hooks if kind is HookType.AROUND
    )
    assert {target: count for target, count in arounds.items() if count > 1} == {}
