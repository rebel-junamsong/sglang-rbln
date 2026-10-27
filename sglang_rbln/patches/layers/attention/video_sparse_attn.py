# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

BLOCKED_MODULES = ("vsa",)


class _BlockedImportFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top_level = fullname.partition(".")[0]
        if top_level in BLOCKED_MODULES:
            raise ImportError(f"{fullname} needs a CUDA device; blocked on rbln")
        return None


def block_without_cuda() -> bool:
    import torch

    if torch.cuda.is_available():
        return False
    if any(isinstance(finder, _BlockedImportFinder) for finder in sys.meta_path):
        return True
    if any(name in sys.modules for name in BLOCKED_MODULES):
        return False
    sys.meta_path.insert(0, _BlockedImportFinder())
    return True
