# SPDX-License-Identifier: Apache-2.0
"""Make ``import vsa`` fail cleanly where there is no CUDA device.

Why: ``vsa/__init__.py`` calls ``torch.cuda.get_device_capability(0)`` at
import time, and the ``sglang serve`` CLI imports it eagerly through
``entrypoints/cli/serve.py -> apps.webui -> minimax_h3 -> video_sparse_attn``,
so the CLI dies before the server starts on a host without an NVIDIA driver.
How: a ``sys.meta_path`` finder raises ``ImportError`` for ``vsa``;
``video_sparse_attn.py`` already handles that by binding
``video_sparse_attn = None`` and refusing the backend at use time.
Upstream: guard the probe in ``vsa`` or catch ``RuntimeError`` at the import.
Remove when: importing ``vsa`` no longer touches the CUDA runtime.
"""

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
