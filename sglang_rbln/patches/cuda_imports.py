# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder


class _BlockedImportFinder(MetaPathFinder):
    def __init__(self) -> None:
        self.modules: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.modules:
            raise ImportError(f"{fullname} needs a CUDA device; blocked on rbln")
        return None


def block_without_cuda(module: str) -> bool:
    import torch

    if torch.cuda.is_available() or module in sys.modules:
        return False
    finder = next(
        (item for item in sys.meta_path if isinstance(item, _BlockedImportFinder)),
        None,
    )
    if finder is None:
        finder = _BlockedImportFinder()
        sys.meta_path.insert(0, finder)
    finder.modules.add(module)
    return True
