# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec


class _BlockedLoader(Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(
            f"{spec.name} needs a CUDA device; blocked on rbln", name=spec.name
        )

    def exec_module(self, module):
        pass


class _BlockedImportFinder(MetaPathFinder):
    def __init__(self) -> None:
        self.modules: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.modules:
            return ModuleSpec(fullname, _BlockedLoader())
        return None


def _imports_cleanly(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def block_without_cuda(module: str) -> bool:
    import torch

    if torch.cuda.is_available() or module in sys.modules:
        return False
    if _imports_cleanly(module):
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
