# SPDX-License-Identifier: MPL-2.0
"""Load an independently downloaded upstream checkout without its dataset imports."""
import importlib
from pathlib import Path
import sys
import types
import torch

def load_model(checkpoint, upstream="reference/RMVPE"):
    source = Path(upstream).resolve() / "src"
    if not (source / "model.py").exists():
        raise FileNotFoundError("Clone https://github.com/yxlllc/RMVPE to reference/RMVPE first")
    package = types.ModuleType("rmvpe_upstream")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    module = importlib.import_module("rmvpe_upstream.model")
    model = module.E2E0(4, 1, (2, 2)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    return model
