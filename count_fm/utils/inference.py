"""Inference context for a model or separate birth/death networks."""
from contextlib import contextmanager
import torch


@contextmanager
def inference_mode(nets):
    roots = list(nets) if isinstance(nets, (list, tuple)) else [nets]
    modules = {module: module.training for root in roots for module in root.modules()}
    try:
        for root in roots:
            root.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modules.items():
            module.training = training
