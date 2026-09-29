"""State-dictionary checkpoint helpers."""

import torch

def save_chkpts(nets, path):
    unwrap = (lambda m: m.module if hasattr(m, "module") else m)
    if isinstance(nets, (list, tuple)):
        blob = {f"net_{i}": unwrap(n).state_dict() for i, n in enumerate(nets)}
    else:
        blob = {"net": unwrap(nets).state_dict()}
    torch.save(blob, path)


def load_chkpts(nets, path, map_location=None, strict=True):
    blob = torch.load(path, map_location=map_location)
    unwrap = (lambda m: m.module if hasattr(m, "module") else m)
    if isinstance(nets, (list, tuple)):
        for i, n in enumerate(nets): unwrap(n).load_state_dict(blob[f"net_{i}"], strict=strict)
    else:
        unwrap(nets).load_state_dict(blob["net"], strict=strict)
