"""
Backend: numpy with optional CuPy on GPU. Use xp() for array ops when you want GPU.
"""

import os

_xp = None
_device = "cpu"


def get_backend():
    """Return (xp, device). xp is numpy or cupy; device is 'cuda' or 'cpu'."""
    global _xp, _device
    if _xp is not None:
        return _xp, _device
    use_gpu = os.environ.get("ASKORACT_USE_GPU", "0").lower() in ("1", "true", "yes")
    if use_gpu:
        try:
            import cupy as cp
            _xp = cp
            _device = "cuda"
            return _xp, _device
        except ImportError:
            pass
    import numpy as np
    _xp = np
    _device = "cpu"
    return _xp, _device


def numpy():
    """Return numpy for RNG and scalar-heavy code (always CPU)."""
    import numpy as np
    return np


# Lazy init on first use
def xp():
    return get_backend()[0]


def device():
    return get_backend()[1]
