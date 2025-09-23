from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:
    torch = None
    TorchTensor = Any
    _HAS_TORCH = False


Array = Union[np.ndarray, TorchTensor]


def is_torch(x: Any) -> bool:
    """True iff x is a torch.Tensor (and torch is available)."""
    return _HAS_TORCH and isinstance(x, TorchTensor)


def as_array(x: Any) -> Array:
    """Return torch.Tensor as-is; numpy arrays as-is; otherwise np.asarray(x)."""
    if is_torch(x):
        return x
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _zeros_like(x: Array) -> Array:
    return torch.zeros_like(x) if is_torch(x) else np.zeros_like(x)


def _ones_like(x: Array) -> Array:
    return torch.ones_like(x) if is_torch(x) else np.ones_like(x)


def _abs(x: Array) -> Array:
    return torch.abs(x) if is_torch(x) else np.abs(x)


def _sum(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if is_torch(x):
        return torch.sum(x, dim=axis, keepdim=keepdims)  # type: ignore[arg-type]
    return np.sum(x, axis=axis, keepdims=keepdims)


def _mean(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if is_torch(x):
        return torch.mean(x, dim=axis, keepdim=keepdims)  # type: ignore[arg-type]
    return np.mean(x, axis=axis, keepdims=keepdims)


def _maximum(a: Array, b: Array) -> Array:
    """
    Elementwise maximum with backend coercion.
    - If either input is torch, upcast both to torch (dtype/device of the tensor).
    - Else, use numpy.
    """
    if is_torch(a) or is_torch(b):
        if not is_torch(a):
            # type: ignore[attr-defined]
            a = torch.as_tensor(
                a,
                dtype=b.dtype if is_torch(b) else None,
                device=b.device if is_torch(b) else None,
            )
        if not is_torch(b):
            # type: ignore[attr-defined]
            b = torch.as_tensor(
                b,
                dtype=a.dtype if is_torch(a) else None,
                device=a.device if is_torch(a) else None,
            )
        return torch.maximum(a, b)  # type: ignore[return-value]
    return np.maximum(a, b)
