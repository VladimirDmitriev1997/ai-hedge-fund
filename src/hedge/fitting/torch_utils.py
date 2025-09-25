from dataclasses import dataclass  # если не нужно — можно удалить
from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    from torch import Tensor as TorchTensor

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    TorchTensor = Any  # type: ignore[misc,assignment]
    _HAS_TORCH = False

Array = Union[np.ndarray, TorchTensor]


# -------- Core backend detection / coercion --------


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


# Aliases expected by losses.py
_is_torch = is_torch
_as_array = as_array


# -------- Elementwise helpers (NumPy/Torch parity) --------


def _zeros_like(x: Array) -> Array:
    return torch.zeros_like(x) if is_torch(x) else np.zeros_like(x)


def _ones_like(x: Array) -> Array:
    return torch.ones_like(x) if is_torch(x) else np.ones_like(x)


def _abs(x: Array) -> Array:
    return torch.abs(x) if is_torch(x) else np.abs(x)


def _sum(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if is_torch(x):
        # torch.sum не принимает dim=None — предполагаем, что axis задан, как и в остальном коде
        return torch.sum(x, dim=axis, keepdim=keepdims)  # type: ignore[arg-type]
    return np.sum(x, axis=axis, keepdims=keepdims)


def _mean(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    if is_torch(x):
        return torch.mean(x, dim=axis, keepdim=keepdims)  # type: ignore[arg-type]
    return np.mean(x, axis=axis, keepdims=keepdims)


def _std(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Standard deviation with ddof=0 (NumPy default).
    """
    if is_torch(x):
        # unbiased=False соответствует ddof=0
        return torch.std(x, dim=axis, unbiased=False, keepdim=keepdims)  # type: ignore[arg-type]
    return np.std(x, axis=axis, ddof=0, keepdims=keepdims)


def _log1p(x: Array) -> Array:
    if is_torch(x):
        return torch.log1p(x)
    return np.log1p(x)


def _clip(
    x: Array,
    min_: Optional[Union[float, Array]] = None,
    max_: Optional[Union[float, Array]] = None,
) -> Array:
    if is_torch(x):
        t = x
        # Приводим границы к тензорам на том же dtype/device (если они заданы)
        min_t = None
        max_t = None
        if min_ is not None:
            min_t = min_ if is_torch(min_) else torch.as_tensor(min_, dtype=t.dtype, device=t.device)  # type: ignore[attr-defined]
        if max_ is not None:
            max_t = max_ if is_torch(max_) else torch.as_tensor(max_, dtype=t.dtype, device=t.device)  # type: ignore[attr-defined]
        # torch.clamp принимает и скаляры, и тензоры
        return torch.clamp(t, min=min_t, max=max_t)  # type: ignore[arg-type]
    return np.clip(x, a_min=min_, a_max=max_)


def _cumsum(x: Array, axis: Optional[int] = None) -> Array:
    if is_torch(x):
        if axis is None:
            # Поведение как у NumPy: cumsum по вектору после выпрямления
            return torch.cumsum(x.reshape(-1), dim=0)
        return torch.cumsum(x, dim=axis)  # type: ignore[arg-type]
    return np.cumsum(x, axis=axis)


def _cumprod(x: Array, axis: Optional[int] = None) -> Array:
    if is_torch(x):
        if axis is None:
            return torch.cumprod(x.reshape(-1), dim=0)
        return torch.cumprod(x, dim=axis)  # type: ignore[arg-type]
    return np.cumprod(x, axis=axis)


def _maximum(a: Array, b: Array) -> Array:
    """
    Elementwise maximum with backend coercion.
    - If either input is torch, upcast both to torch (dtype/device of the tensor).
    - Else, use numpy.
    """
    if is_torch(a) or is_torch(b):
        if not is_torch(a):
            a = torch.as_tensor(  # type: ignore[attr-defined]
                a,
                dtype=b.dtype if is_torch(b) else None,
                device=b.device if is_torch(b) else None,
            )
        if not is_torch(b):
            b = torch.as_tensor(  # type: ignore[attr-defined]
                b,
                dtype=a.dtype if is_torch(a) else None,
                device=a.device if is_torch(a) else None,
            )
        return torch.maximum(a, b)  # type: ignore[return-value]
    return np.maximum(a, b)


def _where(cond: Union[bool, Array], x: Array, y: Array) -> Array:
    """
    Backend-agnostic where:
    - Если любой из (cond, x, y) — torch.Tensor, приводим остальные к torch на тот же device/dtype.
    - Иначе используем NumPy.
    """
    if is_torch(cond) or is_torch(x) or is_torch(y):
        # Опорный тензор для dtype/device
        ref = None
        for z in (x, y, cond):
            if is_torch(z):
                ref = z
                break
        assert ref is not None  # для type checker
        ct = cond if is_torch(cond) else torch.as_tensor(cond, dtype=torch.bool, device=ref.device)  # type: ignore[attr-defined]
        xt = x if is_torch(x) else torch.as_tensor(x, dtype=getattr(ref, "dtype", None), device=getattr(ref, "device", None))  # type: ignore[attr-defined]
        yt = y if is_torch(y) else torch.as_tensor(y, dtype=getattr(ref, "dtype", None), device=getattr(ref, "device", None))  # type: ignore[attr-defined]
        return torch.where(ct, xt, yt)  # type: ignore[return-value]
    return np.where(cond, x, y)
