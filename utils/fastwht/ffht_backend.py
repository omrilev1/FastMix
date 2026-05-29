from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


_DLL_PATH = Path(__file__).with_name("_ffht_axis0.dll")
_LIB = None

if _DLL_PATH.exists():
    _LIB = ctypes.CDLL(str(_DLL_PATH))
    _LIB.ffht_axis0_float.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _LIB.ffht_axis0_float.restype = ctypes.c_int
    _LIB.ffht_axis0_double.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _LIB.ffht_axis0_double.restype = ctypes.c_int


def fwht_axis0_inplace(arr: np.ndarray) -> np.ndarray:
    if _LIB is None:
        raise ImportError(f"FFHT backend DLL not found at {_DLL_PATH}")

    if arr.ndim not in (1, 2):
        raise TypeError("fwht_axis0_inplace expects a 1D vector or 2D matrix")

    n = arr.shape[0]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("axis-0 length must be a power of two")

    if arr.dtype not in (np.float32, np.float64):
        raise TypeError("fwht_axis0_inplace expects float32 or float64 input")

    if arr.ndim == 1:
        if not arr.flags.c_contiguous:
            raise ValueError("1D input must be contiguous")
        d = 1
    else:
        if not arr.flags.f_contiguous:
            raise ValueError("2D input must be Fortran contiguous")
        d = arr.shape[1]

    ptr = ctypes.c_void_p(arr.ctypes.data)
    if arr.dtype == np.float32:
        status = _LIB.ffht_axis0_float(ptr, int(n), int(d))
    else:
        status = _LIB.ffht_axis0_double(ptr, int(n), int(d))

    if status != 0:
        raise RuntimeError(f"FFHT backend returned status {status}")

    return arr
