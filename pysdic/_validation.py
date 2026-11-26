"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# =============================================================================
# Shared input validation helpers
# =============================================================================
# Internal use only: these helpers are shared by all the sub-packages of pysdic
# (core, fem, geometry, ...) so that the same input is accepted (or rejected) with
# the same error everywhere.
#
#     # In a module of a pysdic sub-package (e.g. pysdic/geometry/point_cloud.py)
#     from .._validation import _as_array, _validate_real
#
# Conventions:
#     - A wrong type raises a ``TypeError``, a wrong value or shape a ``ValueError``,
#       an out-of-range index an ``IndexError`` and a missing file a
#       ``FileNotFoundError``.
#     - ``numpy.bool_`` is accepted wherever ``bool`` is accepted.
#     - Booleans are rejected wherever a number is expected (``True`` is not ``1``).
#     - Validated values are returned as python scalars, numpy arrays or
#       ``pathlib.Path``.
#     - ``_as_array`` and ``_as_batch_array`` have no default for ``finite``: every call writes
#       ``finite=True`` or ``finite=False`` so that the reader directly knows if NaN
#       and inf are allowed.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping
from numbers import Integral, Real
from numpy.typing import ArrayLike

import math
import os
import pathlib

import numpy


def _is_bool(value) -> bool:
    r"""
    Return True if ``value`` is a python or numpy boolean.

    .. important::

        Internal use only.

    """
    return isinstance(value, (bool, numpy.bool_))


def _validate_bool(value: bool, name: str) -> bool:
    r"""
    Check that ``value`` is a boolean (``numpy.bool_`` accepted) and return it as a
    python ``bool``.

    .. important::

        Internal use only.

    """
    if not _is_bool(value):
        raise TypeError(f"Input '{name}' must be a boolean value. Got type {type(value)}.")
    return bool(value)


def _validate_real(
    value: Optional[Real],
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
    finite: bool = False,
    allow_none: bool = False,
) -> Optional[float]:
    r"""
    Check that ``value`` is a real number (booleans are rejected) and return it as a
    python ``float``.

    NaN is accepted only if ``finite=False``, ``non_negative=False`` and
    ``positive=False``. Infinity is accepted only if ``finite=False`` (``+inf`` is
    positive).

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Real
        Value to check.

    name: str
        Name of the input used in the error messages.

    non_negative: bool, optional (default: ``False``)
        If True, the value must be greater than or equal to zero (NaN is rejected).

    positive: bool, optional (default: ``False``)
        If True, the value must be strictly greater than zero (NaN is rejected).

    finite: bool, optional (default: ``False``)
        If True, the value must be finite (NaN, ``+inf`` and ``-inf`` are rejected).

    allow_none: bool, optional (default: ``False``)
        If True, ``None`` is accepted and returned as is.

    """
    if value is None and allow_none:
        return None
    if _is_bool(value) or not isinstance(value, Real):
        raise TypeError(
            f"Input '{name}' must be a real number{' or None' if allow_none else ''}. "
            f"Got type {type(value)}."
        )
    value = float(value)
    if finite and not math.isfinite(value):
        raise ValueError(f"Input '{name}' must be finite. Got {value}.")
    if non_negative and not value >= 0.0:  # also rejects NaN
        raise ValueError(f"Input '{name}' must be non-negative. Got {value}.")
    if positive and not value > 0.0:  # also rejects NaN
        raise ValueError(f"Input '{name}' must be strictly positive. Got {value}.")
    return value


def _validate_integer(
    value: Optional[Integral],
    name: str,
    *,
    min_value: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
    r"""
    Check that ``value`` is an integer (booleans are rejected) and return it as a
    python ``int``.

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Optional[Integral]
        Value to check.

    name: str
        Name of the input used in the error messages.

    min_value: Optional[int], optional (default: ``None``)
        If given, the value must be greater than or equal to ``min_value``.

    allow_none: bool, optional (default: ``False``)
        If True, ``None`` is accepted and returned as is.

    """
    if value is None and allow_none:
        return None
    if _is_bool(value) or not isinstance(value, Integral):
        raise TypeError(
            f"Input '{name}' must be an integer{' or None' if allow_none else ''}. "
            f"Got type {type(value)}."
        )
    value = int(value)
    if min_value is not None and value < min_value:
        raise ValueError(
            f"Input '{name}' must be greater than or equal to {min_value}. Got {value}."
        )
    return value


def _validate_tolerances(rtol: Real, atol: Real) -> Tuple[float, float]:
    r"""
    Check the tolerances ``rtol`` and ``atol`` of a numerical comparison (see
    ``numpy.allclose``): both must be finite and non-negative.

    .. important::

        Internal use only.

    """
    rtol = _validate_real(rtol, "rtol", non_negative=True, finite=True)
    atol = _validate_real(atol, "atol", non_negative=True, finite=True)
    return rtol, atol


def _validate_kwargs(kwargs: Optional[Mapping], name: str) -> Dict[str, Any]:
    r"""
    Check that ``kwargs`` is a dictionary of keyword arguments or None, and return a
    new ``dict`` (an empty one for None) to protect the user's dictionary.

    .. important::

        Internal use only.

    """
    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(
            f"Input '{name}' must be a dictionary or None. Got type {type(kwargs)}."
        )
    return dict(kwargs)


def _validate_kwargs_sequence(
    seq_kwargs: Optional[Sequence[Optional[Mapping]]], size: int, name: str
) -> List[Dict[str, Any]]:
    r"""
    Check that ``seq_kwargs`` is a list (or tuple) of ``size`` dictionaries of keyword
    arguments or None, and return a new list of new ``dict`` (an empty one for each None).
    If ``seq_kwargs`` is None, a list of ``size`` empty dictionaries is returned.

    .. important::

        Internal use only.

    """
    if seq_kwargs is None:
        return [{} for _ in range(size)]
    if not isinstance(seq_kwargs, (list, tuple)):
        raise TypeError(
            f"Input '{name}' must be a list or tuple of dictionaries or None. "
            f"Got type {type(seq_kwargs)}."
        )
    if len(seq_kwargs) != size:
        raise ValueError(f"Input '{name}' must contain {size} elements. Got {len(seq_kwargs)}.")
    return [_validate_kwargs(kwargs, f"{name}[{k}]") for k, kwargs in enumerate(seq_kwargs)]


def _validate_str(
    value: Optional[str],
    name: str,
    *,
    allow_empty: bool = False,
    allow_none: bool = False,
) -> Optional[str]:
    r"""
    Check that ``value`` is a string (``numpy.str_`` accepted) and return it as a
    python ``str``.

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Optional[str]
        Value to check.

    name: str
        Name of the input used in the error messages.

    allow_empty: bool, optional (default: ``False``)
        If True, the empty string ``""`` is accepted.

    allow_none: bool, optional (default: ``False``)
        If True, ``None`` is accepted and returned as is.

    """
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Input '{name}' must be a string{' or None' if allow_none else ''}. "
            f"Got type {type(value)}."
        )
    value = str(value)
    if not allow_empty and value == "":
        raise ValueError(f"Input '{name}' must be a non-empty string.")
    return value


def _validate_choice(
    value: Optional[str],
    name: str,
    choices: Sequence[str],
    *,
    allow_none: bool = False,
) -> Optional[str]:
    r"""
    Check that ``value`` is one of the strings in ``choices`` (case-sensitive) and
    return it as a python ``str``.

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Optional[str]
        Value to check.

    name: str
        Name of the input used in the error messages.

    choices: Sequence[str]
        Accepted values, e.g. ``("meshio", "pyvista", "vtk")``.

    allow_none: bool, optional (default: ``False``)
        If True, ``None`` is accepted and returned as is.

    """
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Input '{name}' must be a string{' or None' if allow_none else ''}. "
            f"Got type {type(value)}."
        )
    value = str(value)
    if value not in choices:
        raise ValueError(
            f"Input '{name}' must be one of {', '.join(repr(c) for c in choices)}. "
            f"Got {value!r}."
        )
    return value


def _validate_path(
    value: Union[str, os.PathLike],
    name: str,
    *,
    must_exist: bool = False,
    must_be_dir: bool = False,
    extensions: Optional[Sequence[str]] = None,
) -> pathlib.Path:
    r"""
    Check that ``value`` is a path (``str`` or ``os.PathLike``) and return it as a
    ``pathlib.Path``.

    The path is not resolved: ``~`` and relative paths are kept as given.

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Union[str, os.PathLike]
        Value to check. Bytes paths are rejected.

    name: str
        Name of the input used in the error messages.

    must_exist: bool, optional (default: ``False``)
        If True, the path must be an existing file (``FileNotFoundError`` otherwise,
        with a specific message if it is an existing folder).

    must_be_dir: bool, optional (default: ``False``)
        If True, the path must be an existing folder (``FileNotFoundError`` if it does not
        exist, ``NotADirectoryError`` if it is a file). Exclusive with ``must_exist``.

    extensions: Optional[Sequence[str]], optional (default: ``None``)
        If given, the extension of the path must be one of them (case-insensitive),
        e.g. ``(".vtk", ".vtu")``.

    """
    if must_exist and must_be_dir:
        raise ValueError("Options 'must_exist' (existing file) and 'must_be_dir' (existing folder) are exclusive.")
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(
            f"Input '{name}' must be a string or a path-like object. Got type {type(value)}."
        )
    path = os.fspath(value)
    if not isinstance(path, str):
        raise TypeError(f"Input '{name}' must be a text path. Got a bytes path.")
    if path == "":
        raise ValueError(f"Input '{name}' must be a non-empty path.")
    path = pathlib.Path(path)
    if extensions is not None:
        allowed = tuple(extension.lower() for extension in extensions)
        if path.suffix.lower() not in allowed:
            raise ValueError(
                f"Input '{name}' must have one of the extensions {', '.join(allowed)}. "
                f"Got '{path.suffix}' for path '{path}'."
            )
    if must_exist and not path.is_file():
        if path.is_dir():
            raise FileNotFoundError(
                f"Input '{name}' must be an existing file. Got '{path}', which is a folder."
            )
        raise FileNotFoundError(f"Input '{name}' must be an existing file. Got '{path}'.")
    if must_be_dir and not path.is_dir():
        if path.exists():
            raise NotADirectoryError(
                f"Input '{name}' must be an existing folder. Got '{path}', which is a file."
            )
        raise FileNotFoundError(f"Input '{name}' must be an existing folder. Got '{path}'.")
    return path


def _check_finite(array: numpy.ndarray, name: str) -> None:
    r"""
    Raise a ``ValueError`` if a float array contains NaN, ``+inf`` or ``-inf``
    (integer arrays are always finite).

    .. important::

        Internal use only.

    """
    if numpy.issubdtype(array.dtype, numpy.inexact):
        non_finite = numpy.count_nonzero(~numpy.isfinite(array))
        if non_finite > 0:
            raise ValueError(
                f"Input '{name}' must contain only finite values (no NaN or inf). "
                f"Got {non_finite} non-finite value(s)."
            )


def _check_shape(
    array: numpy.ndarray,
    name: str,
    shape: Tuple[Optional[int], ...],
    shape_hint: Optional[str] = None,
) -> None:
    r"""
    Raise a ``ValueError`` if the shape of ``array`` does not match ``shape``.

    ``None`` in ``shape`` means any size for this dimension, e.g. ``(None, 3)`` for an
    array of 3D points of shape :math:`(N_p, 3)`.

    .. important::

        Internal use only.


    Parameters
    ----------
    array: numpy.ndarray
        Array to check (already converted, e.g. by ``_as_array``).

    name: str
        Name of the input used in the error messages.

    shape: Tuple[Optional[int], ...]
        Expected shape, with ``None`` for the free dimensions.

    shape_hint: Optional[str], optional (default: ``None``)
        Expected shape written in the error messages, e.g. ``"(N_p, 3)"``. By
        default, it is built from ``shape`` with ``*`` for the free dimensions.

    """
    if shape_hint is None:
        shape_hint = "(" + ", ".join("*" if s is None else str(s) for s in shape)
        shape_hint += ",)" if len(shape) == 1 else ")"
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(
            f"Input '{name}' must be an array of shape {shape_hint}. "
            f"Got shape {array.shape}."
        )


def _check_matching_size(
    size_a: int, size_b: int, name_a: str, name_b: str, dimension: str
) -> None:
    r"""
    Raise a ``ValueError`` if two dimensions that must match are different.

    .. important::

        Internal use only.

    """
    if size_a != size_b:
        raise ValueError(
            f"The dimension {dimension} of '{name_a}' and '{name_b}' must match. "
            f"Got {size_a} and {size_b} respectively."
        )


def _as_array(
    array: ArrayLike,
    name: str,
    dtype: type,
    *,
    shape: Tuple[Optional[int], ...],
    finite: bool,
    shape_hint: Optional[str] = None,
    accept_1d_as_column: bool = False,
    copy: bool = False,
) -> numpy.ndarray:
    r"""
    Convert ``array`` to a numpy array with the given ``dtype`` and check its shape.

    Use ``_as_batch_array`` instead when the number of dimensions is not fixed.

    .. code-block:: python

        _as_array(value, "parameters", numpy.float64, shape=(n_p,), finite=True, copy=True)
        _as_array(hmat, "homography_matrix", numpy.float64, shape=(3, 3), finite=True)
        _as_array(modes, "modes", numpy.float64, shape=(None, None), shape_hint="(n_o, n_i)", finite=True)

    .. important::

        Internal use only.


    Parameters
    ----------
    array: ArrayLike
        Input array.

    name: str
        Name of the input used in the error messages.

    dtype: type
        Target dtype (``numpy.float64`` or ``numpy.int64``).

    shape: Tuple[Optional[int], ...]
        **Required.** Expected shape, with ``None`` for the dimensions of any size: the
        number of dimensions is ``len(shape)``. For example ``(None, 3)`` for 3D points
        of shape :math:`(N_p, 3)`.

    finite: bool
        **Required** (no default, so that every call says it explicitly). If True,
        the array must not contain NaN, ``+inf`` or ``-inf``. If False, these values
        are accepted. Integer arrays are always finite.

    shape_hint: Optional[str], optional (default: ``None``)
        Expected shape written in the error messages, e.g. ``"(N_p, N_vpe)"``. By
        default, it is built from ``shape`` with ``*`` for the dimensions of any size.

    accept_1d_as_column: bool, optional (default: ``False``)
        If True and ``len(shape) == 2``, a 1D-array of shape :math:`(N,)` is accepted and
        converted to shape :math:`(N, 1)`.

    copy: bool, optional (default: ``False``)
        If True, the returned array never shares memory with the input. Otherwise, a
        copy is only made when the dtype conversion requires it.


    Returns
    -------
    numpy.ndarray
        Converted array.

    """
    if not isinstance(shape, tuple):  # Error of the developer, not of the user
        raise TypeError(f"Argument 'shape' must be a tuple. Got type {type(shape)}.")
    if copy:
        array = numpy.array(array, dtype=dtype)  # always a new array (numpy 1.x and 2.x)
    else:
        array = numpy.asarray(array, dtype=dtype)
    if accept_1d_as_column and len(shape) == 2 and array.ndim == 1:
        array = array[:, numpy.newaxis]
    _check_shape(array, name, shape, shape_hint)
    if finite:
        _check_finite(array, name)
    return array


def _as_scalar_or_1d_array(
    value: Union[Real, ArrayLike],
    name: str,
    size: int,
    dtype: type,
    *,
    finite: bool,
) -> numpy.ndarray:
    r"""
    Convert a scalar or an array of :math:`N` = ``size`` values to a 1D numpy array of
    shape :math:`(N,)`: a scalar is repeated :math:`N` times.

    An array is flattened, so that an array with the batch shape of the points (e.g.
    :math:`(H, W)` for points of shape :math:`(H, W, 2)`) is accepted, as long as it
    contains exactly :math:`N` values. Booleans are rejected.

    .. important::

        Internal use only.


    Parameters
    ----------
    value: Union[Real, ArrayLike]
        A real number, or an array with :math:`N` real numbers.

    name: str
        Name of the input used in the error messages.

    size: int
        Number of values :math:`N` of the returned array.

    dtype: type
        Target dtype (``numpy.float64`` or ``numpy.int64``).

    finite: bool
        **Required** (no default, so that every call says it explicitly). If True,
        the values must not contain NaN, ``+inf`` or ``-inf``.


    Returns
    -------
    numpy.ndarray
        Array of shape :math:`(N,)`.

    """
    raw = numpy.asarray(value)
    if raw.dtype == numpy.bool_ or not (
        numpy.issubdtype(raw.dtype, numpy.integer) or numpy.issubdtype(raw.dtype, numpy.floating)
    ):
        raise TypeError(
            f"Input '{name}' must be a real number or an array of real numbers. "
            f"Got an array with dtype {raw.dtype}."
        )
    if raw.ndim == 0:
        array = numpy.full(size, raw, dtype=dtype)
    else:
        array = raw.astype(dtype, copy=False).ravel()
        if array.shape != (size,):
            raise ValueError(
                f"Input '{name}' must be a scalar or contain {size} values. Got {array.size} values "
                f"(shape {raw.shape})."
            )
    if finite:
        _check_finite(array, name)
    return array


def _as_batch_array(
    array: ArrayLike,
    name: str,
    last_dim: Union[int, Tuple[int, ...]],
    dtype: type,
    *,
    finite: bool,
    min_ndim: int = 1,
    copy: bool = False,
) -> numpy.ndarray:
    r"""
    Convert ``array`` to a numpy array of shape :math:`(\dots, n)` with :math:`n` =
    ``last_dim`` (or one of the values of ``last_dim``) and any batch shape :math:`(\dots)`.

    Use ``_as_array`` instead when the number of dimensions is fixed.

    .. important::

        Internal use only.


    Parameters
    ----------
    array: ArrayLike
        Input array.

    name: str
        Name of the input used in the error messages.

    last_dim: Union[int, Tuple[int, ...]]
        Expected size :math:`n` of the last dimension, or a tuple of the accepted sizes
        (e.g. ``(2, 3)`` for 2D or 3D points).

    dtype: type
        Target dtype (``numpy.float64`` or ``numpy.int64``).

    finite: bool
        **Required** (no default, so that every call says it explicitly). If True,
        the array must not contain NaN, ``+inf`` or ``-inf``.

    min_ndim: int, optional (default: ``1``)
        Minimum number of dimensions, e.g. ``2`` to require a batch dimension (a
        single point must then be given with shape :math:`(1, n)`).

    copy: bool, optional (default: ``False``)
        If True, the returned array never shares memory with the input. Otherwise, a
        copy is only made when the dtype conversion requires it.


    Returns
    -------
    numpy.ndarray
        Converted array of shape :math:`(\dots, n)`.

    """
    if copy:
        array = numpy.array(array, dtype=dtype)  # always a new array (numpy 1.x and 2.x)
    else:
        array = numpy.asarray(array, dtype=dtype)
    accepted = (last_dim,) if isinstance(last_dim, int) else tuple(last_dim)
    if array.ndim < max(min_ndim, 1) or array.shape[-1] not in accepted:
        shapes = " or ".join(f"(..., {n})" for n in accepted)
        raise ValueError(
            f"Input '{name}' must be an array of shape {shapes} with at least "
            f"{max(min_ndim, 1)} dimension(s). Got shape {array.shape}."
        )
    if finite:
        _check_finite(array, name)
    return array


def _as_bool_mask(
    mask: ArrayLike,
    size: Optional[int],
    name: str = "mask",
    *,
    copy: bool = False,
) -> numpy.ndarray:
    r"""
    Convert ``mask`` to a boolean array of shape :math:`(N,)` with :math:`N` = ``size``.

    The mask must already contain booleans: an integer array is rejected instead of
    being cast, since ``[0, 2]`` is much more likely to be a list of indices than a mask.

    .. important::

        Internal use only.


    Parameters
    ----------
    mask: ArrayLike
        Input mask.

    size: Optional[int]
        Expected size :math:`N` of the mask. If None, any 1D mask is accepted.

    name: str, optional (default: ``"mask"``)
        Name of the input used in the error messages.

    copy: bool, optional (default: ``False``)
        If True, the returned array never shares memory with the input.

    """
    mask = numpy.array(mask) if copy else numpy.asarray(mask)
    if mask.size == 0:
        mask = mask.astype(bool)  # numpy.asarray([]) is a float array
    if mask.dtype != numpy.bool_:
        raise TypeError(
            f"Input '{name}' must be a boolean array. Got an array with dtype {mask.dtype}."
        )
    if size is None:
        if mask.ndim != 1:
            raise ValueError(
                f"Input '{name}' must be a 1D array of shape (N,). Got shape {mask.shape}."
            )
    elif mask.shape != (size,):
        raise ValueError(
            f"Input '{name}' must be a 1D array of shape ({size},). Got shape {mask.shape}."
        )
    return mask


def _as_index_array(indices: ArrayLike, size: int, name: str = "indices") -> numpy.ndarray:
    r"""
    Convert ``indices`` to an integer array of shape :math:`(R,)` and check that all the
    indices are in :math:`[0, N)` with :math:`N` = ``size``.

    Float arrays are rejected instead of being truncated, and negative indices are
    rejected instead of counting from the end.

    .. important::

        Internal use only.

    """
    indices = numpy.asarray(indices)
    if indices.size == 0:
        indices = indices.astype(numpy.int64)  # numpy.asarray([]) is a float array
    if not numpy.issubdtype(indices.dtype, numpy.integer):
        raise TypeError(
            f"Input '{name}' must be an integer array. Got an array with dtype {indices.dtype}."
        )
    if indices.ndim != 1:
        raise ValueError(
            f"Input '{name}' must be a 1D array of shape (R,). Got shape {indices.shape}."
        )
    if indices.size > 0 and (indices.min() < 0 or indices.max() >= size):
        raise IndexError(
            f"Input '{name}' contains indices out of the range [0, {size}). "
            f"Got min={indices.min()} and max={indices.max()}."
        )
    return indices.astype(numpy.int64, copy=False)


def _as_integer_array(
    array: ArrayLike,
    name: str,
    *,
    shape: Tuple[Optional[int], ...],
    shape_hint: Optional[str] = None,
    accept_1d_as_column: bool = False,
    copy: bool = False,
) -> numpy.ndarray:
    r"""
    Convert ``array`` to a ``numpy.int64`` array and check its shape.

    Unlike ``_as_array(..., dtype=numpy.int64)``, the values are never truncated:
    floats are accepted only if they are finite and integral (``2.0`` is accepted,
    ``2.7`` is rejected), and booleans are rejected. The result is always finite, so
    there is no ``finite`` parameter.

    .. important::

        Internal use only.

    The parameters are the same as ``_as_array`` (without ``dtype`` and ``finite``).

    """
    raw = numpy.asarray(array)
    if raw.size == 0:
        raw = raw.astype(numpy.int64)  # numpy.asarray([]) is a float array
    if raw.dtype == numpy.bool_ or not (
        numpy.issubdtype(raw.dtype, numpy.integer)
        or numpy.issubdtype(raw.dtype, numpy.floating)
    ):
        raise TypeError(
            f"Input '{name}' must be an integer array. Got an array with dtype {raw.dtype}."
        )
    if numpy.issubdtype(raw.dtype, numpy.floating) and not numpy.all(
        numpy.isfinite(raw) & (raw == numpy.round(raw))
    ):
        raise ValueError(
            f"Input '{name}' must contain integer values. Got non-integral or "
            f"non-finite floats."
        )
    return _as_array(
        raw, name, numpy.int64,
        shape=shape, finite=True, shape_hint=shape_hint,
        accept_1d_as_column=accept_1d_as_column, copy=copy,
    )