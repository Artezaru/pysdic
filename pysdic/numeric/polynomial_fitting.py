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

from __future__ import annotations

from math import comb
from numbers import Integral
from typing import Union

import numpy
from numpy.typing import ArrayLike

from .._validation import _as_array, _validate_integer


__all__ = [
    "fit_polynomial",
    "evaluate_polynomial",
    "polynomial_predict",
]


def _normalize_axis(axis: Integral, ndim: int) -> int:
    axis = _validate_integer(axis, "axis")
    if not -ndim <= axis < ndim:
        raise ValueError(f"Input 'axis' must be in [{-ndim}, {ndim - 1}]. Got {axis}.")
    return axis % ndim


def fit_polynomial(x: ArrayLike, data: ArrayLike, degree: Integral, *, axis: Integral = -1) -> numpy.ndarray:
    r"""
    Fit a polynomial :math:`P(x) = c_0 + c_1 x + \dots + c_d x^d` to sampled data, in the
    least squares sense, along an axis.

    All the signals along the axis are fitted at once. To keep the problem well
    conditioned, the fit is computed with the centered and scaled variable
    :math:`x' = (x - \bar{x}) / s` (:math:`\bar{x}` the mean and :math:`s` the half range
    of :math:`x`), and the coefficients are converted back to the variable :math:`x`.

    Parameters
    ----------
    x : ArrayLike
        The finite sampling points, of shape :math:`(N,)` with at least :math:`d + 1`
        distinct values.

    data : ArrayLike
        The finite sampled values, with :math:`N` values along ``axis``.

    degree : Integral
        The degree :math:`d \geq 0` of the polynomial.

    axis : Integral, optional (default: ``-1``)
        The axis of the sampling in ``data``.

    Returns
    -------
    numpy.ndarray
        The coefficients :math:`(c_0, \dots, c_d)` in increasing degree, with the shape of
        ``data`` where the axis has the length :math:`d + 1`.

    Raises
    ------
    ValueError
        If an input has a wrong shape or non-finite values, or there are not enough
        distinct sampling points for the degree.

    Examples
    --------
    .. code-block:: python

        import numpy
        from pysdic.numeric import fit_polynomial

        x = numpy.linspace(0, 1, 10)
        fit_polynomial(x, 1 + 2 * x - x**2, 2)  # [1.0, 2.0, -1.0]

    """
    x = _as_array(x, "x", numpy.float64, shape=(None,), shape_hint="(N,)", finite=True)
    data = numpy.asarray(data, dtype=numpy.float64)
    if data.ndim == 0:
        raise ValueError("Input 'data' must have at least one dimension.")
    if not numpy.all(numpy.isfinite(data)):
        raise ValueError("Input 'data' must have finite values.")
    axis = _normalize_axis(axis, data.ndim)
    degree = _validate_integer(degree, "degree", min_value=0)
    if data.shape[axis] != x.size:
        raise ValueError(f"Input 'data' must have {x.size} values along the axis {axis}. Got {data.shape[axis]}.")
    if numpy.unique(x).size < degree + 1:
        raise ValueError(f"A polynomial of degree {degree} requires at least {degree + 1} distinct sampling points.")

    K = degree + 1
    center = float(numpy.mean(x))
    scale = float(numpy.max(numpy.abs(x - center))) or 1.0
    vandermonde = numpy.vander((x - center) / scale, K, increasing=True)  # (N, K)

    moved = numpy.moveaxis(data, axis, -1)  # (..., N)
    values = moved.reshape((-1, x.size)).T  # (N, M)
    scaled_coefficients = numpy.linalg.lstsq(vandermonde, values, rcond=None)[0]  # (K, M)

    # P(x) = sum_j c'_j ((x - center) / scale)^j  ->  c_k = sum_{j >= k} B[k, j] c'_j
    conversion = numpy.zeros((K, K))
    for j in range(K):
        for k in range(j + 1):
            conversion[k, j] = comb(j, k) * (-center) ** (j - k) / scale**j
    coefficients = (conversion @ scaled_coefficients).T.reshape(moved.shape[:-1] + (K,))
    return numpy.moveaxis(coefficients, -1, axis)


def evaluate_polynomial(coefficients: ArrayLike, x: Union[float, ArrayLike], *, axis: Integral = -1) -> numpy.ndarray:
    r"""
    Evaluate polynomials given by their coefficients in increasing degree (Horner scheme).

    Parameters
    ----------
    coefficients : ArrayLike
        The coefficients :math:`(c_0, \dots, c_d)` along ``axis`` (e.g. the output of
        :func:`fit_polynomial`).

    x : Union[float, ArrayLike]
        The evaluation points, a scalar or an array of shape :math:`(N',)`.

    axis : Integral, optional (default: ``-1``)
        The axis of the coefficients.

    Returns
    -------
    numpy.ndarray
        The values, with the shape of ``coefficients`` where the axis has the length
        :math:`N'` (or is removed for a scalar ``x``).

    """
    coefficients = numpy.asarray(coefficients, dtype=numpy.float64)
    if coefficients.ndim == 0:
        raise ValueError("Input 'coefficients' must have at least one dimension.")
    axis = _normalize_axis(axis, coefficients.ndim)
    x = numpy.asarray(x, dtype=numpy.float64)
    scalar = x.ndim == 0
    x = numpy.atleast_1d(x)
    if x.ndim != 1:
        raise ValueError(f"Input 'x' must be a scalar or a 1D array. Got shape {x.shape}.")

    moved = numpy.moveaxis(coefficients, axis, -1)[..., numpy.newaxis, :]  # (..., 1, K)
    values = numpy.zeros(moved.shape[:-2] + (x.size,))
    for k in range(moved.shape[-1] - 1, -1, -1):
        values = values * x + moved[..., 0, k : k + 1]
    values = numpy.moveaxis(values, -1, axis)
    return numpy.squeeze(values, axis=axis) if scalar else values


def polynomial_predict(
    x: ArrayLike,
    data: ArrayLike,
    degree: Integral,
    new_x: Union[float, ArrayLike],
    *,
    axis: Integral = -1,
) -> numpy.ndarray:
    r"""
    Fit a polynomial to sampled data along an axis and evaluate it at new points
    (interpolation or extrapolation).

    Shortcut for :func:`fit_polynomial` followed by :func:`evaluate_polynomial`.

    Parameters
    ----------
    x, data, degree, axis
        See :func:`fit_polynomial`.

    new_x : Union[float, ArrayLike]
        The evaluation points, a scalar or an array of shape :math:`(N',)`.

    Returns
    -------
    numpy.ndarray
        The predicted values, with the shape of ``data`` where the axis has the length
        :math:`N'` (or is removed for a scalar ``new_x``).

    """
    data = numpy.asarray(data, dtype=numpy.float64)
    coefficients = fit_polynomial(x, data, degree, axis=axis)
    return evaluate_polynomial(coefficients, new_x, axis=_normalize_axis(axis, data.ndim))
