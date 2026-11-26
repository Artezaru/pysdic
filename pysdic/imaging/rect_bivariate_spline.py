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

from numbers import Integral, Real
from typing import Dict, List, Sequence, Tuple, Union

import numpy
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline as _ScipyRectBivariateSpline

from .._validation import (
    _as_array,
    _as_integer_array,
    _validate_integer,
    _validate_real,
)


# Maximum spline degree supported by scipy.interpolate.RectBivariateSpline (FITPACK).
_MAX_SPLINE_DEGREE = 5


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_degree(value: Integral, name: str) -> int:
    value = _validate_integer(value, name, min_value=1)
    if value > _MAX_SPLINE_DEGREE:
        raise ValueError(f"Input '{name}' must be lower or equal to {_MAX_SPLINE_DEGREE}. Got {value}.")
    return value


def _basis_functions(
    knots: numpy.ndarray, x: numpy.ndarray, degree: int, orders: Sequence[int]
) -> Tuple[numpy.ndarray, Dict[int, numpy.ndarray]]:
    r"""
    Compute the ``degree + 1`` B-spline basis functions non-zero at the points ``x`` and
    their derivatives.

    The points are clamped to the domain :math:`[t_k, t_{n-k-1}]` of the spline (as
    FITPACK), NaN coordinates stay NaN. The basis functions of degree :math:`d` are built
    from those of degree :math:`d - 1` (Cox-de Boor):

    .. math::

        B_{i,d}(x) = \frac{x - t_i}{t_{i+d} - t_i} B_{i,d-1}(x)
        + \frac{t_{i+d+1} - x}{t_{i+d+1} - t_{i+1}} B_{i+1,d-1}(x)

    The derivative of order :math:`m` is built from the basis functions of degree
    :math:`k - m`, with :math:`m` derivative steps:

    .. math::

        \frac{d}{dx} B_{i,d}(x) = d \left( \frac{B_{i,d-1}(x)}{t_{i+d} - t_i}
        - \frac{B_{i+1,d-1}(x)}{t_{i+d+1} - t_{i+1}} \right)

    .. important::

        Internal use only.

    Parameters
    ----------
    knots : numpy.ndarray
        The knots :math:`t` of the spline along the axis, shape :math:`(n,)`.

    x : numpy.ndarray
        The coordinates of the points along the axis, shape :math:`(N,)`.

    degree : int
        The degree :math:`k` of the spline along the axis.

    orders : Sequence[int]
        The derivative orders to compute, each in :math:`[0, k]`.

    Returns
    -------
    first : numpy.ndarray
        The index of the first non-zero basis function (and coefficient) of each point,
        shape :math:`(N,)`.

    basis : Dict[int, numpy.ndarray]
        For each order of ``orders``, the derivatives of the non-zero basis functions at
        the points, shape :math:`(N, k + 1)`.

    """
    x = numpy.clip(x, knots[degree], knots[-degree - 1])
    # Interval [t_i, t_{i+1}) of each point, with k <= i <= n - k - 2 (the last interval is closed)
    i = numpy.searchsorted(knots, x, side="right") - 1
    i = numpy.clip(i, degree, knots.size - degree - 2)

    def step(values: numpy.ndarray, d: int, derivative: bool) -> numpy.ndarray:
        # From the d functions of degree d - 1 to the d + 1 functions of degree d (or their derivatives)
        out = numpy.zeros((x.size, d + 1), dtype=numpy.float64)
        for r in range(d):
            left, right = knots[i + r + 1 - d], knots[i + r + 1]
            if derivative:
                weight = d * values[:, r] / (right - left)
                out[:, r] -= weight
                out[:, r + 1] += weight
            else:
                weight = values[:, r] / (right - left)
                out[:, r] += (right - x) * weight
                out[:, r + 1] += (x - left) * weight
        return out

    # Values of the basis functions of the degrees k - max(orders), ..., k
    lowest_degree = degree - max(orders)
    values_by_degree = {}
    values = numpy.ones((x.size, 1), dtype=numpy.float64)
    for d in range(0, degree + 1):
        if d > 0:
            values = step(values, d, derivative=False)
        if d >= lowest_degree:
            values_by_degree[d] = values

    basis = {}
    for m in orders:
        out = values_by_degree[degree - m]
        for d in range(degree - m + 1, degree + 1):
            out = step(out, d, derivative=True)
        basis[m] = out
    return i - degree, basis


# =============================================================================
# RectBivariateSpline
# =============================================================================
class RectBivariateSpline(object):
    r"""
    A bivariate spline on a rectangular grid with one or several channels, built by
    :class:`scipy.interpolate.RectBivariateSpline` (one scipy spline per channel), which
    evaluates all the channels and several derivatives in a single pass (see
    :meth:`evaluate`).

    .. note::

        The construction of the splines is done by scipy
        (:class:`scipy.interpolate.RectBivariateSpline`, one spline per channel): this class
        does not change the splines. Its purpose is the evaluation: when several splines
        (the channels) and several derivative orders are evaluated at the same points, the
        search of the interval of knots of each point and the computation of the B-spline
        basis functions are done **only once**, and shared by all the channels and all the
        derivative orders. Scipy would repeat them for each call of ``ev`` (one call per
        channel and per derivative order).

    The knots and the coefficients are those of scipy: the values and the derivatives are
    the same as ``scipy.interpolate.RectBivariateSpline.ev`` up to the rounding errors. Only
    the evaluation is rewritten (vectorized with numpy). The channels share the same knots
    (the knots of an interpolating spline only depend on the grid), so for each point:

    1. the interval of knots is found once along each axis (binary search),
    2. the non-zero basis functions and their derivatives are computed once along each axis,
    3. the :math:`(k_x + 1) \times (k_y + 1)` coefficients of all the channels are gathered once,

    and all the requested derivatives of all the channels are combined from them:

    .. math::

        \frac{\partial^{d_x + d_y} f_c}{\partial x^{d_x} \partial y^{d_y}}(x, y)
        = \sum_{a=0}^{k_x} \sum_{b=0}^{k_y} c_{i+a, j+b, c} \, B^{(d_x)}_{i+a}(x) \, B^{(d_y)}_{j+b}(y)

    Evaluating the value and the gradient is then much faster than three calls of ``ev``
    per channel (whose search of the knots is also linear in the number of knots).

    As in scipy, ``x`` is the first axis of the grid (e.g. the rows of an image) and ``y``
    the second one (the columns). The points out of the domain of the spline are clamped to
    its bounds (as scipy), and the points with a NaN coordinate get NaN.

    .. seealso::

        - :class:`pysdic.Image`, which builds one spline with the channels of its ROI.

    Parameters
    ----------
    x : ArrayLike
        The strictly increasing coordinates of the grid along the first axis, shape
        :math:`(M,)`.

    y : ArrayLike
        The strictly increasing coordinates of the grid along the second axis, shape
        :math:`(P,)`.

    z : ArrayLike
        The values on the grid (finite values), shape :math:`(M, P)` for a single channel or
        :math:`(M, P, C)` for :math:`C` channels. The shape gives the shape of the evaluations
        (see :meth:`evaluate`).

    kx : Integral, optional (default: ``3``)
        The degree of the spline along ``x``, between 1 and 5 (:math:`M > k_x`).

    ky : Integral, optional (default: ``3``)
        The degree of the spline along ``y``, between 1 and 5 (:math:`P > k_y`).

    s : Real, optional (default: ``0.0``)
        The non-negative smoothing factor of scipy (0: interpolation of the grid values).
        With :math:`s > 0` the knots depend on the values: the channels must still get the
        same knots.

    chunk_size : Integral, optional (default: ``65536``)
        The number of points evaluated together: the memory used by an evaluation is
        proportional to it, and smaller chunks stay in the cache.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If an input has a wrong shape or value, or if the channels get different knots
        (see also scipy).

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.imaging.rect_bivariate_spline import RectBivariateSpline

        image = numpy.random.rand(500, 600, 3)  # 3 channels
        spline = RectBivariateSpline(numpy.arange(500.0), numpy.arange(600.0), image)

        u, v = numpy.random.uniform(0, 499, 1000), numpy.random.uniform(0, 599, 1000)
        value, d_du, d_dv = spline.evaluate(u, v, ((0, 0), (1, 0), (0, 1)))  # each (1000, 3)

    """

    __slots__ = ["_scipy", "_multichannel", "_kx", "_ky", "_tx", "_ty", "_coefficients", "_chunk_size"]

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        kx: Integral = 3,
        ky: Integral = 3,
        s: Real = 0.0,
        chunk_size: Integral = 65536,
    ) -> None:
        x = _as_array(x, "x", numpy.float64, shape=(None,), finite=True)
        y = _as_array(y, "y", numpy.float64, shape=(None,), finite=True)
        multichannel = numpy.ndim(z) == 3
        z = _as_array(
            z, "z", numpy.float64,
            shape=(x.size, y.size, None) if multichannel else (x.size, y.size),
            shape_hint="(M, P, C)" if multichannel else "(M, P) or (M, P, C)",
            finite=True,
        )
        if multichannel and z.shape[2] == 0:
            raise ValueError("Input 'z' must have at least one channel. Got shape (M, P, 0).")
        kx = _validate_degree(kx, "kx")
        ky = _validate_degree(ky, "ky")
        s = _validate_real(s, "s", non_negative=True, finite=True)
        channels = [z[:, :, c] for c in range(z.shape[2])] if multichannel else [z]
        splines = [_ScipyRectBivariateSpline(x, y, values, kx=kx, ky=ky, s=s) for values in channels]
        self._set_scipy(splines, multichannel, chunk_size)

    @classmethod
    def from_scipy(
        cls,
        splines: Union[_ScipyRectBivariateSpline, Sequence[_ScipyRectBivariateSpline]],
        *,
        chunk_size: Integral = 65536,
    ) -> RectBivariateSpline:
        r"""
        Create a spline from :class:`scipy.interpolate.RectBivariateSpline` already built
        (their knots and coefficients are used, not recomputed).

        Parameters
        ----------
        splines : Union[:class:`scipy.interpolate.RectBivariateSpline`, Sequence[:class:`scipy.interpolate.RectBivariateSpline`]]
            A scipy spline (single channel), or a list/tuple of scipy splines with the same
            degrees and knots (one per channel, multi-channel).

        chunk_size : Integral, optional (default: ``65536``)
            The number of points evaluated together.

        Returns
        -------
        :class:`RectBivariateSpline`
            The spline.

        Raises
        ------
        TypeError
            If a spline is not a scipy RectBivariateSpline.

        ValueError
            If the list is empty, or the splines have different degrees or knots.

        """
        multichannel = isinstance(splines, (list, tuple))
        splines = list(splines) if multichannel else [splines]
        if len(splines) == 0:
            raise ValueError("Input 'splines' must contain at least one spline.")
        for k, spline in enumerate(splines):
            if not isinstance(spline, _ScipyRectBivariateSpline):
                raise TypeError(
                    f"Input 'splines[{k}]' must be a scipy.interpolate.RectBivariateSpline. Got type {type(spline)}."
                )
        new = cls.__new__(cls)
        new._set_scipy(splines, multichannel, chunk_size)
        return new

    def _set_scipy(
        self, splines: List[_ScipyRectBivariateSpline], multichannel: bool, chunk_size: Integral
    ) -> None:
        # Shared knots and stacked coefficients (n_cx, n_cy, C) of the scipy splines (one per channel)
        self._chunk_size = _validate_integer(chunk_size, "chunk_size", min_value=1)
        kx, ky = (int(k) for k in splines[0].degrees)
        tx, ty = (numpy.ascontiguousarray(t, dtype=numpy.float64) for t in splines[0].get_knots())
        for c, spline in enumerate(splines[1:], start=1):
            if tuple(int(k) for k in spline.degrees) != (kx, ky):
                raise ValueError(f"All the channels must have the same degrees. Got {(kx, ky)} and {spline.degrees} (channel {c}).")
            other_tx, other_ty = spline.get_knots()
            if not (numpy.array_equal(other_tx, tx) and numpy.array_equal(other_ty, ty)):
                raise ValueError(
                    f"All the channels must have the same knots (channel {c} differs from channel 0). "
                    f"This is the case for interpolating splines (s=0) on the same grid."
                )
        shape = (tx.size - kx - 1, ty.size - ky - 1)
        coefficients = numpy.stack(
            [numpy.asarray(spline.get_coeffs(), dtype=numpy.float64).reshape(shape) for spline in splines], axis=-1
        )  # (n_cx, n_cy, C)
        self._scipy = tuple(splines)
        self._multichannel = bool(multichannel)
        self._kx, self._ky = kx, ky
        self._tx, self._ty = tx, ty
        self._coefficients = numpy.ascontiguousarray(coefficients)
        for array in (self._tx, self._ty, self._coefficients):
            array.flags.writeable = False

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def degrees(self) -> Tuple[int, int]:
        r"""
        [Get] The degrees :math:`(k_x, k_y)` of the spline.

        Returns
        -------
        Tuple[int, int]
            The degrees along ``x`` and ``y``.
        """
        return self._kx, self._ky

    @property
    def n_channels(self) -> int:
        r"""
        [Get] The number of channels :math:`C` (1 for a single channel).

        Returns
        -------
        int
            The number of channels.
        """
        return self._coefficients.shape[2]

    @property
    def is_multichannel(self) -> bool:
        r"""
        [Get] Whether the spline was built from values :math:`(M, P, C)` (or a list of scipy
        splines): the evaluations then have a last dimension :math:`C` (even for
        :math:`C = 1`).

        Returns
        -------
        bool
            True for a multi-channel spline.
        """
        return self._multichannel

    @property
    def chunk_size(self) -> int:
        r"""
        [Get] The number of points evaluated together.

        Returns
        -------
        int
            The size of the chunks of points.
        """
        return self._chunk_size

    @property
    def scipy_splines(self) -> Tuple[_ScipyRectBivariateSpline, ...]:
        r"""
        [Get] The scipy splines, one per channel (e.g. for their methods ``integral`` or
        ``partial_derivative``).

        Returns
        -------
        Tuple[:class:`scipy.interpolate.RectBivariateSpline`, ...]
            The scipy splines used to build this spline.
        """
        return self._scipy

    def get_knots(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Get the knots of the spline along ``x`` and ``y``, shared by the channels (read-only).

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            The knots :math:`t_x` and :math:`t_y`.
        """
        return self._tx, self._ty

    def get_coeffs(self) -> numpy.ndarray:
        r"""
        Get the coefficients of the spline (read-only).

        Returns
        -------
        numpy.ndarray
            The coefficients, shape :math:`(n_x - k_x - 1, n_y - k_y - 1, C)` (scipy returns
            them flattened, one array per channel).
        """
        return self._coefficients

    def __repr__(self) -> str:
        return (
            f"RectBivariateSpline(degrees={self.degrees}, n_channels={self.n_channels}, "
            f"is_multichannel={self._multichannel}, coefficients={self._coefficients.shape[:2]}, "
            f"chunk_size={self._chunk_size})"
        )

    # ===================================================================
    # Evaluation
    # ===================================================================
    def _validate_derivatives(self, derivatives: ArrayLike) -> List[Tuple[int, int]]:
        # List of the orders (dx, dy) with 0 <= dx < kx and 0 <= dy < ky
        orders = _as_integer_array(derivatives, "derivatives", shape=(None, 2), shape_hint="(R, 2)")
        if orders.shape[0] == 0:
            raise ValueError("Input 'derivatives' must contain at least one order (dx, dy).")
        if numpy.any(orders < 0) or numpy.any(orders[:, 0] >= self._kx) or numpy.any(orders[:, 1] >= self._ky):
            raise ValueError(
                f"The derivative orders (dx, dy) must satisfy 0 <= dx < kx={self._kx} and "
                f"0 <= dy < ky={self._ky}. Got {orders.tolist()}."
            )
        return [(int(dx), int(dy)) for dx, dy in orders]

    def evaluate(
        self,
        xi: ArrayLike,
        yi: ArrayLike,
        derivatives: ArrayLike = ((0, 0),),
    ) -> Tuple[numpy.ndarray, ...]:
        r"""
        Evaluate all the channels of the spline and/or several of their derivatives at the
        points :math:`(x_i, y_i)` in a single pass.

        Parameters
        ----------
        xi : ArrayLike
            The coordinates of the points along ``x``, any shape :math:`S`.

        yi : ArrayLike
            The coordinates of the points along ``y``, same shape as ``xi``.

        derivatives : ArrayLike, optional (default: ``((0, 0),)``)
            The orders :math:`(d_x, d_y)` to compute, shape :math:`(R, 2)`, with
            :math:`0 \leq d_x < k_x` and :math:`0 \leq d_y < k_y`. For example
            ``((0, 0), (1, 0), (0, 1))`` for the value and the gradient.

        Returns
        -------
        Tuple[numpy.ndarray, ...]
            :math:`R` arrays in float64 dtype, one per order of ``derivatives`` (in the same
            order), of shape :math:`S` for a single channel or :math:`(*S, C)` for a
            multi-channel spline.

        Raises
        ------
        ValueError
            If ``xi`` and ``yi`` have different shapes, or if an order is out of range.

        """
        xi = numpy.asarray(xi, dtype=numpy.float64)
        yi = numpy.asarray(yi, dtype=numpy.float64)
        if xi.shape != yi.shape:
            raise ValueError(f"Inputs 'xi' and 'yi' must have the same shape. Got {xi.shape} and {yi.shape}.")
        orders = self._validate_derivatives(derivatives)

        shape = xi.shape
        xi, yi = xi.ravel(), yi.ravel()
        C = self.n_channels
        outputs = [numpy.empty((xi.size, C), dtype=numpy.float64) for _ in orders]
        for start in range(0, xi.size, self._chunk_size):
            chunk = slice(start, start + self._chunk_size)
            for output, values in zip(outputs, self._evaluate_chunk(xi[chunk], yi[chunk], orders)):
                output[chunk] = values
        if self._multichannel:
            return tuple(output.reshape(shape + (C,)) for output in outputs)
        return tuple(output[:, 0].reshape(shape) for output in outputs)

    def _evaluate_chunk(
        self, x: numpy.ndarray, y: numpy.ndarray, orders: List[Tuple[int, int]]
    ) -> List[numpy.ndarray]:
        # Derivatives of the given orders of all the channels at the points (N,), shape (N, C) each
        kx, ky = self._kx, self._ky
        n_cx, n_cy, C = self._coefficients.shape
        x_orders = sorted({dx for dx, _ in orders})
        y_orders = sorted({dy for _, dy in orders})
        first_x, basis_x = _basis_functions(self._tx, x, kx, x_orders)  # Search of the knots: once for all the channels
        first_y, basis_y = _basis_functions(self._ty, y, ky, y_orders)

        flat = self._coefficients.reshape(n_cx * n_cy, C)
        base = first_x * n_cy + first_y  # Flat index of the first coefficient of each point
        results = {order: numpy.zeros((x.size, C), dtype=numpy.float64) for order in orders}
        for a in range(kx + 1):
            # The row a of the (kx + 1) x (ky + 1) block of coefficients (N, C), contracted along y
            row = [flat[base + (a * n_cy + b)] for b in range(ky + 1)]
            contracted = {
                dy: sum(row[b] * basis_y[dy][:, b, numpy.newaxis] for b in range(ky + 1)) for dy in y_orders
            }
            for dx, dy in results:
                results[(dx, dy)] += basis_x[dx][:, a, numpy.newaxis] * contracted[dy]
        return [results[order] for order in orders]

    def ev(self, xi: ArrayLike, yi: ArrayLike, dx: Integral = 0, dy: Integral = 0) -> numpy.ndarray:
        r"""
        Evaluate the spline or one of its derivatives at the points :math:`(x_i, y_i)`,
        as ``scipy.interpolate.RectBivariateSpline.ev`` (all the channels).

        Parameters
        ----------
        xi : ArrayLike
            The coordinates of the points along ``x``, any shape :math:`S`.

        yi : ArrayLike
            The coordinates of the points along ``y``, same shape as ``xi``.

        dx : Integral, optional (default: ``0``)
            The order of the derivative along ``x``, :math:`0 \leq d_x < k_x`.

        dy : Integral, optional (default: ``0``)
            The order of the derivative along ``y``, :math:`0 \leq d_y < k_y`.

        Returns
        -------
        numpy.ndarray
            The values in float64 dtype, shape :math:`S` (single channel) or :math:`(*S, C)`.

        """
        dx = _validate_integer(dx, "dx", min_value=0)
        dy = _validate_integer(dy, "dy", min_value=0)
        return self.evaluate(xi, yi, ((dx, dy),))[0]