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

r"""
Finite difference approximations of the derivatives of sampled signals.

All the stencils are computed with the algorithm of Fornberg, for any set of sampling
points (uniform or not). A derivative of order :math:`n` is approximated at the
precision :math:`p` (error :math:`O(h^p)`):

- with :math:`n + p` points in general (one-sided stencils, non-uniform sampling);
- with :math:`2 \lfloor (n + p - 1) / 2 \rfloor + 1` points for a centered stencil on a
  uniform sampling (the symmetry cancels one more term of the Taylor expansion), with
  :math:`p` even.
"""


from numbers import Integral, Real
from typing import List, Optional, Tuple, Union


import numpy
import scipy.ndimage
import scipy.sparse
from numpy.typing import ArrayLike

from .._validation import _as_array, _validate_bool, _validate_choice, _validate_integer, _validate_real


__all__ = [
    "compute_finite_difference_coefficients",
    "compute_forward_finite_difference_coefficients",
    "compute_backward_finite_difference_coefficients",
    "compute_central_finite_difference_coefficients",
    "assemble_finite_difference_matrix",
    "assemble_forward_finite_difference_matrix",
    "assemble_backward_finite_difference_matrix",
    "assemble_central_finite_difference_matrix",
    "apply_forward_finite_difference",
    "apply_backward_finite_difference",
    "apply_central_finite_difference",
]

_SCHEMES = ("forward", "backward", "central")
_BOUNDARIES = ("one_sided", "valid")
_APPLY_MODES = ("one_sided", "reflect", "constant", "nearest", "mirror", "wrap")

# Relative tolerance to consider a centered stencil as symmetric (uniform sampling)
_SYMMETRY_TOLERANCE = 1e-9


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_order_accuracy(order: Integral, accuracy: Integral, even_accuracy: bool) -> Tuple[int, int]:
    order = _validate_integer(order, "order", min_value=1)
    accuracy = _validate_integer(accuracy, "accuracy", min_value=1)
    if even_accuracy and accuracy % 2 != 0:
        raise ValueError(f"Input 'accuracy' must be an even integer for central stencils. Got {accuracy}.")
    return order, accuracy


def _fornberg(offsets: numpy.ndarray, order: int) -> numpy.ndarray:
    # Weights (L,) of the derivative of order 'order' at 0 from the values at the offsets
    # (B. Fornberg, Generation of finite difference formulas on arbitrarily spaced grids,
    # Mathematics of Computation 51, 1988)
    L = offsets.size
    c = numpy.zeros((L, order + 1))
    c[0, 0] = 1.0
    c1, c4 = 1.0, offsets[0]
    for i in range(1, L):
        mn = min(i, order)
        c2, c5, c4 = 1.0, c4, offsets[i]
        for j in range(i):
            c3 = offsets[i] - offsets[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, order]


def _window(index: int, size: int, n_points: int, scheme: str) -> Tuple[int, bool]:
    # Start of the window of 'size' consecutive points used at the point 'index' (among
    # n_points), and whether the window had to be shifted to stay in [0, n_points)
    if scheme == "forward":
        start = index
    elif scheme == "backward":
        start = index - size + 1
    else:
        start = index - (size - 1) // 2
    clamped = min(max(start, 0), n_points - size)
    return clamped, clamped != start


def _stencils(
    times: numpy.ndarray, order: int, accuracy: int, scheme: str, boundary: str, max_gap: Optional[float]
) -> Tuple[List[int], List[numpy.ndarray], List[numpy.ndarray]]:
    # For each evaluated point: its index, the indices of its stencil and the weights
    general_size = order + accuracy  # Points of a one-sided or non-uniform stencil
    centered_size = 2 * ((order + accuracy - 1) // 2) + 1  # Points of a symmetric stencil

    # Segments of the sampling separated by the gaps larger than max_gap
    cuts = [0]
    if max_gap is not None:
        cuts += list(numpy.flatnonzero(numpy.diff(times) > max_gap) + 1)
    cuts.append(times.size)

    rows, columns, weights = [], [], []
    for first, last in zip(cuts[:-1], cuts[1:]):
        segment = times[first:last]
        n_points = segment.size
        for index in range(n_points):
            stencil = None
            if scheme == "central" and n_points >= centered_size:
                start, clamped = _window(index, centered_size, n_points, scheme)
                offsets = segment[start : start + centered_size] - segment[index]
                symmetric = numpy.allclose(
                    offsets, -offsets[::-1], rtol=0.0, atol=_SYMMETRY_TOLERANCE * numpy.ptp(offsets)
                )
                if not clamped and symmetric:
                    stencil = (start, centered_size)
            if stencil is None:
                if n_points < general_size:
                    continue  # Segment too short for the stencil
                start, clamped = _window(index, general_size, n_points, scheme)
                if clamped and boundary == "valid":
                    continue
                stencil = (start, general_size)
            start, size = stencil
            offsets = segment[start : start + size] - segment[index]
            rows.append(first + index)
            columns.append(first + numpy.arange(start, start + size))
            weights.append(_fornberg(offsets, order))
    return rows, columns, weights


def _assemble(
    times: numpy.ndarray,
    order: int,
    accuracy: int,
    scheme: str,
    boundary: str,
    max_gap: Optional[float],
    n_dim: int,
    sparse: bool,
) -> Tuple[Union[numpy.ndarray, scipy.sparse.csr_matrix], numpy.ndarray]:
    # Operator (N_r n_dim, N_t n_dim) in time-major order and the evaluated indices (N_r,)
    rows, columns, weights = _stencils(times, order, accuracy, scheme, boundary, max_gap)
    N_t, N_r = times.size, len(rows)
    if N_r == 0:
        matrix = scipy.sparse.csr_matrix((0, N_t * n_dim))
        return (matrix if sparse else matrix.toarray()), numpy.zeros(0, dtype=numpy.int64)
    counts = [w.size for w in weights]
    T = scipy.sparse.csr_matrix(
        (numpy.concatenate(weights), (numpy.repeat(numpy.arange(N_r), counts), numpy.concatenate(columns))),
        shape=(N_r, N_t),
    )
    matrix = scipy.sparse.kron(T, scipy.sparse.identity(n_dim, format="csr"), format="csr")
    return (matrix if sparse else matrix.toarray()), numpy.asarray(rows, dtype=numpy.int64)


def _validate_times(times: ArrayLike) -> numpy.ndarray:
    times = _as_array(times, "times", numpy.float64, shape=(None,), shape_hint="(N_t,)", finite=True)
    if numpy.any(numpy.diff(times) <= 0):
        raise ValueError("Input 'times' must be strictly increasing.")
    return times


def _uniform_times(n_times: Integral, spacing: Real) -> numpy.ndarray:
    n_times = _validate_integer(n_times, "n_times", min_value=1)
    spacing = _validate_real(spacing, "spacing", positive=True, finite=True)
    return numpy.arange(n_times) * spacing


# =============================================================================
# Coefficients
# =============================================================================
def compute_finite_difference_coefficients(offsets: ArrayLike, order: Integral) -> numpy.ndarray:
    r"""
    Compute the finite difference coefficients of a derivative for any set of sampling
    points.

    The coefficients :math:`c_j` approximate the derivative of order :math:`n` at a point
    :math:`t` from the values at the points :math:`t + \delta_j`:

    .. math::

        \frac{d^n f}{dt^n}(t) \approx \sum_{j} c_j \, f(t + \delta_j)

    They are exact for the polynomials of degree :math:`L - 1` (:math:`L` points), so the
    error is :math:`O(h^{L - n})` with :math:`h` the largest offset (one more order for a
    symmetric stencil). They are computed with the algorithm of Fornberg, stable for any
    distinct offsets.

    Parameters
    ----------
    offsets : ArrayLike
        The distinct finite offsets :math:`\delta_j` of the sampling points from the
        evaluation point, of shape :math:`(L,)` with :math:`L > n`, in time units.

    order : Integral
        The order :math:`n \geq 1` of the derivative.

    Returns
    -------
    numpy.ndarray
        The coefficients :math:`c_j`, of shape :math:`(L,)`, in the order of ``offsets``
        (they include the spacing: no division by :math:`h^n` is needed).

    Raises
    ------
    ValueError
        If the offsets are not distinct, or there are not more offsets than the order.

    Examples
    --------
    First derivative at :math:`t = 1` from the samples at :math:`t = 0, 1, 3`:

    .. code-block:: python

        from pysdic.numeric import compute_finite_difference_coefficients

        compute_finite_difference_coefficients([-1.0, 0.0, 2.0], 1)
        # array([-0.66666667,  0.5       ,  0.16666667])

    """
    offsets = _as_array(offsets, "offsets", numpy.float64, shape=(None,), shape_hint="(L,)", finite=True)
    order = _validate_integer(order, "order", min_value=1)
    if offsets.size <= order:
        raise ValueError(f"The derivative of order {order} requires more than {order} offsets. Got {offsets.size}.")
    if numpy.unique(offsets).size != offsets.size:
        raise ValueError("Input 'offsets' must contain distinct values.")
    return _fornberg(offsets, order)


def compute_forward_finite_difference_coefficients(
    order: Integral, spacing: Real = 1.0, accuracy: Integral = 1
) -> numpy.ndarray:
    r"""
    Compute the coefficients of the forward finite difference approximation of a
    derivative on a uniform sampling.

    .. math::

        \frac{d^n f}{dt^n}(t) = \sum_{j=0}^{n + p - 1} c_j \, f(t + j h) + O(h^p)

    Parameters
    ----------
    order : Integral
        The order :math:`n \geq 1` of the derivative.

    spacing : Real, optional (default: ``1.0``)
        The sampling step :math:`h > 0`.

    accuracy : Integral, optional (default: ``1``)
        The order :math:`p \geq 1` of the error.

    Returns
    -------
    numpy.ndarray
        The coefficients ``[c_0, ..., c_{n+p-1}]`` of shape :math:`(n + p,)`.

    Examples
    --------
    .. code-block:: python

        compute_forward_finite_difference_coefficients(1, accuracy=2)  # [-1.5, 2.0, -0.5]
        compute_forward_finite_difference_coefficients(2, spacing=0.1)  # [100.0, -200.0, 100.0]

    """
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=False)
    spacing = _validate_real(spacing, "spacing", positive=True, finite=True)
    return _fornberg(numpy.arange(order + accuracy) * spacing, order)


def compute_backward_finite_difference_coefficients(
    order: Integral, spacing: Real = 1.0, accuracy: Integral = 1
) -> numpy.ndarray:
    r"""
    Compute the coefficients of the backward finite difference approximation of a
    derivative on a uniform sampling.

    .. math::

        \frac{d^n f}{dt^n}(t) = \sum_{j=0}^{n + p - 1} c_j \, f(t - j h) + O(h^p)

    Parameters
    ----------
    order, spacing, accuracy
        See :func:`compute_forward_finite_difference_coefficients`.

    Returns
    -------
    numpy.ndarray
        The coefficients ``[c_0, ..., c_{n+p-1}]`` of shape :math:`(n + p,)`, for the
        values ``[f(t), f(t - h), ...]``.

    Examples
    --------
    .. code-block:: python

        compute_backward_finite_difference_coefficients(1, accuracy=2)  # [1.5, -2.0, 0.5]

    """
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=False)
    spacing = _validate_real(spacing, "spacing", positive=True, finite=True)
    return _fornberg(-numpy.arange(order + accuracy) * spacing, order)


def compute_central_finite_difference_coefficients(
    order: Integral, spacing: Real = 1.0, accuracy: Integral = 2
) -> numpy.ndarray:
    r"""
    Compute the coefficients of the central finite difference approximation of a
    derivative on a uniform sampling.

    .. math::

        \frac{d^n f}{dt^n}(t) = \sum_{j=-M}^{M} c_j \, f(t + j h) + O(h^p),
        \qquad M = \left\lfloor \frac{n + p - 1}{2} \right\rfloor

    Parameters
    ----------
    order : Integral
        The order :math:`n \geq 1` of the derivative.

    spacing : Real, optional (default: ``1.0``)
        The sampling step :math:`h > 0`.

    accuracy : Integral, optional (default: ``2``)
        The order :math:`p` of the error, a positive even integer.

    Returns
    -------
    numpy.ndarray
        The coefficients ``[c_{-M}, ..., c_M]`` of shape :math:`(2 M + 1,)`.

    Examples
    --------
    .. code-block:: python

        compute_central_finite_difference_coefficients(1)  # [-0.5, 0.0, 0.5]
        compute_central_finite_difference_coefficients(2, accuracy=4)  # [-1/12, 4/3, -5/2, 4/3, -1/12]

    """
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=True)
    spacing = _validate_real(spacing, "spacing", positive=True, finite=True)
    M = (order + accuracy - 1) // 2
    return _fornberg(numpy.arange(-M, M + 1) * spacing, order)


# =============================================================================
# Operator matrices
# =============================================================================
def assemble_finite_difference_matrix(
    times: ArrayLike,
    order: Integral,
    *,
    accuracy: Integral = 2,
    boundary: str = "one_sided",
    max_gap: Optional[Real] = None,
    n_dim: Integral = 1,
    sparse: bool = True,
    return_indices: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix, Tuple]:
    r"""
    Assemble the finite difference operator of a derivative for a signal sampled at any
    strictly increasing times, with uniform or non-uniform steps.

    Each row approximates the derivative of order :math:`n` at one sampling time
    :math:`t_i` from the values at the neighboring sampling times, with coefficients
    computed for the actual offsets (see :func:`compute_finite_difference_coefficients`):

    - **centered stencil** (:math:`2 \lfloor (n + p - 1) / 2 \rfloor + 1` points) when the
      neighbors are uniformly spaced around :math:`t_i`;
    - otherwise (non-uniform steps around :math:`t_i`) **the most centered
      window of** :math:`n + p` **consecutive samples**, which keeps the error in
      :math:`O(h^p)` without symmetry;
    - **near the ends of the sampling**, the window is shifted inside the sampling
      (one-sided stencil of :math:`n + p` points) with ``boundary="one_sided"``, or the
      row is removed with ``boundary="valid"``.

    The operator only uses the given samples: the coefficients of each row depend on the
    actual steps between them. For example, the frames of an analysis without
    displacement are simply not given: the step between the given frames around them is
    larger, and no value is interpolated or extrapolated for them.

    With ``max_gap``, the sampling is split into segments at the steps larger than
    ``max_gap``, and each segment is processed independently (as if the step was an end of
    the sampling): no stencil spans such a step. The segments shorter than the stencil have
    no row.

    The signal can have :math:`D` components at each time (``n_dim``). The values are
    ordered time by time, :math:`F = [f_1(t_1), \dots, f_D(t_1), f_1(t_2), \dots]`, and
    the operator is :math:`T \otimes I_D`.

    Parameters
    ----------
    times : ArrayLike
        The strictly increasing sampling times, of shape :math:`(N_t,)`.

    order : Integral
        The order :math:`n \geq 1` of the derivative.

    accuracy : Integral, optional (default: ``2``)
        The order :math:`p` of the error, a positive even integer.

    boundary : str, optional (default: ``"one_sided"``)
        The treatment of the ends of the sampling (and of the segments): ``"one_sided"`` or
        ``"valid"`` (see above).

    max_gap : Optional[Real], optional
        The largest step between two consecutive samples that a stencil can span. By
        default, no limit.

    n_dim : Integral, optional (default: ``1``)
        The number :math:`D \geq 1` of components of the signal at each time.

    sparse : bool, optional (default: ``True``)
        If True, the operator is a ``scipy.sparse.csr_matrix``, otherwise a dense array.

    return_indices : bool, optional (default: ``False``)
        If True, also return the indices of the sampling times of the rows.

    Returns
    -------
    matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The operator of shape :math:`(N_r D, N_t D)`, :math:`N_r` being the number of
        evaluated times (:math:`N_t` with ``boundary="one_sided"`` and no ``max_gap``).

    indices : numpy.ndarray
        Only with ``return_indices=True``: the indices :math:`(N_r,)` of the sampling
        times of the rows.

    Raises
    ------
    ValueError
        If the times are not strictly increasing or an input has a wrong value.

    Examples
    --------
    Second derivative of a signal sampled at the times 0, 1, 2, 3, 6, 7, 8, 9 (a step of 3
    between 3 and 6):

    .. code-block:: python

        import numpy
        from pysdic.numeric import assemble_finite_difference_matrix

        times = numpy.array([0, 1, 2, 3, 6, 7, 8, 9], dtype=float)
        D = assemble_finite_difference_matrix(times, 2)
        D @ times**2  # 2 everywhere (exact for the polynomials of degree <= 3)

    """
    times = _validate_times(times)
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=True)
    boundary = _validate_choice(boundary, "boundary", _BOUNDARIES)
    max_gap = _validate_real(max_gap, "max_gap", positive=True, finite=True, allow_none=True)
    n_dim = _validate_integer(n_dim, "n_dim", min_value=1)
    sparse = _validate_bool(sparse, "sparse")
    return_indices = _validate_bool(return_indices, "return_indices")
    matrix, indices = _assemble(times, order, accuracy, "central", boundary, max_gap, n_dim, sparse)
    return (matrix, indices) if return_indices else matrix


def _assemble_uniform(
    scheme: str,
    order: Integral,
    n_times: Integral,
    n_dim: Integral,
    spacing: Real,
    accuracy: Integral,
    boundary: str,
    sparse: bool,
    return_indices: bool,
):
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=(scheme == "central"))
    times = _uniform_times(n_times, spacing)
    boundary = _validate_choice(boundary, "boundary", _BOUNDARIES)
    n_dim = _validate_integer(n_dim, "n_dim", min_value=1)
    sparse = _validate_bool(sparse, "sparse")
    return_indices = _validate_bool(return_indices, "return_indices")
    matrix, indices = _assemble(times, order, accuracy, scheme, boundary, None, n_dim, sparse)
    return (matrix, indices) if return_indices else matrix


def assemble_forward_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    *,
    n_dim: Integral = 1,
    spacing: Real = 1.0,
    accuracy: Integral = 1,
    boundary: str = "one_sided",
    sparse: bool = True,
    return_indices: bool = False,
):
    r"""
    Assemble the forward finite difference operator of a derivative on a uniform sampling.

    The row :math:`i` uses the samples :math:`i, \dots, i + n + p - 1`
    (see :func:`compute_forward_finite_difference_coefficients`). At the end of the
    sampling, the window is shifted backward (``boundary="one_sided"``) or the row is
    removed (``boundary="valid"``). See :func:`assemble_finite_difference_matrix` for the
    ordering of the components and the outputs.

    Parameters
    ----------
    order : Integral
        The order :math:`n \geq 1` of the derivative.

    n_times : Integral
        The number of samples :math:`N_t`.

    n_dim, boundary, sparse, return_indices
        See :func:`assemble_finite_difference_matrix`.

    spacing : Real, optional (default: ``1.0``)
        The sampling step :math:`h > 0`.

    accuracy : Integral, optional (default: ``1``)
        The order :math:`p \geq 1` of the error.

    Examples
    --------
    .. code-block:: python

        assemble_forward_finite_difference_matrix(1, 4, sparse=False)
        # [[-1,  1,  0,  0],
        #  [ 0, -1,  1,  0],
        #  [ 0,  0, -1,  1],
        #  [ 0,  0, -1,  1]]  (last row: shifted window)

    """
    return _assemble_uniform("forward", order, n_times, n_dim, spacing, accuracy, boundary, sparse, return_indices)


def assemble_backward_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    *,
    n_dim: Integral = 1,
    spacing: Real = 1.0,
    accuracy: Integral = 1,
    boundary: str = "one_sided",
    sparse: bool = True,
    return_indices: bool = False,
):
    r"""
    Assemble the backward finite difference operator of a derivative on a uniform sampling.

    The row :math:`i` uses the samples :math:`i - n - p + 1, \dots, i`
    (see :func:`compute_backward_finite_difference_coefficients`). At the beginning of the
    sampling, the window is shifted forward (``boundary="one_sided"``) or the row is
    removed (``boundary="valid"``).

    Parameters
    ----------
    order, n_times, n_dim, spacing, accuracy, boundary, sparse, return_indices
        See :func:`assemble_forward_finite_difference_matrix`.

    """
    return _assemble_uniform("backward", order, n_times, n_dim, spacing, accuracy, boundary, sparse, return_indices)


def assemble_central_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    *,
    n_dim: Integral = 1,
    spacing: Real = 1.0,
    accuracy: Integral = 2,
    boundary: str = "one_sided",
    sparse: bool = True,
    return_indices: bool = False,
):
    r"""
    Assemble the central finite difference operator of a derivative on a uniform sampling.

    Shortcut for :func:`assemble_finite_difference_matrix` with the times
    :math:`0, h, \dots, (N_t - 1) h`: centered stencils inside the sampling and one-sided
    stencils of the same accuracy at its ends (or no row with ``boundary="valid"``).

    Parameters
    ----------
    order : Integral
        The order :math:`n \geq 1` of the derivative.

    n_times : Integral
        The number of samples :math:`N_t`.

    accuracy : Integral, optional (default: ``2``)
        The order :math:`p` of the error, a positive even integer.

    n_dim, spacing, boundary, sparse, return_indices
        See :func:`assemble_forward_finite_difference_matrix`.

    Examples
    --------
    .. code-block:: python

        assemble_central_finite_difference_matrix(2, 5, sparse=False)
        # [[ 2, -5,  4, -1,  0],   (one-sided)
        #  [ 1, -2,  1,  0,  0],
        #  [ 0,  1, -2,  1,  0],
        #  [ 0,  0,  1, -2,  1],
        #  [ 0, -1,  4, -5,  2]]   (one-sided)

    """
    return _assemble_uniform("central", order, n_times, n_dim, spacing, accuracy, boundary, sparse, return_indices)


# =============================================================================
# Application to arrays
# =============================================================================
def _apply(
    scheme: str,
    data: ArrayLike,
    order: Integral,
    axis: Integral,
    spacing: Real,
    accuracy: Integral,
    mode: str,
    value: Real,
) -> numpy.ndarray:
    data = numpy.asarray(data, dtype=numpy.float64)
    axis = _validate_integer(axis, "axis")
    if not -data.ndim <= axis < data.ndim:
        raise ValueError(f"Input 'axis' must be in [{-data.ndim}, {data.ndim - 1}]. Got {axis}.")
    mode = _validate_choice(mode, "mode", _APPLY_MODES)
    value = _validate_real(value, "value")
    order, accuracy = _validate_order_accuracy(order, accuracy, even_accuracy=(scheme == "central"))
    spacing = _validate_real(spacing, "spacing", positive=True, finite=True)

    if mode == "one_sided":
        matrix = _assemble(numpy.arange(data.shape[axis]) * spacing, order, accuracy, scheme, "one_sided", None, 1, True)[0]
        moved = numpy.moveaxis(data, axis, 0)
        result = (matrix @ moved.reshape((moved.shape[0], -1))).reshape(moved.shape)
        return numpy.moveaxis(result, 0, axis)

    if scheme == "forward":
        coefficients = compute_forward_finite_difference_coefficients(order, spacing, accuracy)
        origin = -(coefficients.size // 2)  # The first coefficient is at the current point
    elif scheme == "backward":
        coefficients = compute_backward_finite_difference_coefficients(order, spacing, accuracy)[::-1]
        origin = (coefficients.size - 1) // 2  # The last coefficient is at the current point
    else:
        coefficients = compute_central_finite_difference_coefficients(order, spacing, accuracy)
        origin = 0
    return scipy.ndimage.correlate1d(data, coefficients, axis=axis, mode=mode, cval=value, origin=origin)


_APPLY_DOC = r"""
    The available modes for the ends of the sampling are:

    +-----------------+--------------------------------------------------------------------+
    | Mode            | Description                                                        |
    +=================+====================================================================+
    | ``"one_sided"`` | Stencils shifted inside the sampling, of the same accuracy: the    |
    |                 | result is a derivative everywhere (default).                       |
    +-----------------+--------------------------------------------------------------------+
    | ``"reflect"``   | :math:`(d c b a | a b c d | d c b a)`                              |
    +-----------------+--------------------------------------------------------------------+
    | ``"constant"``  | :math:`(k k k k | a b c d | k k k k)`                              |
    +-----------------+--------------------------------------------------------------------+
    | ``"nearest"``   | :math:`(a a a a | a b c d | d d d d)`                              |
    +-----------------+--------------------------------------------------------------------+
    | ``"mirror"``    | :math:`(d c b | a b c d | c b a)`                                  |
    +-----------------+--------------------------------------------------------------------+
    | ``"wrap"``      | :math:`(a b c d | a b c d | a b c d)`                              |
    +-----------------+--------------------------------------------------------------------+

    With the modes of :func:`scipy.ndimage.correlate1d`, the values near the ends use the
    extended signal: they are not approximations of the derivative (except for periodic
    signals with ``"wrap"``).
"""


def apply_forward_finite_difference(
    data: ArrayLike,
    order: Integral,
    *,
    axis: Integral = -1,
    spacing: Real = 1.0,
    accuracy: Integral = 1,
    mode: str = "one_sided",
    value: Real = 0.0,
) -> numpy.ndarray:
    return _apply("forward", data, order, axis, spacing, accuracy, mode, value)


def apply_backward_finite_difference(
    data: ArrayLike,
    order: Integral,
    *,
    axis: Integral = -1,
    spacing: Real = 1.0,
    accuracy: Integral = 1,
    mode: str = "one_sided",
    value: Real = 0.0,
) -> numpy.ndarray:
    return _apply("backward", data, order, axis, spacing, accuracy, mode, value)


def apply_central_finite_difference(
    data: ArrayLike,
    order: Integral,
    *,
    axis: Integral = -1,
    spacing: Real = 1.0,
    accuracy: Integral = 2,
    mode: str = "one_sided",
    value: Real = 0.0,
) -> numpy.ndarray:
    return _apply("central", data, order, axis, spacing, accuracy, mode, value)


for _function, _scheme in (
    (apply_forward_finite_difference, "forward"),
    (apply_backward_finite_difference, "backward"),
    (apply_central_finite_difference, "central"),
):
    _function.__doc__ = (
        f"""
    Apply the {_scheme} finite difference approximation of a derivative to an array
    sampled uniformly along an axis.
"""
        + _APPLY_DOC
        + f"""
    Parameters
    ----------
    data : ArrayLike
        The sampled signal (converted to ``numpy.float64``).

    order : Integral
        The order :math:`n \\geq 1` of the derivative.

    axis : Integral, optional (default: ``-1``)
        The axis of the sampling.

    spacing : Real, optional (default: ``1.0``)
        The sampling step :math:`h > 0`.

    accuracy : Integral, optional
        The order :math:`p` of the error (see
        :func:`compute_{_scheme}_finite_difference_coefficients`).

    mode : str, optional (default: ``"one_sided"``)
        The treatment of the ends of the sampling (see above).

    value : Real, optional (default: ``0.0``)
        The padding value of the mode ``"constant"``.

    Returns
    -------
    numpy.ndarray
        The derivative, with the shape of ``data``.
"""
    )
