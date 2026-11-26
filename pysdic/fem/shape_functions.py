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
from typing import Callable, Dict, Optional, Tuple, Union
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
from .._validation import _validate_bool, _validate_real, _as_array
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import _validate_element_type


# =============================================================================
# Internal helpers: input validation
# =============================================================================
def _validate_natural_coordinates(
    natural_coordinates: ArrayLike, topological_dimension: int
) -> numpy.ndarray:
    r"""
    Convert ``natural_coordinates`` to a ``numpy.float64`` array of shape
    :math:`(N_{p}, K)` (the shape :math:`(N_{p},)` is accepted for :math:`K = 1`).

    .. important::

        Internal use only. ``topological_dimension`` is assumed to be valid.

    """
    # NaN accepted: the points with NaN coordinates are outside the reference element
    return _as_array(
        natural_coordinates,
        "natural_coordinates",
        numpy.float64,
        shape=(None, topological_dimension),
        shape_hint=f"(N_p, {topological_dimension})",
        accept_1d_as_column=(topological_dimension == 1),
        finite=False,
    )


# =============================================================================
# Internal helpers: reference domains
# =============================================================================
def _inside_hypercube(natural_coordinates: numpy.ndarray, tolerance: float) -> numpy.ndarray:
    r"""
    Return a boolean mask of the points lying inside the reference hypercube
    :math:`[-1, 1]^K` (segments for :math:`K=1`, quadrangles for :math:`K=2`).

    .. important::

        Internal use only.

    A point is considered inside if all its natural coordinates satisfy
    :math:`-1 - \text{tol} \leq \xi_k \leq 1 + \text{tol}`.
    Points with NaN coordinates are considered outside.

    Parameters
    ----------
    natural_coordinates: numpy.ndarray
        Validated natural coordinates with shape :math:`(N_{p}, K)`.

    tolerance: float
        Non-negative tolerance enlarging the reference domain.


    Returns
    -------
    numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)`, True for points inside the domain.

    """
    return numpy.all(numpy.abs(natural_coordinates) <= 1.0 + tolerance, axis=1)


def _inside_simplex(natural_coordinates: numpy.ndarray, tolerance: float) -> numpy.ndarray:
    r"""
    Return a boolean mask of the points lying inside the reference simplex
    :math:`\{\xi_k \geq 0, \sum_k \xi_k \leq 1\}` (triangles for :math:`K=2`).

    .. important::

        Internal use only.

    A point is considered inside if :math:`\xi_k \geq -\text{tol}` for all :math:`k`
    and :math:`\sum_k \xi_k \leq 1 + \text{tol}`.
    Points with NaN coordinates are considered outside.

    Parameters
    ----------
    natural_coordinates: numpy.ndarray
        Validated natural coordinates with shape :math:`(N_{p}, K)`.

    tolerance: float
        Non-negative tolerance enlarging the reference domain.


    Returns
    -------
    numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)`, True for points inside the domain.

    """
    return numpy.all(natural_coordinates >= -tolerance, axis=1) & (
        numpy.sum(natural_coordinates, axis=1) <= 1.0 + tolerance
    )


# =============================================================================
# Internal helpers: element kernels
# =============================================================================
# Each kernel evaluates the shape functions (and optionally their derivatives) at
# ALL the given points, without any masking. The masking of the points outside the
# reference domain is handled by ``_evaluate_shape_functions``.
#
# Kernel signature:
#     kernel(natural_coordinates, return_derivatives) -> (N, dN)
#     - natural_coordinates: validated array with shape (N_p, K)
#     - N: shape functions with shape (N_p, N_vpe)
#     - dN: derivatives with shape (N_p, N_vpe, K) if return_derivatives else None
#           with the convention dN[p, j, k] = dN_j / dxi_k evaluated at point p.
#
# The node ordering and the formulas are documented in the docstrings of the
# corresponding public functions ``compute_<element_type>_shape_functions``.


def _constant_derivatives(values: ArrayLike, n_points: int) -> numpy.ndarray:
    r"""
    Broadcast constant derivatives with shape :math:`(N_{vpe}, K)` to an owned
    (writable) array of shape :math:`(N_{p}, N_{vpe}, K)`.

    .. important::

        Internal use only.

    """
    values = numpy.asarray(values, dtype=numpy.float64)
    return numpy.broadcast_to(values, (n_points,) + values.shape).copy()


def _segment_2_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 2-node segment (see :func:`compute_segment_2_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]

    shape_functions = numpy.stack(
        [
            0.5 * (1.0 - xi),  # N_1
            0.5 * (1.0 + xi),  # N_2
        ],
        axis=1,
    )  # (N_p, 2)

    derivatives = None
    if return_derivatives:
        derivatives = _constant_derivatives(
            [
                [-0.5],  # dN_1/dxi
                [0.5],  # dN_2/dxi
            ],
            xi.shape[0],
        )  # (N_p, 2, 1)

    return shape_functions, derivatives


def _segment_3_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 3-node segment (see :func:`compute_segment_3_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]

    shape_functions = numpy.stack(
        [
            0.5 * xi * (xi - 1.0),  # N_1
            0.5 * xi * (xi + 1.0),  # N_2
            1.0 - xi**2,  # N_3
        ],
        axis=1,
    )  # (N_p, 3)

    derivatives = None
    if return_derivatives:
        derivatives = numpy.stack(
            [
                xi - 0.5,  # dN_1/dxi
                xi + 0.5,  # dN_2/dxi
                -2.0 * xi,  # dN_3/dxi
            ],
            axis=1,
        )[:, :, numpy.newaxis]  # (N_p, 3, 1)

    return shape_functions, derivatives


def _triangle_3_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 3-node triangle (see :func:`compute_triangle_3_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]
    eta = natural_coordinates[:, 1]

    shape_functions = numpy.stack(
        [
            1.0 - xi - eta,  # N_1
            xi,  # N_2
            eta,  # N_3
        ],
        axis=1,
    )  # (N_p, 3)

    derivatives = None
    if return_derivatives:
        derivatives = _constant_derivatives(
            [
                [-1.0, -1.0],  # (dN_1/dxi, dN_1/deta)
                [1.0, 0.0],  # (dN_2/dxi, dN_2/deta)
                [0.0, 1.0],  # (dN_3/dxi, dN_3/deta)
            ],
            xi.shape[0],
        )  # (N_p, 3, 2)

    return shape_functions, derivatives


def _triangle_6_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 6-node triangle (see :func:`compute_triangle_6_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]
    eta = natural_coordinates[:, 1]
    zeta = 1.0 - xi - eta  # Third barycentric coordinate

    shape_functions = numpy.stack(
        [
            zeta * (1.0 - 2.0 * xi - 2.0 * eta),  # N_1
            xi * (2.0 * xi - 1.0),  # N_2
            eta * (2.0 * eta - 1.0),  # N_3
            4.0 * xi * zeta,  # N_4
            4.0 * xi * eta,  # N_5
            4.0 * eta * zeta,  # N_6
        ],
        axis=1,
    )  # (N_p, 6)

    derivatives = None
    if return_derivatives:
        zeros = numpy.zeros_like(xi)
        d_xi = numpy.stack(
            [
                -3.0 + 4.0 * xi + 4.0 * eta,  # dN_1/dxi
                4.0 * xi - 1.0,  # dN_2/dxi
                zeros,  # dN_3/dxi
                4.0 - 8.0 * xi - 4.0 * eta,  # dN_4/dxi
                4.0 * eta,  # dN_5/dxi
                -4.0 * eta,  # dN_6/dxi
            ],
            axis=1,
        )  # (N_p, 6)
        d_eta = numpy.stack(
            [
                -3.0 + 4.0 * xi + 4.0 * eta,  # dN_1/deta
                zeros,  # dN_2/deta
                4.0 * eta - 1.0,  # dN_3/deta
                -4.0 * xi,  # dN_4/deta
                4.0 * xi,  # dN_5/deta
                4.0 - 4.0 * xi - 8.0 * eta,  # dN_6/deta
            ],
            axis=1,
        )  # (N_p, 6)
        derivatives = numpy.stack([d_xi, d_eta], axis=2)  # (N_p, 6, 2)

    return shape_functions, derivatives


def _quadrangle_4_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 4-node quadrangle (see :func:`compute_quadrangle_4_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]
    eta = natural_coordinates[:, 1]

    shape_functions = numpy.stack(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),  # N_1
            0.25 * (1.0 + xi) * (1.0 - eta),  # N_2
            0.25 * (1.0 + xi) * (1.0 + eta),  # N_3
            0.25 * (1.0 - xi) * (1.0 + eta),  # N_4
        ],
        axis=1,
    )  # (N_p, 4)

    derivatives = None
    if return_derivatives:
        d_xi = numpy.stack(
            [
                0.25 * (eta - 1.0),  # dN_1/dxi
                0.25 * (1.0 - eta),  # dN_2/dxi
                0.25 * (1.0 + eta),  # dN_3/dxi
                -0.25 * (1.0 + eta),  # dN_4/dxi
            ],
            axis=1,
        )  # (N_p, 4)
        d_eta = numpy.stack(
            [
                0.25 * (xi - 1.0),  # dN_1/deta
                -0.25 * (1.0 + xi),  # dN_2/deta
                0.25 * (1.0 + xi),  # dN_3/deta
                0.25 * (1.0 - xi),  # dN_4/deta
            ],
            axis=1,
        )  # (N_p, 4)
        derivatives = numpy.stack([d_xi, d_eta], axis=2)  # (N_p, 4, 2)

    return shape_functions, derivatives


def _quadrangle_8_kernel(
    natural_coordinates: numpy.ndarray, return_derivatives: bool
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
    r"""
    Kernel of the 8-node serendipity quadrangle
    (see :func:`compute_quadrangle_8_shape_functions`).

    .. important::

        Internal use only. No masking is applied.

    """
    xi = natural_coordinates[:, 0]
    eta = natural_coordinates[:, 1]

    shape_functions = numpy.stack(
        [
            0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),  # N_1
            0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),  # N_2
            0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),  # N_3
            0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),  # N_4
            0.5 * (1.0 - xi**2) * (1.0 - eta),  # N_5
            0.5 * (1.0 + xi) * (1.0 - eta**2),  # N_6
            0.5 * (1.0 - xi**2) * (1.0 + eta),  # N_7
            0.5 * (1.0 - xi) * (1.0 - eta**2),  # N_8
        ],
        axis=1,
    )  # (N_p, 8)

    derivatives = None
    if return_derivatives:
        d_xi = numpy.stack(
            [
                0.25 * (1.0 - eta) * (2.0 * xi + eta),  # dN_1/dxi
                0.25 * (1.0 - eta) * (2.0 * xi - eta),  # dN_2/dxi
                0.25 * (1.0 + eta) * (2.0 * xi + eta),  # dN_3/dxi
                0.25 * (1.0 + eta) * (2.0 * xi - eta),  # dN_4/dxi
                -xi * (1.0 - eta),  # dN_5/dxi
                0.5 * (1.0 - eta**2),  # dN_6/dxi
                -xi * (1.0 + eta),  # dN_7/dxi
                -0.5 * (1.0 - eta**2),  # dN_8/dxi
            ],
            axis=1,
        )  # (N_p, 8)
        d_eta = numpy.stack(
            [
                0.25 * (1.0 - xi) * (xi + 2.0 * eta),  # dN_1/deta
                -0.25 * (1.0 + xi) * (xi - 2.0 * eta),  # dN_2/deta
                0.25 * (1.0 + xi) * (xi + 2.0 * eta),  # dN_3/deta
                0.25 * (1.0 - xi) * (-xi + 2.0 * eta),  # dN_4/deta
                -0.5 * (1.0 - xi**2),  # dN_5/deta
                -eta * (1.0 + xi),  # dN_6/deta
                0.5 * (1.0 - xi**2),  # dN_7/deta
                -eta * (1.0 - xi),  # dN_8/deta
            ],
            axis=1,
        )  # (N_p, 8)
        derivatives = numpy.stack([d_xi, d_eta], axis=2)  # (N_p, 8, 2)

    return shape_functions, derivatives


# =============================================================================
# Internal helpers: registry and generic evaluator
# =============================================================================
_KernelType = Callable[
    [numpy.ndarray, bool], Tuple[numpy.ndarray, Optional[numpy.ndarray]]
]
_InsideType = Callable[[numpy.ndarray, float], numpy.ndarray]

# element_type -> (kernel, reference domain membership)
# The topological dimension K and the number of nodes N_vpe are read from
# _MAPPING_ELEMENT_TYPE_TO_PROPERTIES (single source of truth).
_SHAPE_FUNCTIONS_REGISTRY: Dict[str, Tuple[_KernelType, _InsideType]] = {
    "segment_2": (_segment_2_kernel, _inside_hypercube),
    "segment_3": (_segment_3_kernel, _inside_hypercube),
    "triangle_3": (_triangle_3_kernel, _inside_simplex),
    "triangle_6": (_triangle_6_kernel, _inside_simplex),
    "quadrangle_4": (_quadrangle_4_kernel, _inside_hypercube),
    "quadrangle_8": (_quadrangle_8_kernel, _inside_hypercube),
}

# Consistency checks at import time (not stripped by ``python -O`` unlike an assert)
if _SHAPE_FUNCTIONS_REGISTRY.keys() != _MAPPING_ELEMENT_TYPE_TO_PROPERTIES.keys():
    raise RuntimeError(
        "[INTERNAL] The keys of _SHAPE_FUNCTIONS_REGISTRY must match the supported "
        "element types defined in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES. "
        f"Got {sorted(_SHAPE_FUNCTIONS_REGISTRY)} and "
        f"{sorted(_MAPPING_ELEMENT_TYPE_TO_PROPERTIES)}."
    )
for _element_type, (_kernel, _) in _SHAPE_FUNCTIONS_REGISTRY.items():
    _properties = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[_element_type]
    _K, _N_vpe = _properties["expected_K"], _properties["expected_N_vpe"]
    _N, _dN = _kernel(numpy.zeros((1, _K), dtype=numpy.float64), True)
    if _N.shape != (1, _N_vpe) or _dN.shape != (1, _N_vpe, _K):
        raise RuntimeError(
            f"[INTERNAL] Inconsistent shape functions kernel for '{_element_type}': "
            f"got N shape {_N.shape} and dN shape {_dN.shape}, "
            f"expected (1, {_N_vpe}) and (1, {_N_vpe}, {_K})."
        )
del _element_type, _kernel, _properties, _K, _N_vpe, _N, _dN


def _evaluate_shape_functions(
    natural_coordinates: ArrayLike,
    element_type: str,
    return_derivatives: bool,
    default: Optional[Real],
    tolerance: Real,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Generic evaluation of the shape functions of a registered element type.

    .. important::

        Internal use only. ``element_type`` is assumed to be a valid key of
        ``_SHAPE_FUNCTIONS_REGISTRY``.

    The evaluation is performed in three steps:

    1. Validation of the inputs.
    2. Evaluation of the element kernel at all the points (vectorized).
    3. If ``default`` is not None, the shape functions (and derivatives) of the
       points outside the reference domain (enlarged by ``tolerance``) are
       overwritten by ``default``.

    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates with shape :math:`(N_{p}, K)` (or :math:`(N_{p},)` for
        :math:`K=1`).

    element_type: str
        Registered element type.

    return_derivatives: bool
        If True, the derivatives with respect to the natural coordinates are returned.

    default: Optional[Real]
        Value assigned to the points outside the reference domain, or None to
        disable the masking.

    tolerance: Real
        Non-negative tolerance enlarging the reference domain.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions with shape :math:`(N_{p}, N_{vpe})`.

    shape_function_derivatives: numpy.ndarray, optional
        Derivatives with shape :math:`(N_{p}, N_{vpe}, K)`, only if
        ``return_derivatives`` is True.

    """
    kernel, _ = _SHAPE_FUNCTIONS_REGISTRY[element_type]
    topological_dimension = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type][
        "expected_K"
    ]

    # Input validation and conversion
    return_derivatives = _validate_bool(return_derivatives, "return_derivatives")
    default = _validate_real(default, "default", allow_none=True)
    tolerance = _validate_real(tolerance, "tolerance", non_negative=True, finite=True)
    natural_coordinates = _validate_natural_coordinates(
        natural_coordinates, topological_dimension
    )

    if default is None:
        # No masking: the polynomials are evaluated (extrapolated) everywhere
        shape_functions, derivatives = kernel(natural_coordinates, return_derivatives)
    else:
        shape_functions, derivatives, inside_mask = _shape_functions_impl(
            natural_coordinates, element_type, return_derivatives, tolerance
        )
        shape_functions[~inside_mask] = default
        if derivatives is not None:
            derivatives[~inside_mask] = default

    if return_derivatives:
        return shape_functions, derivatives
    return shape_functions


def _shape_functions_impl(
    natural_coordinates: numpy.ndarray,
    element_type: str,
    return_derivatives: bool,
    tolerance: float,
) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], numpy.ndarray]:
    r"""
    Evaluate the shape functions of a registered element type without any validation
    nor masking, and return the mask of the points inside the reference element.

    .. important::

        Internal use only. All the inputs are assumed to be validated:
        ``natural_coordinates`` is a ``numpy.float64`` array of shape
        :math:`(N_{p}, K)`, ``element_type`` a registered element type and
        ``tolerance`` a finite non-negative float.

    Used by the public functions of :mod:`pysdic.fem` to avoid validating the same
    inputs several times.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions with shape :math:`(N_{p}, N_{vpe})` (extrapolated outside the
        reference element).

    shape_function_derivatives: Optional[numpy.ndarray]
        Derivatives with shape :math:`(N_{p}, N_{vpe}, K)` if ``return_derivatives``
        is True, None otherwise.

    inside_mask: numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)`, True for the points inside the
        reference element enlarged by ``tolerance`` (False for NaN coordinates).

    """
    kernel, inside = _SHAPE_FUNCTIONS_REGISTRY[element_type]
    # Points outside the domain (inf, overflow, ...) may raise floating point warnings,
    # their values are masked by the callers.
    with numpy.errstate(all="ignore"):
        shape_functions, derivatives = kernel(natural_coordinates, return_derivatives)
    return shape_functions, derivatives, inside(natural_coordinates, tolerance)


# =============================================================================
# Public functions
# =============================================================================
def compute_segment_2_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 2-node segment for given
    ``natural_coordinates`` :math:`\xi`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`, where :math:`N_{p}` is the
        number of points to evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not in :math:`[-1, 1]`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`[-1 - \text{tol}, 1 + \text{tol}]`. Useful for points located on the
        boundary of the element up to rounding errors. Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 2)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 2, 1)` such that ``shape_function_derivatives[p, j, 0]``
        is :math:`\frac{dN_j}{d\xi}` at point :math:`p`.


    Notes
    -----
    A 2-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | Node No. | :math:`(\xi)`   | Shape Function :math:`N`                | First Derivative :math:`(\frac{dN}{d\xi})`          |
    +==========+=================+=========================================+=====================================================+
    | 1        | :math:`-1`      | :math:`N_1(\xi) = \frac{1}{2}(1 - \xi)` | :math:`-\frac{1}{2}`                                |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | 2        | :math:`1`       | :math:`N_2(\xi) = \frac{1}{2}(1 + \xi)` | :math:`\frac{1}{2}`                                 |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_2.png
        :alt: 2-node segment element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_segment_3_shape_functions:
        Shape functions for 3-node segment 1D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_segment_2_gauss_points:
        Gauss integration points for 2-node segment 1D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_segment_2_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = compute_segment_2_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.  0. ]
         [0.5 0.5]
         [0.  1. ]
         [0.  0. ]]

    Extrapolate the shape functions outside the element with ``default=None``:

    .. code-block:: python
        :linenos:

        shape_functions = compute_segment_2_shape_functions([1.5], default=None)
        print(shape_functions)

    .. code-block:: console

        [[-0.25  1.25]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "segment_2", return_derivatives, default, tolerance
    )


def compute_segment_3_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node segment for given
    ``natural_coordinates`` :math:`\xi`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`, where :math:`N_{p}` is the number
        of points to evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not in :math:`[-1, 1]`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`[-1 - \text{tol}, 1 + \text{tol}]`. Useful for points located on the
        boundary of the element up to rounding errors. Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 3)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 3, 1)` such that ``shape_function_derivatives[p, j, 0]``
        is :math:`\frac{dN_j}{d\xi}` at point :math:`p`.


    Notes
    -----
    A 3-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | Node No. | :math:`(\xi)`   | Shape Function :math:`N`                      | First Derivative :math:`(\frac{dN}{d\xi})`              |
    +==========+=================+===============================================+=========================================================+
    | 1        | :math:`-1`      | :math:`N_1(\xi) = \frac{1}{2}\xi(\xi - 1)`    | :math:`\xi - \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 2        | :math:`1`       | :math:`N_2(\xi) = \frac{1}{2}\xi(\xi + 1)`    | :math:`\xi + \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 3        | :math:`0`       | :math:`N_3(\xi) = 1 - \xi^2`                  | :math:`-2\xi`                                           |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_3.png
        :alt: 3-node segment element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_segment_2_shape_functions:
        Shape functions for 2-node segment 1D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_segment_3_gauss_points:
        Gauss integration points for 3-node segment 1D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_segment_3_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = compute_segment_3_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. -0.  0.]
         [-0.  0.  1.]
         [ 0.  1.  0.]
         [ 0.  0.  0.]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "segment_3", return_derivatives, default, tolerance
    )


def compute_triangle_3_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node triangle for given
    ``natural_coordinates`` :math:`(\xi, \eta)`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p}, 2)`, where :math:`N_{p}` is the number of points to
        evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not
        :math:`\xi, \eta \geq 0` with :math:`\xi + \eta \leq 1`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`\xi, \eta \geq -\text{tol}` with :math:`\xi + \eta \leq 1 + \text{tol}`.
        Useful for points located on the boundary of the element up to rounding
        errors. Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 3)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 3, 2)` such that ``shape_function_derivatives[p, j, k]``
        is :math:`\frac{\partial N_j}{\partial \xi_k}` at point :math:`p`
        (with :math:`\xi_0 = \xi` and :math:`\xi_1 = \eta`).


    Notes
    -----
    A 3-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                      | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})` |
    +==========+=======================+===============================================+==============================================================+
    | 1        | :math:`(0, 0)`        | :math:`N_1(\xi, \eta) = 1 - \xi - \eta`       | :math:`(-1, -1)`                                             |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 2        | :math:`(1, 0)`        | :math:`N_2(\xi, \eta) = \xi`                  | :math:`(1, 0)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 3        | :math:`(0, 1)`        | :math:`N_3(\xi, \eta) = \eta`                 | :math:`(0, 1)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_3.png
        :alt: 3-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_triangle_6_shape_functions:
        Shape functions for 6-node triangle 2D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_triangle_3_gauss_points:
        Gauss integration points for 3-node triangle 2D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_triangle_3_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = compute_triangle_3_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.  0.  0. ]
         [0.5 0.5 0. ]
         [0.5 0.  0.5]
         [0.  0.  0. ]]

    A point on the edge :math:`\xi + \eta = 1` up to rounding errors is considered
    outside without tolerance:

    .. code-block:: python
        :linenos:

        coords = numpy.array([[0.7, 0.3 + 1e-15]])  # xi + eta = 1.000000000000001
        print(compute_triangle_3_shape_functions(coords))
        print(compute_triangle_3_shape_functions(coords, tolerance=1e-12))

    .. code-block:: console

        [[0. 0. 0.]]
        [[-9.43689571e-16  7.00000000e-01  3.00000000e-01]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "triangle_3", return_derivatives, default, tolerance
    )


def compute_triangle_6_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 6-node triangle for given
    ``natural_coordinates`` :math:`(\xi, \eta)`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p}, 2)`, where :math:`N_{p}` is the number of points to
        evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not
        :math:`\xi, \eta \geq 0` with :math:`\xi + \eta \leq 1`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`\xi, \eta \geq -\text{tol}` with :math:`\xi + \eta \leq 1 + \text{tol}`.
        Useful for points located on the boundary of the element up to rounding
        errors. Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 6)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 6, 2)` such that ``shape_function_derivatives[p, j, k]``
        is :math:`\frac{\partial N_j}{\partial \xi_k}` at point :math:`p`
        (with :math:`\xi_0 = \xi` and :math:`\xi_1 = \eta`).


    Notes
    -----
    A 6-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                                             | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`       |
    +==========+=======================+======================================================================================+====================================================================+
    | 1        | :math:`(0, 0)`        | :math:`N_1(\xi, \eta) = (1 - \xi - \eta)(1 - 2\xi - 2\eta)`                          | :math:`(-3 + 4\xi + 4\eta, -3 + 4\xi + 4\eta)`                     |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | :math:`(1, 0)`        | :math:`N_2(\xi, \eta) = \xi(2\xi - 1)`                                               | :math:`(4\xi - 1, 0)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | :math:`(0, 1)`        | :math:`N_3(\xi, \eta) = \eta(2\eta - 1)`                                             | :math:`(0, 4\eta - 1)`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | :math:`(0.5, 0)`      | :math:`N_4(\xi, \eta) = 4\xi(1 - \xi - \eta)`                                        | :math:`(4 - 8\xi - 4\eta, -4\xi)`                                  |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 5        | :math:`(0.5, 0.5)`    | :math:`N_5(\xi, \eta) = 4\xi\eta`                                                    | :math:`(4\eta, 4\xi)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 6        | :math:`(0, 0.5)`      | :math:`N_6(\xi, \eta) = 4\eta(1 - \xi - \eta)`                                       | :math:`(-4\eta, 4 - 4\xi - 8\eta)`                                 |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_6.png
        :alt: 6-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_triangle_3_shape_functions:
        Shape functions for 3-node triangle 2D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_triangle_6_gauss_points:
        Gauss integration points for 6-node triangle 2D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_triangle_6_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = compute_triangle_6_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. -0. -0.  0.  0.  0.]
         [ 0.  0. -0.  1.  0.  0.]
         [ 0. -0.  0.  0.  0.  1.]
         [ 0.  0.  0.  0.  0.  0.]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "triangle_6", return_derivatives, default, tolerance
    )


def compute_quadrangle_4_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 4-node quadrangle for given
    ``natural_coordinates`` :math:`(\xi, \eta)`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p}, 2)`, where :math:`N_{p}` is the number of points to
        evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not in
        :math:`\xi, \eta \in [-1, 1]`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`\xi, \eta \in [-1 - \text{tol}, 1 + \text{tol}]`. Useful for points
        located on the boundary of the element up to rounding errors.
        Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 4)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 4, 2)` such that ``shape_function_derivatives[p, j, k]``
        is :math:`\frac{\partial N_j}{\partial \xi_k}` at point :math:`p`
        (with :math:`\xi_0 = \xi` and :math:`\xi_1 = \eta`).


    Notes
    -----
    A 4-node quadrangle represented in the figure below has the following shape
    functions:

    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                      | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`       |
    +==========+=======================+===============================================================+====================================================================+
    | 1        | :math:`(-1, -1)`      | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(\eta - 1), \frac{1}{4}(\xi - 1))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | :math:`(1, -1)`       | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(1 - \eta), -\frac{1}{4}(1 + \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | :math:`(1, 1)`        | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)`       | :math:`(\frac{1}{4}(1 + \eta), \frac{1}{4}(1 + \xi))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | :math:`(-1, 1)`       | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)`       | :math:`(-\frac{1}{4}(1 + \eta), \frac{1}{4}(1 - \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_4.png
        :alt: 4-node quadrangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_quadrangle_8_shape_functions:
        Shape functions for 8-node quadrangle 2D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_quadrangle_4_gauss_points:
        Gauss integration points for 4-node quadrangle 2D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_quadrangle_4_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        shape_functions = compute_quadrangle_4_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.  0.  0.  0. ]
         [0.5 0.5 0.  0. ]
         [0.  0.5 0.5 0. ]
         [0.  0.  0.  0. ]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "quadrangle_4", return_derivatives, default, tolerance
    )


def compute_quadrangle_8_shape_functions(
    natural_coordinates: ArrayLike,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for an 8-node (serendipity) quadrangle for given
    ``natural_coordinates`` :math:`(\xi, \eta)`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.


    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p}, 2)`, where :math:`N_{p}` is the number of points to
        evaluate.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range (i.e., not in
        :math:`\xi, \eta \in [-1, 1]`).
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range to
        :math:`\xi, \eta \in [-1 - \text{tol}, 1 + \text{tol}]`. Useful for points
        located on the boundary of the element up to rounding errors.
        Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, 8)`, where each row corresponds to a point and each
        column to a node.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, 8, 2)` such that ``shape_function_derivatives[p, j, k]``
        is :math:`\frac{\partial N_j}{\partial \xi_k}` at point :math:`p`
        (with :math:`\xi_0 = \xi` and :math:`\xi_1 = \eta`).


    Notes
    -----
    An 8-node serendipity quadrangle represented in the figure below has the
    following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                                                   | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`                                 |
    +==========+=======================+============================================================================================+==============================================================================================+
    | 1        | :math:`(-1, -1)`      | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)(- \xi - \eta - 1)`                  | :math:`(\frac{1}{4}(1 - \eta)(2\xi + \eta), \frac{1}{4}(1 - \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 2        | :math:`(1, -1)`       | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)(\xi - \eta - 1)`                    | :math:`(\frac{1}{4}(1 - \eta)(2\xi - \eta), -\frac{1}{4}(1 + \xi)(\xi - 2\eta))`             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 3        | :math:`(1, 1)`        | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)(\xi + \eta - 1)`                    | :math:`(\frac{1}{4}(1 + \eta)(2\xi + \eta), \frac{1}{4}(1 + \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 4        | :math:`(-1, 1)`       | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)(- \xi + \eta - 1)`                  | :math:`(\frac{1}{4}(1 + \eta)(2\xi - \eta), \frac{1}{4}(1 - \xi)(- \xi + 2\eta))`            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 5        | :math:`(0, -1)`       | :math:`N_5(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 - \eta)`                                  | :math:`(-\xi(1 - \eta), -\frac{1}{2}(1 - \xi^2))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 6        | :math:`(1, 0)`        | :math:`N_6(\xi, \eta) = \frac{1}{2}(1 + \xi)(1 - \eta^2)`                                  | :math:`(\frac{1}{2}(1 - \eta^2), -\eta(1 + \xi))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 7        | :math:`(0, 1)`        | :math:`N_7(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 + \eta)`                                  | :math:`(-\xi(1 + \eta), \frac{1}{2}(1 - \xi^2))`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 8        | :math:`(-1, 0)`       | :math:`N_8(\xi, \eta) = \frac{1}{2}(1 - \xi)(1 - \eta^2)`                                  | :math:`(-\frac{1}{2}(1 - \eta^2), -\eta(1 - \xi))`                                           |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_8.png
        :alt: 8-node quadrangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_quadrangle_4_shape_functions:
        Shape functions for 4-node quadrangle 2D-elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.

    pysdic.fem.get_quadrangle_8_gauss_points:
        Gauss integration points for 8-node quadrangle 2D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_quadrangle_8_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        shape_functions = compute_quadrangle_8_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. -0. -0. -0.  0.  0.  0.  0.]
         [ 0.  0. -0. -0.  1.  0.  0.  0.]
         [-0.  0.  0. -0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.]]

    """
    return _evaluate_shape_functions(
        natural_coordinates, "quadrangle_8", return_derivatives, default, tolerance
    )


def compute_shape_functions(
    natural_coordinates: ArrayLike,
    element_type: str,
    *,
    return_derivatives: bool = False,
    default: Optional[Real] = 0.0,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a given ``element_type`` at specified
    ``natural_coordinates``.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Output arrays will be ``numpy.float64``.

    Parameters
    ----------
    natural_coordinates: ArrayLike
        Natural coordinates where to evaluate the shape functions. The array must have
        shape :math:`(N_{p}, K)`, where :math:`N_{p}` is the number of points to
        evaluate and :math:`K` is the topological dimension of the element.
        For 1D-elements (:math:`K = 1`), the shape :math:`(N_{p},)` is also accepted.

    element_type: str
        Type of the finite element (case insensitive). Supported types are:

        - ``'segment_2'``: 2-node segment element.
        - ``'segment_3'``: 3-node segment element.
        - ``'triangle_3'``: 3-node triangle element.
        - ``'triangle_6'``: 6-node triangle element.
        - ``'quadrangle_4'``: 4-node quadrangle element.
        - ``'quadrangle_8'``: 8-node (serendipity) quadrangle element.

    return_derivatives: bool, optional (default: ``False``)
        If True, the function will also return the first derivatives of the shape
        functions with respect to the natural coordinates.

    default: Optional[Real], optional (default: ``0.0``)
        Default value to assign to shape functions (and derivatives) when the input
        natural coordinates are out of the valid range of the specified element type.
        If None, no masking is applied and the shape functions are extrapolated
        outside the element.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the valid range of the element. Useful for
        points located on the boundary of the element up to rounding errors.
        Ignored if ``default`` is None.


    Returns
    -------
    shape_functions: numpy.ndarray
        Shape functions evaluated at the given natural coordinates. The returned array
        has shape :math:`(N_{p}, N_{vpe})`, where each row corresponds to a point and
        each column to a node, and :math:`N_{vpe}` is the number of vertices per element
        for the specified element type.

    shape_function_derivatives: numpy.ndarray
        Only returned if ``return_derivatives`` is True. First derivatives of the
        shape functions with respect to the natural coordinates, with shape
        :math:`(N_{p}, N_{vpe}, K)` such that ``shape_function_derivatives[p, j, k]``
        is :math:`\frac{\partial N_j}{\partial \xi_k}` at point :math:`p`.


    See Also
    --------
    pysdic.fem.compute_segment_2_shape_functions:
        Shape functions for 2-node segment 1D-elements.

    pysdic.fem.compute_segment_3_shape_functions:
        Shape functions for 3-node segment 1D-elements.

    pysdic.fem.compute_triangle_3_shape_functions:
        Shape functions for 3-node triangle 2D-elements.

    pysdic.fem.compute_triangle_6_shape_functions:
        Shape functions for 6-node triangle 2D-elements.

    pysdic.fem.compute_quadrangle_4_shape_functions:
        Shape functions for 4-node quadrangle 2D-elements.

    pysdic.fem.compute_quadrangle_8_shape_functions:
        Shape functions for 8-node quadrangle 2D-elements.


    Examples
    --------
    Compute shape functions for a 6-node triangle element for 3 valid points and
    1 invalid point:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_funcs = compute_shape_functions(coords, element_type='triangle_6')
        print("Shape function values for triangle_6:")
        print(shape_funcs)

    .. code-block:: console

        Shape function values for triangle_6:
        [[ 1. -0. -0.  0.  0.  0.]
         [ 0.  0. -0.  1.  0.  0.]
         [ 0. -0.  0.  0.  0.  1.]
         [ 0.  0.  0.  0.  0.  0.]]

    """
    element_type = _validate_element_type(element_type)

    return _evaluate_shape_functions(
        natural_coordinates, element_type, return_derivatives, default, tolerance
    )