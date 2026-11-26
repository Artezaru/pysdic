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
from typing import Dict, Tuple, Union

import numpy
from .._validation import _validate_bool
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import _validate_element_type


# =============================================================================
# Internal helpers: quadrature rules
# =============================================================================
# Each rule is stored once at import time as a pair of read-only arrays:
#     - gauss_points: natural coordinates with shape (N_gp, K)
#     - weights: weights with shape (N_gp,)
#
# The public functions return copies, so the user can modify the returned arrays
# without corrupting the stored rules.
#
# The point ordering and the values are documented in the docstrings of the
# corresponding public functions ``get_<element_type>_gauss_points``.


def _make_rule(
    gauss_points: list, weights: list
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Build a read-only quadrature rule from python lists.

    .. important::

        Internal use only.


    Parameters
    ----------
    gauss_points: list
        Natural coordinates of the Gauss points with shape :math:`(N_{gp}, K)`.

    weights: list
        Weights of the Gauss points with shape :math:`(N_{gp},)`.


    Returns
    -------
    gauss_points: numpy.ndarray
        Read-only array of natural coordinates with shape :math:`(N_{gp}, K)`.

    weights: numpy.ndarray
        Read-only array of weights with shape :math:`(N_{gp},)`.

    """
    gauss_points = numpy.array(gauss_points, dtype=numpy.float64)
    weights = numpy.array(weights, dtype=numpy.float64)
    gauss_points.setflags(write=False)
    weights.setflags(write=False)
    return gauss_points, weights


_A = 1.0 / numpy.sqrt(3.0)  # 2-point Gauss-Legendre abscissa
_B = numpy.sqrt(3.0 / 5.0)  # 3-point Gauss-Legendre abscissa

# element_type -> (gauss_points, weights)
_GAUSS_POINTS_REGISTRY: Dict[str, Tuple[numpy.ndarray, numpy.ndarray]] = {
    # 1-point Gauss-Legendre rule (exact up to degree 1)
    "segment_2": _make_rule(
        [[0.0]],
        [2.0],
    ),
    # 2-point Gauss-Legendre rule (exact up to degree 3)
    "segment_3": _make_rule(
        [[-_A], [_A]],
        [1.0, 1.0],
    ),
    # 1-point centroid rule (exact up to degree 1)
    "triangle_3": _make_rule(
        [[1.0 / 3.0, 1.0 / 3.0]],
        [0.5],
    ),
    # 3-point interior rule (exact up to degree 2)
    "triangle_6": _make_rule(
        [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    ),
    # 2x2 tensor Gauss-Legendre rule (exact up to degree 3 in each direction)
    "quadrangle_4": _make_rule(
        [[-_A, -_A], [_A, -_A], [_A, _A], [-_A, _A]],
        [1.0, 1.0, 1.0, 1.0],
    ),
    # 3x3 tensor Gauss-Legendre rule (exact up to degree 5 in each direction)
    "quadrangle_8": _make_rule(
        [
            [0.0, 0.0],  # center
            [_B, _B],  # corners
            [-_B, _B],
            [-_B, -_B],
            [_B, -_B],
            [0.0, _B],  # mid-edges
            [-_B, 0.0],
            [0.0, -_B],
            [_B, 0.0],
        ],
        [64.0 / 81.0] + [25.0 / 81.0] * 4 + [40.0 / 81.0] * 4,
    ),
}

# Consistency checks at import time (not stripped by ``python -O`` unlike an assert)
if _GAUSS_POINTS_REGISTRY.keys() != _MAPPING_ELEMENT_TYPE_TO_PROPERTIES.keys():
    raise RuntimeError(
        "[INTERNAL] The keys of _GAUSS_POINTS_REGISTRY must match the supported "
        "element types defined in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES. "
        f"Got {sorted(_GAUSS_POINTS_REGISTRY)} and "
        f"{sorted(_MAPPING_ELEMENT_TYPE_TO_PROPERTIES)}."
    )
for _element_type, (_points, _weights) in _GAUSS_POINTS_REGISTRY.items():
    _K = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[_element_type]["expected_K"]
    if _points.ndim != 2 or _points.shape[1] != _K or _weights.shape != (_points.shape[0],):
        raise RuntimeError(
            f"[INTERNAL] Inconsistent Gauss rule for '{_element_type}': "
            f"points shape {_points.shape}, weights shape {_weights.shape}, "
            f"expected K = {_K}."
        )
del _element_type, _points, _weights, _K


def _get_gauss_points(
    element_type: str, return_weights: bool
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Generic access to the Gauss quadrature rule of a registered element type.

    .. important::

        Internal use only. ``element_type`` is assumed to be a valid key of
        ``_GAUSS_POINTS_REGISTRY``.


    Parameters
    ----------
    element_type: str
        Registered element type.

    return_weights: bool
        If True, the weights are also returned. ``numpy.bool_`` values are accepted.


    Returns
    -------
    gauss_points: numpy.ndarray
        Copy of the natural coordinates of the Gauss points with shape
        :math:`(N_{gp}, K)`.

    weights: numpy.ndarray, optional
        Copy of the weights with shape :math:`(N_{gp},)`, only if ``return_weights``
        is True.

    """
    return_weights = _validate_bool(return_weights, "return_weights")

    gauss_points, weights = _GAUSS_POINTS_REGISTRY[element_type]

    if return_weights:
        return gauss_points.copy(), weights.copy()
    return gauss_points.copy()


# =============================================================================
# Public functions
# =============================================================================
def get_segment_2_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`\xi` and weights of Gauss quadrature points for a
    2-node segment element.

    The 1-point Gauss-Legendre rule is used. It integrates exactly polynomials up to
    degree 1 over the reference segment :math:`[-1, 1]`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(1, 1)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(1,)`. They sum to :math:`2`, the length of
        the reference segment.


    Notes
    -----
    A 2-node segment element uses the following Gauss points and weights for numerical
    integration:

    +------------+-----------------+-----------------+
    | Point No.  | :math:`(\xi)`   | Weight :math:`w`|
    +============+=================+=================+
    | 1          | :math:`0`       | :math:`2`       |
    +------------+-----------------+-----------------+

    .. figure:: /_static/shape_functions/segment_2_gp.png
        :alt: Gauss quadrature points for 2-node segment element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_segment_2_shape_functions:
        Shape functions for 2-node segment 1D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for a 2-node segment element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_segment_2_gauss_points

        gauss_points = get_segment_2_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.]]

    """
    return _get_gauss_points("segment_2", return_weights)


def get_segment_3_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`\xi` and weights of Gauss quadrature points for a
    3-node segment element.

    The 2-point Gauss-Legendre rule is used. It integrates exactly polynomials up to
    degree 3 over the reference segment :math:`[-1, 1]`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(2, 1)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(2,)`. They sum to :math:`2`, the length of
        the reference segment.


    Notes
    -----
    A 3-node segment element uses the following Gauss points and weights for numerical
    integration:

    +------------+-----------------------------+-----------------+
    | Point No.  | :math:`(\xi)`               | Weight :math:`w`|
    +============+=============================+=================+
    | 1          | :math:`-\frac{1}{\sqrt{3}}` | :math:`1`       |
    +------------+-----------------------------+-----------------+
    | 2          | :math:`\frac{1}{\sqrt{3}}`  | :math:`1`       |
    +------------+-----------------------------+-----------------+

    .. figure:: /_static/shape_functions/segment_3_gp.png
        :alt: Gauss quadrature points for 3-node segment element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_segment_3_shape_functions:
        Shape functions for 3-node segment 1D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for a 3-node segment element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_segment_3_gauss_points

        gauss_points = get_segment_3_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[-0.57735027]
         [ 0.57735027]]

    """
    return _get_gauss_points("segment_3", return_weights)


def get_triangle_3_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature
    points for a 3-node triangle element.

    The 1-point centroid rule is used. It integrates exactly polynomials up to
    degree 1 over the reference triangle :math:`\{\xi, \eta \geq 0, \xi + \eta \leq 1\}`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(1, 2)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(1,)`. They sum to :math:`\frac{1}{2}`, the
        area of the reference triangle.


    Notes
    -----
    A 3-node triangle element uses the following Gauss points and weights for numerical
    integration:

    +------------+-----------------------------------------+---------------------+
    | Point No.  | :math:`(\xi, \eta)`                     | Weight :math:`w`    |
    +============+=========================================+=====================+
    | 1          | :math:`(\frac{1}{3}, \frac{1}{3})`      | :math:`\frac{1}{2}` |
    +------------+-----------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/triangle_3_gp.png
        :alt: Gauss quadrature points for 3-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_triangle_3_shape_functions:
        Shape functions for 3-node triangle 2D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for a 3-node triangle element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_triangle_3_gauss_points

        gauss_points = get_triangle_3_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.33333333 0.33333333]]

    """
    return _get_gauss_points("triangle_3", return_weights)


def get_triangle_6_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature
    points for a 6-node triangle element.

    A 3-point interior rule is used. It integrates exactly polynomials up to
    degree 2 over the reference triangle :math:`\{\xi, \eta \geq 0, \xi + \eta \leq 1\}`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(3, 2)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(3,)`. They sum to :math:`\frac{1}{2}`, the
        area of the reference triangle.


    Notes
    -----
    A 6-node triangle element uses the following Gauss points and weights for numerical
    integration:

    +------------+-----------------------------------------+---------------------+
    | Point No.  | :math:`(\xi, \eta)`                     | Weight :math:`w`    |
    +============+=========================================+=====================+
    | 1          | :math:`(\frac{1}{6}, \frac{1}{6})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+
    | 2          | :math:`(\frac{2}{3}, \frac{1}{6})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+
    | 3          | :math:`(\frac{1}{6}, \frac{2}{3})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/triangle_6_gp.png
        :alt: Gauss quadrature points for 6-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_triangle_6_shape_functions:
        Shape functions for 6-node triangle 2D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for a 6-node triangle element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_triangle_6_gauss_points

        gauss_points = get_triangle_6_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.16666667 0.16666667]
         [0.66666667 0.16666667]
         [0.16666667 0.66666667]]

    """
    return _get_gauss_points("triangle_6", return_weights)


def get_quadrangle_4_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature
    points for a 4-node quadrangle element.

    The :math:`2 \times 2` tensor Gauss-Legendre rule is used. It integrates exactly
    polynomials up to degree 3 in each direction over the reference square
    :math:`[-1, 1]^2`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(4, 2)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(4,)`. They sum to :math:`4`, the area of the
        reference square.


    Notes
    -----
    A 4-node quadrangle element uses the following Gauss points and weights for
    numerical integration:

    +------------+----------------------------------------------------+---------------------+
    | Point No.  | :math:`(\xi, \eta)`                                | Weight :math:`w`    |
    +============+====================================================+=====================+
    | 1          | :math:`(-\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}})` | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 2          | :math:`(\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}})`  | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 3          | :math:`(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}})`   | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 4          | :math:`(-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}})`  | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/quadrangle_4_gp.png
        :alt: Gauss quadrature points for 4-node quadrangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_quadrangle_4_shape_functions:
        Shape functions for 4-node quadrangle 2D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for a 4-node quadrangle element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_quadrangle_4_gauss_points

        gauss_points = get_quadrangle_4_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[-0.57735027 -0.57735027]
         [ 0.57735027 -0.57735027]
         [ 0.57735027  0.57735027]
         [-0.57735027  0.57735027]]

    """
    return _get_gauss_points("quadrangle_4", return_weights)


def get_quadrangle_8_gauss_points(
    return_weights: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature
    points for an 8-node (serendipity) quadrangle element.

    The :math:`3 \times 3` tensor Gauss-Legendre rule is used. It integrates exactly
    polynomials up to degree 5 in each direction over the reference square
    :math:`[-1, 1]^2`.


    Parameters
    ----------
    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The returned array has shape
        :math:`(9, 2)`.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(9,)`. They sum to :math:`4`, the area of the
        reference square.


    Notes
    -----
    An 8-node quadrangle element uses the following Gauss points and weights for
    numerical integration (center, then corners, then mid-edges):

    +------------+----------------------------------------------------+----------------------+
    | Point No.  | :math:`(\xi, \eta)`                                | Weight :math:`w`     |
    +============+====================================================+======================+
    | 1          | :math:`(0, 0)`                                     | :math:`\frac{64}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 2          | :math:`(\sqrt{\frac{3}{5}}, \sqrt{\frac{3}{5}})`   | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 3          | :math:`(-\sqrt{\frac{3}{5}}, \sqrt{\frac{3}{5}})`  | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 4          | :math:`(-\sqrt{\frac{3}{5}}, -\sqrt{\frac{3}{5}})` | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 5          | :math:`(\sqrt{\frac{3}{5}}, -\sqrt{\frac{3}{5}})`  | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 6          | :math:`(0, \sqrt{\frac{3}{5}})`                    | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 7          | :math:`(-\sqrt{\frac{3}{5}}, 0)`                   | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 8          | :math:`(0, -\sqrt{\frac{3}{5}})`                   | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 9          | :math:`(\sqrt{\frac{3}{5}}, 0)`                    | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+

    .. figure:: /_static/shape_functions/quadrangle_8_gp.png
        :alt: Gauss quadrature points for 8-node quadrangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.fem.compute_quadrangle_8_shape_functions:
        Shape functions for 8-node quadrangle 2D-elements.

    pysdic.fem.get_gauss_points:
        Gauss points for any supported element type.


    Examples
    --------
    Get the Gauss points for an 8-node quadrangle element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_quadrangle_8_gauss_points

        gauss_points = get_quadrangle_8_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[ 0.          0.        ]
         [ 0.77459667  0.77459667]
         [-0.77459667  0.77459667]
         [-0.77459667 -0.77459667]
         [ 0.77459667 -0.77459667]
         [ 0.          0.77459667]
         [-0.77459667  0.        ]
         [ 0.         -0.77459667]
         [ 0.77459667  0.        ]]

    """
    return _get_gauss_points("quadrangle_8", return_weights)


def get_gauss_points(
    element_type: str, *, return_weights: bool = False
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates and weights of Gauss quadrature points for a specified
    element type.

    The integral of a function :math:`f` over the reference element is approximated by:

    .. math::

        \int_{\hat{\Omega}} f(\xi) \, d\xi \approx \sum_{g=1}^{N_{gp}} w_g f(\xi_g)


    Parameters
    ----------
    element_type: str
        Type of the finite element (case insensitive). Supported types are:

        - ``'segment_2'``: 2-node segment element.
        - ``'segment_3'``: 3-node segment element.
        - ``'triangle_3'``: 3-node triangle element.
        - ``'triangle_6'``: 6-node triangle element.
        - ``'quadrangle_4'``: 4-node quadrangle element.
        - ``'quadrangle_8'``: 8-node (serendipity) quadrangle element.

    return_weights: bool, optional (default: ``False``)
        If True, the function will also return the weights associated with each
        Gauss point.


    Returns
    -------
    gauss_points: numpy.ndarray
        Natural coordinates of the Gauss points. The shape of the returned array is
        :math:`(N_{gp}, K)`, where :math:`N_{gp}` is the number of Gauss points for
        the specified element type and :math:`K` is the topological dimension of the
        element.

    weights: numpy.ndarray
        Only returned if ``return_weights`` is True. Weights associated with each
        Gauss point, with shape :math:`(N_{gp},)`. They sum to the measure of the
        reference element.


    See Also
    --------
    pysdic.fem.get_segment_2_gauss_points:
        Gauss points for 2-node segment elements.

    pysdic.fem.get_segment_3_gauss_points:
        Gauss points for 3-node segment elements.

    pysdic.fem.get_triangle_3_gauss_points:
        Gauss points for 3-node triangle elements.

    pysdic.fem.get_triangle_6_gauss_points:
        Gauss points for 6-node triangle elements.

    pysdic.fem.get_quadrangle_4_gauss_points:
        Gauss points for 4-node quadrangle elements.

    pysdic.fem.get_quadrangle_8_gauss_points:
        Gauss points for 8-node quadrangle elements.

    pysdic.fem.compute_shape_functions:
        Shape functions for any supported element type.


    Examples
    --------
    Get the Gauss points for a 3-node triangle element:

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import get_gauss_points

        gauss_points_array = get_gauss_points("triangle_3")
        print("Gauss points for triangle_3:")
        print(gauss_points_array)

    .. code-block:: console

        Gauss points for triangle_3:
        [[0.33333333 0.33333333]]

    Integrate :math:`f(\xi, \eta) = \xi \eta` over the reference triangle
    (exact value :math:`\frac{1}{24}`):

    .. code-block:: python
        :linenos:

        points, weights = get_gauss_points("triangle_6", return_weights=True)
        integral = numpy.sum(weights * points[:, 0] * points[:, 1])
        print(integral, 1.0 / 24.0)

    .. code-block:: console

        0.041666666666666664 0.041666666666666664

    """
    element_type = _validate_element_type(element_type)

    return _get_gauss_points(element_type, return_weights)