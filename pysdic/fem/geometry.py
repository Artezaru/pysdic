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
Geometric and mechanical quantities of the elements of a mesh: centers of the elements,
areas and normals of surface elements (:math:`K = 2`), and strains.

All the quantities are computed from the Jacobian matrices
:math:`J = \frac{\partial X}{\partial \Xi}` of the elements (see
:func:`pysdic.fem.compute_jacobian_matrix`), so the same functions work for every
element type (linear or quadratic, triangles or quadrangles).
"""

from numbers import Real
from numpy.typing import ArrayLike

import numpy

from .._validation import _as_array
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import _validate_connectivity, _validate_element_type
from .gauss_points import get_gauss_points
from .operations import (
    compute_jacobian_matrix,
    compute_property_derivative,
    compute_property_interpolation,
)


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_surface_inputs(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    *,
    require_3d: bool,
):
    # Validate the inputs of a surface computation (K = 2, and E = 3 if required)
    element_type = _validate_element_type(element_type)
    properties = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type]
    if properties["expected_K"] != 2:
        raise ValueError(
            f"Only surface elements (K = 2) are supported. Got element type "
            f"'{element_type}' with K={properties['expected_K']}."
        )
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, 3) if require_3d else (None, None),
        shape_hint="(N_v, 3)" if require_3d else "(N_v, E)", finite=False,
    )
    if not require_3d and vertices_coordinates.shape[1] < 2:
        raise ValueError(
            f"Input 'vertices_coordinates' must have E >= 2 for surface elements. "
            f"Got shape {vertices_coordinates.shape}."
        )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, vertices_coordinates.shape[0] or None, properties["expected_N_vpe"]
    )
    return vertices_coordinates, connectivity_array, element_type


def _normalize(vectors: numpy.ndarray) -> numpy.ndarray:
    # Normalize vectors (N, 3): a zero (or non-finite) vector gives NaN
    norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / norms
    unit[~(norms[:, 0] > 0)] = numpy.nan
    return unit


def _elements_centers(element_type: str, n_elements: int):
    # Natural coordinates of the center of each element (mean of the Gauss points)
    gauss_points = get_gauss_points(element_type)
    center = gauss_points.mean(axis=0, keepdims=True)  # (1, K)
    return numpy.repeat(center, n_elements, axis=0), numpy.arange(n_elements)


# =============================================================================
# Centers
# =============================================================================
def compute_elements_centers(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
) -> numpy.ndarray:
    r"""
    Compute the center of each element of a mesh.

    The center of an element is the image :math:`X(\Xi_c)` of the center
    :math:`\Xi_c` of the reference element, the mean of the Gauss points of the element
    type (e.g. :math:`(\frac{1}{3}, \frac{1}{3})` for the triangles and :math:`(0, 0)`
    for the quadrangles). For linear elements, it is the mean of the vertices of the
    element (centroid of the triangles, mean of the 4 vertices of the quadrangles).

    Any element type is supported.

    .. note::

        - Input ``vertices_coordinates`` will be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, E)` with the coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive).

    Returns
    -------
    centers: numpy.ndarray
        An array of shape :math:`(N_e, E)` with the center of each element.

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_elements_centers

        vertices_coordinates = numpy.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 3]])
        connectivity_array = numpy.array([[0, 1, 2], [1, 3, 2]])

        centers = compute_elements_centers(vertices_coordinates, connectivity_array, "triangle_3")
        print(centers)

    .. code-block:: console

        [[1. 1. 0.]
         [2. 2. 1.]]

    """
    element_type = _validate_element_type(element_type)
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, None), shape_hint="(N_v, E)", finite=False,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array,
        vertices_coordinates.shape[0] or None,
        _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type]["expected_N_vpe"],
    )
    natural_coordinates, element_indices = _elements_centers(element_type, connectivity_array.shape[0])
    return compute_property_interpolation(
        vertices_coordinates, connectivity_array, element_type, natural_coordinates, element_indices
    )


# =============================================================================
# Areas
# =============================================================================
def compute_elements_areas(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
) -> numpy.ndarray:
    r"""
    Compute the area of each surface element (:math:`K = 2`) of a mesh.

    The area is integrated with the Gauss points :math:`\Xi_g` and weights :math:`w_g`
    of the element type (see :func:`pysdic.fem.get_gauss_points`):

    .. math::

        A = \int_{\hat{\Omega}} \sqrt{\det(J^T J)} \, d\Xi
          \approx \sum_{g} w_g \sqrt{\det(J(\Xi_g)^T J(\Xi_g))}

    where :math:`J` is the Jacobian matrix of shape :math:`(E, 2)` of the element. The
    area is exact for the elements with a constant Jacobian (``triangle_3``,
    parallelogram ``quadrangle_4``) and approximated by the quadrature otherwise.

    .. note::

        - Input ``vertices_coordinates`` will be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, E)` with the coordinates of the vertices
        (:math:`E \geq 2`).

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type
        (``'triangle_3'``, ``'triangle_6'``, ``'quadrangle_4'`` or ``'quadrangle_8'``).

    Returns
    -------
    areas: numpy.ndarray
        An array of shape :math:`(N_e,)` with the area of each element (NaN for the
        elements with non-finite vertices).

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_elements_areas

        vertices_coordinates = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        areas = compute_elements_areas(vertices_coordinates, connectivity_array, "triangle_3")
        print(areas)

    .. code-block:: console

        [0.5       0.5       0.5       0.8660254]

    """
    vertices_coordinates, connectivity_array, element_type = _validate_surface_inputs(
        vertices_coordinates, connectivity_array, element_type, require_3d=False
    )
    N_e = connectivity_array.shape[0]
    gauss_points, gauss_weights = get_gauss_points(element_type, return_weights=True)
    G = gauss_points.shape[0]

    jacobians = compute_jacobian_matrix(
        vertices_coordinates,
        connectivity_array,
        element_type,
        numpy.tile(gauss_points, (N_e, 1)),
        numpy.repeat(numpy.arange(N_e), G),
    )  # (N_e * G, E, 2)
    metric = numpy.einsum("pek,pel->pkl", jacobians, jacobians)  # J^T J, (N_e * G, 2, 2)
    determinants = numpy.linalg.det(metric) if N_e > 0 else numpy.empty(0)
    surface_elements = numpy.sqrt(numpy.clip(determinants, 0.0, None))  # Rounding errors < 0
    return (surface_elements.reshape(N_e, G) * gauss_weights).sum(axis=1)


# =============================================================================
# Normals
# =============================================================================
def compute_integration_points_normals(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
) -> numpy.ndarray:
    r"""
    Compute the unit normal of a 3D surface mesh (:math:`K = 2`, :math:`E = 3`) at
    integration points.

    The normal is the normalized cross product of the columns of the Jacobian matrix
    :math:`J = \frac{\partial X}{\partial \Xi}` of the element:

    .. math::

        \vec{n} = \frac{\frac{\partial X}{\partial \xi} \times \frac{\partial X}{\partial \eta}}
                       {\| \frac{\partial X}{\partial \xi} \times \frac{\partial X}{\partial \eta} \|}

    Its orientation follows the order of the vertices in the elements (right-hand rule).

    .. note::

        - Inputs ``vertices_coordinates`` and ``natural_coordinates`` will be converted
          to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` with the coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_p, 2)` with the natural coordinates of the points.

    element_indices: ArrayLike
        An array of shape :math:`(N_p,)` with the element index of each point.

    allow_unlocated_points: bool, optional (default: ``False``)
        If True, the points with element index ``-1`` get NaN normals. Otherwise, they
        raise an error.

    tolerance: Real, optional (default: ``0.0``)
        The tolerance to decide if a point is inside the reference element.

    Returns
    -------
    normals: numpy.ndarray
        An array of shape :math:`(N_p, 3)` with the unit normals. The normal is NaN for
        the unlocated points and where the element is degenerated
        (:math:`\frac{\partial X}{\partial \xi} \times \frac{\partial X}{\partial \eta} = 0`).

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_integration_points_normals

        vertices_coordinates = numpy.array([[-1, -1, 0], [1, -1, 0], [1, 1, 1], [-1, 1, 1]])
        connectivity_array = numpy.array([[0, 1, 2, 3]])

        normals = compute_integration_points_normals(
            vertices_coordinates, connectivity_array, "quadrangle_4",
            natural_coordinates=[[0.0, 0.0]], element_indices=[0],
        )
        print(normals)

    .. code-block:: console

        [[ 0.         -0.4472136   0.89442719]]

    """
    vertices_coordinates, connectivity_array, element_type = _validate_surface_inputs(
        vertices_coordinates, connectivity_array, element_type, require_3d=True
    )
    jacobians = compute_jacobian_matrix(
        vertices_coordinates,
        connectivity_array,
        element_type,
        natural_coordinates,
        element_indices,
        allow_unlocated_points=allow_unlocated_points,
        tolerance=tolerance,
        default=numpy.nan,
    )  # (N_p, 3, 2)
    return _normalize(numpy.cross(jacobians[:, :, 0], jacobians[:, :, 1]))


def compute_elements_normals(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
) -> numpy.ndarray:
    r"""
    Compute the unit normal of each element of a 3D surface mesh (:math:`K = 2`,
    :math:`E = 3`), at the center of the element.

    The center is the center of the reference element (the mean of the Gauss points of
    the element type), and the normal is computed as in
    :func:`pysdic.fem.compute_integration_points_normals`. For flat elements, the normal
    is the same everywhere in the element.

    .. note::

        - Input ``vertices_coordinates`` will be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` with the coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type.

    Returns
    -------
    normals: numpy.ndarray
        An array of shape :math:`(N_e, 3)` with the unit normal of each element (NaN for
        the degenerated elements).

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_elements_normals

        vertices_coordinates = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        normals = compute_elements_normals(vertices_coordinates, connectivity_array, "triangle_3")
        print(normals)

    .. code-block:: console

        [[ 0.          0.          1.        ]
         [ 0.         -1.          0.        ]
         [ 1.          0.          0.        ]
         [ 0.57735027  0.57735027  0.57735027]]

    """
    vertices_coordinates, connectivity_array, element_type = _validate_surface_inputs(
        vertices_coordinates, connectivity_array, element_type, require_3d=True
    )
    natural_coordinates, element_indices = _elements_centers(
        element_type, connectivity_array.shape[0]
    )
    return compute_integration_points_normals(
        vertices_coordinates, connectivity_array, element_type,
        natural_coordinates, element_indices,
    )


def compute_vertices_normals(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
) -> numpy.ndarray:
    r"""
    Compute the unit normal at each vertex of a 3D surface mesh (:math:`K = 2`,
    :math:`E = 3`).

    The normal at a vertex is the mean of the normals :math:`\vec{n}_e` of the elements
    using this vertex (see :func:`pysdic.fem.compute_elements_normals`), weighted by
    their areas :math:`A_e` (see :func:`pysdic.fem.compute_elements_areas`):

    .. math::

        \vec{n}_v = \frac{\sum_{e \ni v} A_e \vec{n}_e}{\| \sum_{e \ni v} A_e \vec{n}_e \|}

    The degenerated elements (zero area) are ignored.

    .. note::

        - Input ``vertices_coordinates`` will be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` with the coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type.

    Returns
    -------
    normals: numpy.ndarray
        An array of shape :math:`(N_v, 3)` with the unit normal at each vertex. The
        normal is NaN for the vertices used by no (valid) element, and where the
        weighted sum vanishes.

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_vertices_normals

        vertices_coordinates = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        normals = compute_vertices_normals(vertices_coordinates, connectivity_array, "triangle_3")
        print(normals)

    .. code-block:: console

        [[ 0.57735027 -0.57735027  0.57735027]
         [ 0.4472136   0.          0.89442719]
         [ 0.66666667  0.33333333  0.66666667]
         [ 0.89442719  0.          0.4472136 ]]

    """
    vertices_coordinates, connectivity_array, element_type = _validate_surface_inputs(
        vertices_coordinates, connectivity_array, element_type, require_3d=True
    )
    N_vpe = connectivity_array.shape[1]
    areas = compute_elements_areas(vertices_coordinates, connectivity_array, element_type)
    normals = compute_elements_normals(vertices_coordinates, connectivity_array, element_type)

    valid = (areas > 0) & numpy.all(numpy.isfinite(normals), axis=1)  # (N_e,)
    weighted_normals = normals[valid] * areas[valid, numpy.newaxis]  # (N_valid, 3)

    sums = numpy.zeros_like(vertices_coordinates)  # (N_v, 3)
    numpy.add.at(
        sums,
        connectivity_array[valid].ravel(),
        numpy.repeat(weighted_normals, N_vpe, axis=0),
    )
    return _normalize(sums)


# =============================================================================
# Strains
# =============================================================================
def compute_strains(
    displacements: ArrayLike,
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Compute the linear strain tensor of a displacement field at integration points.

    .. math::

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \nabla U + (\nabla U)^T \right)

    where :math:`\nabla U` of shape :math:`(E, E)` is the gradient of the displacements
    with respect to the global coordinates (see
    :func:`pysdic.fem.compute_property_derivative`). For a surface embedded in 3D
    (:math:`K < E`), it is the tangential gradient.

    Any element type is supported.

    .. note::

        - Inputs ``displacements``, ``vertices_coordinates`` and ``natural_coordinates``
          will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    Parameters
    ----------
    displacements: ArrayLike
        An array of shape :math:`(N_v, E)` with the displacement of each vertex.

    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, E)` with the coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive).

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_p, K)` with the natural coordinates of the points.

    element_indices: ArrayLike
        An array of shape :math:`(N_p,)` with the element index of each point.

    allow_unlocated_points: bool, optional (default: ``False``)
        If True, the points with element index ``-1`` get the value ``default``.
        Otherwise, they raise an error.

    tolerance: Real, optional (default: ``0.0``)
        The tolerance to decide if a point is inside the reference element.

    default: Real, optional (default: ``numpy.nan``)
        The value of the unlocated points.

    Returns
    -------
    strains: numpy.ndarray
        An array of shape :math:`(N_p, E, E)` with the strain tensor at each point.

    Examples
    --------
    Strains at the centroid of each triangle of a tetrahedron surface.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_strains, get_gauss_points

        vertices_coordinates = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        displacements = 0.1 * vertices_coordinates

        natural_coordinates = numpy.repeat(get_gauss_points("triangle_3"), 4, axis=0)
        strains = compute_strains(
            displacements, vertices_coordinates, connectivity_array, "triangle_3",
            natural_coordinates, numpy.arange(4),
        )
        print(numpy.round(strains[0], 6))

    .. code-block:: console

        [[0.1 0.  0. ]
         [0.  0.1 0. ]
         [0.  0.  0. ]]

    """
    gradients = compute_property_derivative(
        displacements,
        vertices_coordinates,
        connectivity_array,
        element_type,
        natural_coordinates,
        element_indices,
        allow_unlocated_points=allow_unlocated_points,
        tolerance=tolerance,
        default=default,
    )  # (N_p, P, E)
    if gradients.shape[1] != gradients.shape[2]:
        raise ValueError(
            f"Input 'displacements' must have shape (N_v, E) with E={gradients.shape[2]} "
            f"the dimension of the vertices. Got {gradients.shape[1]} components."
        )
    return 0.5 * (gradients + numpy.transpose(gradients, (0, 2, 1)))