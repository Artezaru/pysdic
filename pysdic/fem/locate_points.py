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
Location of points in 2D surface meshes (:math:`K = 2`, :math:`E = 2`): inverse of the
mapping :math:`X(\Xi)` of the elements.
"""

from typing import Optional, Tuple
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy

from .._validation import _as_array, _validate_integer, _validate_real
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import _validate_connectivity, _validate_element_type
from .cast_rays import _DEFAULT_SUBDIVISIONS, _clamp_to_reference_element, _subdivide_reference_element
from .shape_functions import compute_shape_functions


__all__ = [
    "locate_points",
]


# Relative enlargement of the bounding boxes of the elements: the curved edges of the
# quadratic elements can go slightly beyond the bounding box of their vertices
_BOX_MARGIN = {
    "triangle_3": 0.0,
    "quadrangle_4": 0.0,
    "triangle_6": 0.25,
    "quadrangle_8": 0.25,
}

# Center of the reference elements: initial guess of the Newton iterations
_REFERENCE_CENTER = {
    "triangle_6": (1.0 / 3.0, 1.0 / 3.0),
    "quadrangle_4": (0.0, 0.0),
    "quadrangle_8": (0.0, 0.0),
}

# Number of points processed at once (bounds the memory of the candidate pairs)
_CHUNK_SIZE = 100_000

# Natural coordinates beyond which the Newton iterations are stopped (diverging point,
# far outside the element)
_DIVERGENCE_LIMIT = 10.0


# =============================================================================
# Internal helpers
# =============================================================================
class _ElementGrid:
    # Uniform grid of cells registering the elements whose bounding box overlaps them,
    # to find the candidate elements of a point (the elements whose box contains it)

    def __init__(self, boxes: numpy.ndarray) -> None:
        # boxes: (N_e, 2, 2) with [min, max] of each element
        self.boxes = boxes
        sizes = numpy.max(boxes[:, 1] - boxes[:, 0], axis=1)
        positive = sizes[sizes > 0]
        self.cell = float(numpy.median(positive)) if positive.size > 0 else 1.0
        self.origin = boxes[:, 0].min(axis=0)
        low = numpy.floor((boxes[:, 0] - self.origin) / self.cell).astype(numpy.int64)
        high = numpy.floor((boxes[:, 1] - self.origin) / self.cell).astype(numpy.int64)
        self.n_cells = high.max(axis=0) + 1  # (n_x, n_y)

        # All the (cell, element) pairs, sorted by cell
        spans = high - low + 1  # (N_e, 2)
        counts = spans[:, 0] * spans[:, 1]
        elements = numpy.repeat(numpy.arange(boxes.shape[0]), counts)
        local = numpy.arange(elements.size) - numpy.repeat(numpy.cumsum(counts) - counts, counts)
        cell_x = low[elements, 0] + local % spans[elements, 0]
        cell_y = low[elements, 1] + local // spans[elements, 0]
        cell_ids = cell_x * self.n_cells[1] + cell_y
        order = numpy.argsort(cell_ids, kind="stable")
        self.elements = elements[order]
        self.cells, self.starts, self.counts = numpy.unique(cell_ids[order], return_index=True, return_counts=True)

    def candidates(self, points: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # The pairs (point index, element index) such that the box of the element contains
        # the point. Non-finite points have no candidate.
        cell = numpy.floor((points - self.origin) / self.cell)
        valid = numpy.all(numpy.isfinite(cell), axis=1)
        valid[valid] = numpy.all((cell[valid] >= 0) & (cell[valid] < self.n_cells), axis=1)
        point_indices = numpy.flatnonzero(valid)
        cell_ids = cell[valid, 0].astype(numpy.int64) * self.n_cells[1] + cell[valid, 1].astype(numpy.int64)
        position = numpy.searchsorted(self.cells, cell_ids)
        position = numpy.minimum(position, self.cells.size - 1)
        found = self.cells[position] == cell_ids
        point_indices, position = point_indices[found], position[found]

        counts = self.counts[position]
        pair_points = numpy.repeat(point_indices, counts)
        local = numpy.arange(pair_points.size) - numpy.repeat(numpy.cumsum(counts) - counts, counts)
        pair_elements = self.elements[numpy.repeat(self.starts[position], counts) + local]

        box = self.boxes[pair_elements]  # (M, 2, 2)
        x = points[pair_points]
        inside = numpy.all((x >= box[:, 0]) & (x <= box[:, 1]), axis=1)
        return pair_points[inside], pair_elements[inside]


def _inside(xi: numpy.ndarray, element_type: str, tolerance: float) -> numpy.ndarray:
    # Mask of the natural coordinates (M, 2) inside the reference element enlarged by the
    # tolerance (False for NaN)
    shape_functions = compute_shape_functions(xi, element_type, default=numpy.nan, tolerance=tolerance)
    return numpy.all(numpy.isfinite(shape_functions), axis=1)


def _first_element(pair_points: numpy.ndarray, pair_elements: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # For each located point, the pair with the element of lowest index. Returns the
    # selected pairs and the corresponding points.
    order = numpy.lexsort((pair_elements, pair_points))
    located, first = numpy.unique(pair_points[order], return_index=True)
    return order[first], located


def _locate_in_triangles(
    triangles: numpy.ndarray, points: numpy.ndarray, tolerance: float, scale: float
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Exact location of the points (N_p, 2) in linear triangles (N_t, 3, 2), with the
    # natural coordinates xi such that X = X_0 + [X_1 - X_0, X_2 - X_0] xi. Returns xi
    # (N_p, 2) (NaN for the unlocated points) and the triangle indices (N_p,) (-1).
    N_p = points.shape[0]
    natural_coordinates = numpy.full((N_p, 2), numpy.nan, dtype=numpy.float64)
    indices = numpy.full(N_p, -1, dtype=numpy.int64)
    margin = 1e-9 * scale  # Rounding errors on the boxes
    grid = _ElementGrid(numpy.stack((triangles.min(axis=1) - margin, triangles.max(axis=1) + margin), axis=1))

    for start in range(0, N_p, _CHUNK_SIZE):
        chunk = points[start : start + _CHUNK_SIZE]
        pair_points, pair_triangles = grid.candidates(chunk)
        if pair_points.size == 0:
            continue
        X = triangles[pair_triangles]  # (M, 3, 2)
        a, b = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
        r = chunk[pair_points] - X[:, 0]
        determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        with numpy.errstate(divide="ignore", invalid="ignore"):  # Degenerated triangles: NaN
            xi = (r[:, 0] * b[:, 1] - r[:, 1] * b[:, 0]) / determinant
            eta = (a[:, 0] * r[:, 1] - a[:, 1] * r[:, 0]) / determinant
        inside = (xi >= -tolerance) & (eta >= -tolerance) & (xi + eta <= 1.0 + tolerance)
        pair_points, pair_triangles = pair_points[inside], pair_triangles[inside]
        xi = numpy.stack((xi[inside], eta[inside]), axis=1)
        selected, located = _first_element(pair_points, pair_triangles)
        natural_coordinates[start + located] = xi[selected]
        indices[start + located] = pair_triangles[selected]
    return natural_coordinates, indices


def _solve_newton(
    element_vertices: numpy.ndarray,
    x: numpy.ndarray,
    element_type: str,
    xi0: numpy.ndarray,
    max_iteration: int,
    tolerance: float,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Natural coordinates of the points x (M, 2) in the elements (M, N_vpe, 2) by Newton
    # iterations on X(xi) - x = 0 from xi0 (M, 2). Returns xi (M, 2) and the converged mask (M,).
    M = x.shape[0]
    xi = numpy.array(xi0, dtype=numpy.float64, copy=True)
    active = numpy.ones(M, dtype=bool)
    converged = numpy.zeros(M, dtype=bool)
    for _ in range(max_iteration):
        idx = numpy.flatnonzero(active)
        if idx.size == 0:
            break
        N, dN = compute_shape_functions(xi[idx], element_type, return_derivatives=True, default=None)
        X = numpy.einsum("mv,mvc->mc", N, element_vertices[idx])  # (m, 2)
        J = numpy.einsum("mvk,mvc->mck", dN, element_vertices[idx])  # (m, 2, 2)

        singular = ~(numpy.abs(numpy.linalg.det(J)) > 0)
        active[idx[singular]] = False  # Degenerated element at this point: not converged
        regular = idx[~singular]
        delta = numpy.linalg.solve(J[~singular], (x[regular] - X[~singular])[..., numpy.newaxis])[..., 0]
        xi[regular] += delta

        done = numpy.max(numpy.abs(delta), axis=1) < tolerance
        converged[regular[done]] = True
        active[regular[done]] = False
        diverged = ~numpy.all(numpy.abs(xi[regular]) < _DIVERGENCE_LIMIT, axis=1)
        active[regular[diverged & ~done]] = False
    return xi, converged


# =============================================================================
# Location
# =============================================================================
def locate_points(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    points: ArrayLike,
    *,
    n_subdivisions: Optional[Integral] = None,
    max_iteration: Integral = 20,
    tolerance: Real = 1e-10,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Locate points in a 2D surface mesh (:math:`K = 2`, :math:`E = 2`): compute the element
    containing each point and its natural coordinates :math:`\Xi = (\xi, \eta)` such that
    :math:`X(\Xi) = \vec{x}`.

    - ``triangle_3``: the mapping is affine, the natural coordinates are computed exactly
      in the candidate elements (the elements whose bounding box contains the point, found
      with a uniform grid).
    - Other element types: the mapping :math:`X(\Xi)` is not affine, so the points are
      located in three steps:

      1. Each element is subdivided into triangles (``n_subdivisions`` segments per edge
         in the reference element) and the points are located exactly in these
         triangles. This gives the element and a first approximation :math:`\Xi_0` of
         the natural coordinates.
      2. The natural coordinates are refined with Newton iterations on
         :math:`X(\Xi) - \vec{x} = 0` from :math:`\Xi_0`.
      3. The points not located by the first two steps (points between a curved edge and
         its subdivision, or near an edge between two elements) are searched in all the
         candidate elements (bounding boxes enlarged for the curved edges), with Newton
         iterations from the center of the reference element.

    A point is located in an element if its natural coordinates are inside the reference
    element (enlarged by ``tolerance``).

    A point is unlocated (element index ``-1``) if it is outside the mesh, not finite, or if
    the Newton iterations do not converge. A point on the boundary between several elements
    (or in overlapping elements) is located in the element of lowest index.

    The natural coordinates accepted within the tolerance are moved onto the reference
    element, so that they are inside their element for the strict checks of the other
    functions.

    .. note::

        - Inputs ``vertices_coordinates`` and ``points`` will be converted to
          ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. seealso::

        - :func:`pysdic.fem.cast_rays` to locate the intersections of rays with a 3D surface
          mesh.
        - :meth:`pysdic.Mesh.locate_points` to get the result as
          :class:`pysdic.IntegrationPoints`.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 2)` with the finite coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type
        (``triangle_3``, ``triangle_6``, ``quadrangle_4`` or ``quadrangle_8``).

    points: ArrayLike
        An array of shape :math:`(N_p, 2)` with the coordinates of the points to locate.
        Non-finite points are unlocated.

    n_subdivisions: Optional[Integral], optional (default: ``None``)
        The number of segments per edge of the reference element for the first
        approximation (:math:`\geq 1`). By default, 1 for ``quadrangle_4`` (2 triangles
        per element) and 2 for the quadratic elements. Ignored for ``triangle_3``.

    max_iteration: Integral, optional (default: ``20``)
        The maximum number of Newton iterations (:math:`\geq 1`). Ignored for
        ``triangle_3``.

    tolerance: Real, optional (default: ``1e-10``)
        The convergence tolerance of the Newton iterations on the natural coordinates, also
        used to check that a point is inside an element (strictly positive).

    Returns
    -------
    natural_coordinates: numpy.ndarray
        An array of shape :math:`(N_p, 2)` with the natural coordinates :math:`(\xi, \eta)`
        of the points (NaN for the unlocated points).

    element_indices: numpy.ndarray
        An array of shape :math:`(N_p,)` with the index of the element containing each
        point (``-1`` for the unlocated points).

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a 2D surface mesh or an input has a wrong shape or value.

    Examples
    --------
    Locate two points in a mesh made of two triangles.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import locate_points

        vertices_coordinates = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 2, 3]])

        natural_coordinates, element_indices = locate_points(
            vertices_coordinates, connectivity_array, "triangle_3", [[0.75, 0.25], [2.0, 0.5]]
        )
        print(natural_coordinates, element_indices)

    .. code-block:: console

        [[0.5  0.25]
         [ nan  nan]] [ 0 -1]

    """
    element_type = _validate_element_type(element_type)
    properties = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type]
    if properties["expected_K"] != 2:
        raise ValueError(
            f"Points can only be located in surface elements (K = 2). Got element type "
            f"'{element_type}' with K={properties['expected_K']}."
        )
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, 2), shape_hint="(N_v, 2)", finite=True,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, vertices_coordinates.shape[0] or None, properties["expected_N_vpe"]
    )
    points = _as_array(points, "points", numpy.float64, shape=(None, 2), shape_hint="(N_p, 2)", finite=False)
    max_iteration = _validate_integer(max_iteration, "max_iteration", min_value=1)
    tolerance = _validate_real(tolerance, "tolerance", positive=True, finite=True)

    n_subdivisions = _validate_integer(n_subdivisions, "n_subdivisions", min_value=1, allow_none=True)
    if n_subdivisions is None:
        n_subdivisions = _DEFAULT_SUBDIVISIONS.get(element_type, 1)

    N_p = points.shape[0]
    natural_coordinates = numpy.full((N_p, 2), numpy.nan, dtype=numpy.float64)
    element_indices = numpy.full(N_p, -1, dtype=numpy.int64)
    if N_p == 0 or connectivity_array.shape[0] == 0:
        return natural_coordinates, element_indices
    element_vertices = vertices_coordinates[connectivity_array]  # (N_e, N_vpe, 2)
    scale = max(float(numpy.max(numpy.abs(vertices_coordinates))), 1.0)

    if element_type == "triangle_3":
        xi, elements = _locate_in_triangles(element_vertices, points, tolerance, scale)
        located = elements >= 0
        natural_coordinates[located] = _clamp_to_reference_element(xi[located], element_type)
        element_indices[located] = elements[located]
        return natural_coordinates, element_indices

    # 1. First approximation with the subdivided elements
    sub_natural, sub_triangles = _subdivide_reference_element(element_type, n_subdivisions)
    S_v, S_t = sub_natural.shape[0], sub_triangles.shape[0]
    sub_shape_functions = compute_shape_functions(sub_natural, element_type)  # (S_v, N_vpe)
    sub_vertices = numpy.einsum("sv,evc->esc", sub_shape_functions, element_vertices)  # (N_e, S_v, 2)
    sub_elements = sub_vertices[:, sub_triangles].reshape((-1, 3, 2))  # (N_e S_t, 3, 2)
    sub_xi, sub_indices = _locate_in_triangles(sub_elements, points, tolerance, scale)

    # 2. Newton iterations on the exact element, from the first approximation
    found = numpy.flatnonzero(sub_indices >= 0)
    elements = sub_indices[found] // S_t
    corners = sub_natural[sub_triangles[sub_indices[found] % S_t]]  # (M, 3, 2)
    u, v = sub_xi[found, 0:1], sub_xi[found, 1:2]
    xi0 = (1.0 - u - v) * corners[:, 0] + u * corners[:, 1] + v * corners[:, 2]
    xi, valid = _solve_newton(element_vertices[elements], points[found], element_type, xi0, max_iteration, tolerance)
    valid &= _inside(xi, element_type, tolerance)
    natural_coordinates[found[valid]] = _clamp_to_reference_element(xi[valid], element_type)
    element_indices[found[valid]] = elements[valid]

    # 3. Remaining points (curved boundaries, points near an edge between two elements):
    # Newton iterations in all the candidate elements, from the center of the reference element
    remaining = numpy.flatnonzero((element_indices < 0) & numpy.all(numpy.isfinite(points), axis=1))
    if remaining.size == 0:
        return natural_coordinates, element_indices
    box_min, box_max = element_vertices.min(axis=1), element_vertices.max(axis=1)
    size = numpy.max(box_max - box_min, axis=1, keepdims=True)
    margin = _BOX_MARGIN[element_type] * size + 1e-9 * scale
    grid = _ElementGrid(numpy.stack((box_min - margin, box_max + margin), axis=1))
    for start in range(0, remaining.size, _CHUNK_SIZE):
        chunk = remaining[start : start + _CHUNK_SIZE]
        pair_points, pair_elements = grid.candidates(points[chunk])
        if pair_points.size == 0:
            continue
        center = numpy.tile(numpy.asarray(_REFERENCE_CENTER[element_type], dtype=numpy.float64), (pair_points.size, 1))
        xi, valid = _solve_newton(
            element_vertices[pair_elements], points[chunk][pair_points], element_type, center, max_iteration, tolerance
        )
        valid &= _inside(xi, element_type, tolerance)
        pair_points, pair_elements, xi = pair_points[valid], pair_elements[valid], xi[valid]
        selected, located = _first_element(pair_points, pair_elements)
        natural_coordinates[chunk[located]] = _clamp_to_reference_element(xi[selected], element_type)
        element_indices[chunk[located]] = pair_elements[selected]
    return natural_coordinates, element_indices
