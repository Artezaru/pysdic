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
Intersection of rays with 3D surface meshes (:math:`K = 2`, :math:`E = 3`).

The intersections with ``triangle_3`` meshes are computed with Open3D (optional
dependency). The other surface elements are subdivided into triangles to find a first
approximation of the intersections with Open3D, which is then refined with Newton
iterations on the exact (bilinear or quadratic) surface of the elements.
"""

from typing import Optional, Tuple
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy

from .._optionals import import_optional
from .._validation import _as_array, _validate_bool, _validate_integer, _validate_real
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import _validate_connectivity, _validate_element_type
from .adjacency import assemble_elements_adjacency_graph
from .shape_functions import compute_shape_functions


__all__ = [
    "triangle_3_cast_rays",
    "cast_rays",
]


# Default number of subdivisions of each edge of the elements for the first
# approximation of the intersections (see cast_rays)
_DEFAULT_SUBDIVISIONS = {
    "triangle_6": 2,
    "quadrangle_4": 1,
    "quadrangle_8": 2,
}


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_rays(ray_origins: ArrayLike, ray_directions: ArrayLike) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Rays: finite origins and non-zero finite directions of shape (N_r, 3)
    ray_origins = _as_array(
        ray_origins, "ray_origins", numpy.float64, shape=(None, 3), shape_hint="(N_r, 3)", finite=True
    )
    ray_directions = _as_array(
        ray_directions, "ray_directions", numpy.float64,
        shape=(ray_origins.shape[0], 3), shape_hint=f"({ray_origins.shape[0]}, 3)", finite=True,
    )
    if numpy.any(numpy.all(ray_directions == 0, axis=1)):
        raise ValueError("Input 'ray_directions' must not contain zero vectors.")
    return ray_origins, ray_directions


def _validate_surface_mesh(
    vertices_coordinates: ArrayLike, connectivity_array: ArrayLike, element_type: str
) -> Tuple[numpy.ndarray, numpy.ndarray, str]:
    # 3D surface mesh (K = 2, E = 3) with finite vertices
    element_type = _validate_element_type(element_type)
    properties = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type]
    if properties["expected_K"] != 2:
        raise ValueError(
            f"Rays can only intersect surface elements (K = 2). Got element type "
            f"'{element_type}' with K={properties['expected_K']}."
        )
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, 3), shape_hint="(N_v, 3)", finite=True,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, vertices_coordinates.shape[0] or None, properties["expected_N_vpe"]
    )
    return vertices_coordinates, connectivity_array, element_type


def _triangle_intersections(
    triangles: numpy.ndarray, ray_origins: numpy.ndarray, ray_directions: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Intersections (Moller-Trumbore, float64) of rays with the planes of given triangles.
    # triangles: (M, 3, 3) vertex coordinates. Returns the natural coordinates (M, 2) such
    # that X = (1 - u - v) X_0 + u X_1 + v X_2, and the distances t (M,) along the rays.
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    p = numpy.cross(ray_directions, edge_2)
    determinant = numpy.einsum("mc,mc->m", edge_1, p)
    s = ray_origins - triangles[:, 0]
    q = numpy.cross(s, edge_1)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        u = numpy.einsum("mc,mc->m", s, p) / determinant
        v = numpy.einsum("mc,mc->m", ray_directions, q) / determinant
        t = numpy.einsum("mc,mc->m", edge_2, q) / determinant
    return numpy.stack((u, v), axis=1), t


def _subdivide_reference_element(element_type: str, n_subdivisions: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Regular subdivision of the reference element into triangles, with n_subdivisions
    # segments per edge. Returns the natural coordinates of the sub-vertices (S_v, 2) and
    # the sub-triangles (S_t, 3).
    n = n_subdivisions
    if element_type.startswith("quadrangle"):
        # Reference square [-1, 1]^2: (n + 1)^2 vertices, 2 n^2 triangles
        coordinates = numpy.linspace(-1.0, 1.0, n + 1)
        xi, eta = numpy.meshgrid(coordinates, coordinates, indexing="xy")
        vertices = numpy.stack((xi.ravel(), eta.ravel()), axis=1)  # index = j (n + 1) + i
        i, j = numpy.meshgrid(numpy.arange(n), numpy.arange(n), indexing="xy")
        a = (j * (n + 1) + i).ravel()  # (xi_i, eta_j)
        b, c, d = a + 1, a + n + 2, a + n + 1  # (i+1, j), (i+1, j+1), (i, j+1)
        triangles = numpy.concatenate((numpy.stack((a, b, c), axis=1), numpy.stack((a, c, d), axis=1)))
    else:
        # Reference triangle {xi, eta >= 0, xi + eta <= 1}: (n + 1)(n + 2) / 2 vertices, n^2 triangles
        i, j = numpy.meshgrid(numpy.arange(n + 1), numpy.arange(n + 1), indexing="xy")
        keep = (i + j) <= n
        i, j = i[keep], j[keep]
        vertices = numpy.stack((i / n, j / n), axis=1).astype(numpy.float64)
        index = -numpy.ones((n + 1, n + 1), dtype=numpy.int64)
        index[i, j] = numpy.arange(i.size)
        upward = [(index[a, b], index[a + 1, b], index[a, b + 1]) for a in range(n) for b in range(n - a)]
        downward = [(index[a + 1, b], index[a + 1, b + 1], index[a, b + 1]) for a in range(n) for b in range(n - a - 1)]
        triangles = numpy.array(upward + downward, dtype=numpy.int64).reshape((-1, 3))
    return vertices, triangles


def _clamp_to_reference_element(xi: numpy.ndarray, element_type: str) -> numpy.ndarray:
    # Move natural coordinates (N, 2) just outside the reference element (rounding errors)
    # onto its boundary: [-1, 1]^2 for the quadrangles, the unit triangle for the triangles
    if element_type.startswith("quadrangle"):
        return numpy.clip(xi, -1.0, 1.0)
    xi = numpy.clip(xi, 0.0, None)
    excess = xi.sum(axis=1) - 1.0
    outside = excess > 0
    xi[outside] -= excess[outside, numpy.newaxis] / 2.0
    return numpy.clip(xi, 0.0, 1.0)


# Center of the reference elements: initial guess of the Newton iterations in the
# neighboring elements (see cast_rays, option retry_neighbors)
_REFERENCE_CENTER = {
    "triangle": (1.0 / 3.0, 1.0 / 3.0),
    "quadrangle": (0.0, 0.0),
}


def _newton_intersections(
    X_e: numpy.ndarray,
    O: numpy.ndarray,
    d: numpy.ndarray,
    xi: numpy.ndarray,
    t: numpy.ndarray,
    element_type: str,
    max_iteration: int,
    tolerance: float,
) -> numpy.ndarray:
    # Newton iterations on X(xi) - O - t d = 0 for the rays (M,) in the elements X_e
    # (M, N_vpe, 3), from (xi, t) updated in place. Returns the mask (M,) of the rays whose
    # intersection is converged, inside its element and in the direction of the ray.
    active = numpy.ones(O.shape[0], dtype=bool)  # Rays still iterating
    converged = numpy.zeros(O.shape[0], dtype=bool)
    for _ in range(max_iteration):
        idx = numpy.flatnonzero(active)
        if idx.size == 0:
            break
        N, dN = compute_shape_functions(xi[idx], element_type, return_derivatives=True, default=None)
        X = numpy.einsum("mv,mvc->mc", N, X_e[idx])  # (m, 3)
        J = numpy.einsum("mvk,mvc->mck", dN, X_e[idx])  # (m, 3, 2)
        A = numpy.concatenate((J, -d[idx, :, numpy.newaxis]), axis=2)  # (m, 3, 3)
        residual = X - O[idx] - t[idx, numpy.newaxis] * d[idx]

        singular = ~(numpy.abs(numpy.linalg.det(A)) > 0)
        active[idx[singular]] = False  # Ray tangent to the surface: not converged
        regular = idx[~singular]
        delta = numpy.linalg.solve(A[~singular], -residual[~singular, :, numpy.newaxis])[..., 0]  # (m, 3)
        xi[regular] += delta[:, :2]
        t[regular] += delta[:, 2]

        done = numpy.max(numpy.abs(delta[:, :2]), axis=1) < tolerance
        converged[regular[done]] = True
        active[regular[done]] = False

    inside = numpy.all(
        numpy.isfinite(compute_shape_functions(xi, element_type, default=numpy.nan, tolerance=tolerance)),
        axis=1,
    )
    return converged & inside & (t >= 0)


# =============================================================================
# Triangles
# =============================================================================
def triangle_3_cast_rays(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    ray_origins: ArrayLike,
    ray_directions: ArrayLike,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the first intersection of rays with a 3D ``triangle_3`` mesh.

    The intersected triangles are found with the ray casting scene of Open3D (in single
    precision), then the natural coordinates of the intersections are recomputed in
    double precision in these triangles. An intersection found by Open3D just outside
    its triangle (because of the single precision, for a ray passing close to an edge)
    is moved onto the edge of the triangle.

    A ray is a half-line: only the intersections in the direction of the ray
    (:math:`t \geq 0`) are found. Both sides of the triangles are intersected.

    .. note::

        - Inputs ``vertices_coordinates``, ``ray_origins`` and ``ray_directions`` will
          be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. important::

        Requires the optional package ``open3d`` (``pip install open3d``).

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` with the finite coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, 3)` with the vertex indices of each triangle.

    ray_origins: ArrayLike
        An array of shape :math:`(N_r, 3)` with the finite origins of the rays.

    ray_directions: ArrayLike
        An array of shape :math:`(N_r, 3)` with the finite non-zero directions of the
        rays (not necessarily normalized).

    Returns
    -------
    natural_coordinates: numpy.ndarray
        An array of shape :math:`(N_r, 2)` with the natural coordinates
        :math:`(\xi, \eta)` of the intersections (NaN for the rays without
        intersection).

    element_indices: numpy.ndarray
        An array of shape :math:`(N_r,)` with the index of the intersected triangle
        (``-1`` for the rays without intersection).

    See Also
    --------
    pysdic.fem.cast_rays:
        Intersection of rays with any surface mesh.

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import triangle_3_cast_rays

        vertices_coordinates = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        connectivity_array = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

        ray_origins = numpy.array([[0.1, 0.2, -1.0], [2.0, 2.0, -1.0]])
        ray_directions = numpy.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        natural_coordinates, element_indices = triangle_3_cast_rays(
            vertices_coordinates, connectivity_array, ray_origins, ray_directions
        )
        print(natural_coordinates)
        print(element_indices)

    .. code-block:: console

        [[0.1 0.2]
         [nan nan]]
        [ 0 -1]

    """
    open3d = import_optional("open3d", "pysdic.fem.triangle_3_cast_rays")
    vertices_coordinates, connectivity_array, _ = _validate_surface_mesh(
        vertices_coordinates, connectivity_array, "triangle_3"
    )
    ray_origins, ray_directions = _validate_rays(ray_origins, ray_directions)

    N_r = ray_origins.shape[0]
    natural_coordinates = numpy.full((N_r, 2), numpy.nan, dtype=numpy.float64)
    element_indices = numpy.full(N_r, -1, dtype=numpy.int64)
    if N_r == 0 or connectivity_array.shape[0] == 0:
        return natural_coordinates, element_indices

    # First intersection with Open3D (single precision)
    o3d_mesh = open3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = open3d.core.Tensor(
        vertices_coordinates.astype(numpy.float32), dtype=open3d.core.float32
    )
    o3d_mesh.triangle.indices = open3d.core.Tensor(
        connectivity_array.astype(numpy.int32), dtype=open3d.core.int32
    )
    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)
    rays = numpy.concatenate((ray_origins, ray_directions), axis=1).astype(numpy.float32)
    results = scene.cast_rays(open3d.core.Tensor(rays, dtype=open3d.core.float32))

    hit = numpy.isfinite(results["t_hit"].numpy())  # (N_r,)
    hit_triangles = results["primitive_ids"].numpy()[hit].astype(numpy.int64)

    # Natural coordinates in double precision in the intersected triangles
    triangles = vertices_coordinates[connectivity_array[hit_triangles]]  # (M, 3, 3)
    uv, _ = _triangle_intersections(triangles, ray_origins[hit], ray_directions[hit])
    # Single precision errors near the edges: move the point onto the reference triangle
    natural_coordinates[hit] = _clamp_to_reference_element(uv, "triangle_3")
    element_indices[hit] = hit_triangles
    return natural_coordinates, element_indices


# =============================================================================
# Any surface element
# =============================================================================
def cast_rays(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    ray_origins: ArrayLike,
    ray_directions: ArrayLike,
    *,
    n_subdivisions: Optional[Integral] = None,
    max_iteration: Integral = 20,
    tolerance: Real = 1e-10,
    retry_neighbors: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the first intersection of rays with a 3D surface mesh (:math:`K = 2`,
    :math:`E = 3`) of any element type.

    - ``triangle_3``: the intersections are computed by
      :func:`pysdic.fem.triangle_3_cast_rays` (exact).
    - Other element types: the surface :math:`X(\Xi)` of an element is not flat, so the
      intersections are computed in two (or three) steps:

      1. Each element is subdivided into triangles (``n_subdivisions`` segments per
         edge in the reference element) and the rays are intersected with these
         triangles (:func:`pysdic.fem.triangle_3_cast_rays`). This gives the element
         and a first approximation :math:`(\Xi_0, t_0)` of each intersection.
      2. The intersection with the exact surface of the element is refined with Newton
         iterations on :math:`X(\Xi) - (O + t\,d) = 0` (3 equations, unknowns
         :math:`\xi, \eta, t`).
      3. Optional (``retry_neighbors=True``): the rays rejected by the second step (not
         converged, out of the element or behind the origin) are searched again in the
         elements at a distance :math:`\leq 1` of the element of the first approximation
         (the element itself and the elements sharing a vertex with it, see
         :func:`pysdic.fem.assemble_elements_adjacency_graph`), with Newton iterations
         from the center of the reference element and the distance :math:`t_0`. The valid
         intersection with the smallest distance :math:`t` is kept.

    A ray is unlocated (element index ``-1``) if it intersects no element, if the Newton
    iterations do not converge (e.g. for a ray tangent to the surface), or if the
    refined intersection leaves the element (by more than ``tolerance`` in natural
    coordinates). The last case may happen for a ray passing very close to an edge of a
    curved mesh: the subdivided surface (chords of the curved edges) is intersected on the
    side of an element, while the exact intersection is in its neighbor. The Newton
    iterations do not change the element, so the ray is rejected. Increase
    ``n_subdivisions`` to reduce this band, or set ``retry_neighbors=True`` to search the
    intersection in the neighboring elements.

    .. note::

        The third step only concerns the rejected rays, so its cost is usually small. The
        first intersection along the ray is not guaranteed for a ray recovered in a
        neighbor: it is the closest valid intersection among the neighbors.

    .. note::

        - Inputs ``vertices_coordinates``, ``ray_origins`` and ``ray_directions`` will
          be converted to ``numpy.float64``.
        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. important::

        Requires the optional package ``open3d`` (``pip install open3d``).

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` with the finite coordinates of the vertices.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` with the vertex indices of each
        element.

    element_type: str
        The type of the elements (case insensitive), a surface element type.

    ray_origins: ArrayLike
        An array of shape :math:`(N_r, 3)` with the finite origins of the rays.

    ray_directions: ArrayLike
        An array of shape :math:`(N_r, 3)` with the finite non-zero directions of the
        rays (not necessarily normalized).

    n_subdivisions: Optional[Integral], optional (default: ``None``)
        The number of segments per edge of the reference element for the first
        approximation (:math:`\geq 1`). By default, 1 for ``quadrangle_4`` (2 triangles
        per element) and 2 for the quadratic elements. Ignored for ``triangle_3``.

    max_iteration: Integral, optional (default: ``20``)
        The maximum number of Newton iterations (:math:`\geq 1`). Ignored for
        ``triangle_3``.

    tolerance: Real, optional (default: ``1e-10``)
        The convergence tolerance of the Newton iterations on the natural coordinates,
        also used to check that the intersection is inside the element. Ignored for
        ``triangle_3``.

    retry_neighbors: bool, optional (default: ``False``)
        If True, the rays rejected by the Newton iterations in the element of their first
        approximation are searched again in the elements at a distance :math:`\leq 1`
        (the element itself, restarted from its center, and the elements sharing a vertex).
        Ignored for ``triangle_3`` (exact intersections).

    Returns
    -------
    natural_coordinates: numpy.ndarray
        An array of shape :math:`(N_r, 2)` with the natural coordinates
        :math:`(\xi, \eta)` of the intersections (NaN for the unlocated rays).

    element_indices: numpy.ndarray
        An array of shape :math:`(N_r,)` with the index of the intersected element
        (``-1`` for the unlocated rays).

    Examples
    --------
    Intersect a warped quadrangle with a vertical ray.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import cast_rays

        vertices_coordinates = numpy.array([[-1, -1, 0], [1, -1, 0], [1, 1, 1], [-1, 1, 0]])
        connectivity_array = numpy.array([[0, 1, 2, 3]])

        natural_coordinates, element_indices = cast_rays(
            vertices_coordinates, connectivity_array, "quadrangle_4",
            ray_origins=[[0.5, 0.5, 5.0]], ray_directions=[[0.0, 0.0, -1.0]],
        )
        print(natural_coordinates, element_indices)

    .. code-block:: console

        [[0.5 0.5]] [0]

    """
    vertices_coordinates, connectivity_array, element_type = _validate_surface_mesh(
        vertices_coordinates, connectivity_array, element_type
    )
    if element_type == "triangle_3":
        return triangle_3_cast_rays(vertices_coordinates, connectivity_array, ray_origins, ray_directions)

    n_subdivisions = _validate_integer(n_subdivisions, "n_subdivisions", min_value=1, allow_none=True)
    if n_subdivisions is None:
        n_subdivisions = _DEFAULT_SUBDIVISIONS[element_type]
    max_iteration = _validate_integer(max_iteration, "max_iteration", min_value=1)
    tolerance = _validate_real(tolerance, "tolerance", positive=True, finite=True)
    retry_neighbors = _validate_bool(retry_neighbors, "retry_neighbors")
    ray_origins, ray_directions = _validate_rays(ray_origins, ray_directions)
    N_e = connectivity_array.shape[0]

    # 1. First approximation with the subdivided elements
    sub_natural, sub_triangles = _subdivide_reference_element(element_type, n_subdivisions)
    S_v, S_t = sub_natural.shape[0], sub_triangles.shape[0]
    sub_shape_functions = compute_shape_functions(sub_natural, element_type)  # (S_v, N_vpe)
    element_vertices = vertices_coordinates[connectivity_array]  # (N_e, N_vpe, 3)
    sub_vertices = numpy.einsum("sv,evc->esc", sub_shape_functions, element_vertices)  # (N_e, S_v, 3)
    sub_connectivity = sub_triangles[numpy.newaxis] + (S_v * numpy.arange(N_e))[:, numpy.newaxis, numpy.newaxis]

    sub_uv, sub_indices = triangle_3_cast_rays(
        sub_vertices.reshape((-1, 3)), sub_connectivity.reshape((-1, 3)), ray_origins, ray_directions
    )
    N_r = ray_origins.shape[0]
    natural_coordinates = numpy.full((N_r, 2), numpy.nan, dtype=numpy.float64)
    element_indices = numpy.full(N_r, -1, dtype=numpy.int64)
    rays = numpy.flatnonzero(sub_indices >= 0)
    if rays.size == 0:
        return natural_coordinates, element_indices

    elements = sub_indices[rays] // S_t
    corners = sub_natural[sub_triangles[sub_indices[rays] % S_t]]  # (M, 3, 2)
    u, v = sub_uv[rays, 0:1], sub_uv[rays, 1:2]
    xi = (1.0 - u - v) * corners[:, 0] + u * corners[:, 1] + v * corners[:, 2]  # (M, 2)

    # 2. Newton iterations on X(xi) - O - t d = 0
    X_e = element_vertices[elements]  # (M, N_vpe, 3)
    O, d = ray_origins[rays], ray_directions[rays]
    X = numpy.einsum("mv,mvc->mc", compute_shape_functions(xi, element_type, default=None), X_e)
    t = numpy.einsum("mc,mc->m", X - O, d) / numpy.einsum("mc,mc->m", d, d)  # Projection on the ray
    t_init = t.copy()
    valid = _newton_intersections(X_e, O, d, xi, t, element_type, max_iteration, tolerance)

    # 3. Optional: the rays rejected in their element are searched in its neighbors
    if retry_neighbors and not numpy.all(valid):
        rejected = numpy.flatnonzero(~valid)
        # Elements at a distance <= 1: the element itself (Newton restarted from the center of
        # the element, the first start may have converged to a root of the polynomial
        # extension of the surface) and the elements sharing a vertex with it
        graph = assemble_elements_adjacency_graph(connectivity_array).tolil()
        graph.setdiag(True)
        graph = graph.tocsr()
        graph.sort_indices()
        counts = numpy.diff(graph.indptr)[elements[rejected]]
        pair_rays = numpy.repeat(rejected, counts)  # Index in the rays of the first pass
        local = numpy.arange(pair_rays.size) - numpy.repeat(numpy.cumsum(counts) - counts, counts)
        pair_elements = graph.indices[numpy.repeat(graph.indptr[elements[rejected]], counts) + local]
        if pair_rays.size > 0:
            center = _REFERENCE_CENTER["quadrangle" if element_type.startswith("quadrangle") else "triangle"]
            pair_xi = numpy.tile(numpy.asarray(center, dtype=numpy.float64), (pair_rays.size, 1))
            pair_t = t_init[pair_rays].copy()
            pair_valid = _newton_intersections(
                element_vertices[pair_elements], O[pair_rays], d[pair_rays], pair_xi, pair_t,
                element_type, max_iteration, tolerance,
            )
            # For each ray, the valid intersection with the smallest distance t
            pair_rays, pair_elements = pair_rays[pair_valid], pair_elements[pair_valid]
            pair_xi, pair_t = pair_xi[pair_valid], pair_t[pair_valid]
            order = numpy.lexsort((pair_t, pair_rays))
            retried, first = numpy.unique(pair_rays[order], return_index=True)
            selected = order[first]
            xi[retried], t[retried] = pair_xi[selected], pair_t[selected]
            elements[retried] = pair_elements[selected]
            valid[retried] = True

    # Points accepted within the tolerance are moved onto the reference element, so that
    # they are inside their element for the strict checks of the other functions
    natural_coordinates[rays[valid]] = _clamp_to_reference_element(xi[valid], element_type)
    element_indices[rays[valid]] = elements[valid]
    return natural_coordinates, element_indices