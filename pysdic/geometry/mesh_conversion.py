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

from typing import Optional, Union
from numbers import Integral

import numpy

from .._validation import _validate_bool, _validate_choice
from .mesh import Mesh
from .connectivity import Connectivity
from .point_cloud import PointCloud


# For each diagonal, the vertices (v0, v1, v2, v3) of a quadrangle used by its two triangles
_QUADRANGLE_SPLITS = {
    "first": ((0, 1, 2), (0, 2, 3)),   # diagonal v0 - v2
    "second": ((0, 1, 3), (1, 2, 3)),  # diagonal v1 - v3
}

# Local edges (corner a, corner b) of the linear elements, in the order of their mid-edge
# nodes in the quadratic elements (triangle_6: 3, 4, 5 / quadrangle_8: 4, 5, 6, 7)
_LINEAR_EDGES = {
    "triangle_3": ((0, 1), (1, 2), (2, 0)),
    "quadrangle_4": ((0, 1), (1, 2), (2, 3), (3, 0)),
}


# =============================================================================
# Internal helpers
# =============================================================================
def _add_mid_edge_nodes(
    mesh: Mesh,
    linear_type: str,
    quadratic_type: str,
    propagate_vertices_properties: bool,
    propagate_elements_properties: bool,
) -> Mesh:
    # Quadratic mesh with one new vertex at the middle of each edge (shared by the elements of the edge)
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if mesh.element_type != linear_type:
        raise ValueError(f"Input 'mesh' must have the element type {linear_type!r}. Got {mesh.element_type!r}.")
    propagate_vertices_properties = _validate_bool(propagate_vertices_properties, "propagate_vertices_properties")
    propagate_elements_properties = _validate_bool(propagate_elements_properties, "propagate_elements_properties")
    N_v, N_e = mesh.n_vertices, mesh.n_elements
    local_edges = numpy.array(_LINEAR_EDGES[linear_type])  # (N_c, 2)
    N_c = local_edges.shape[0]

    # Edges of the elements (N_e, N_c, 2) and unique edges (N_m, 2), sorted by their vertices
    element_edges = mesh.elements[:, local_edges]
    edges, edge_indices = numpy.unique(
        numpy.sort(element_edges.reshape((-1, 2)), axis=1), axis=0, return_inverse=True
    )
    elements = numpy.concatenate((mesh.elements, N_v + edge_indices.reshape((N_e, N_c))), axis=1)

    # Mid-edge vertices at the middle of the straight edges, with the mean of the vertex properties
    coordinates = numpy.concatenate((mesh.coordinates, mesh.coordinates[edges].mean(axis=1)), axis=0)
    vertex_properties = {}
    if propagate_vertices_properties:
        for key, values in mesh.vertices.properties.items():
            values = numpy.asarray(values)
            if not numpy.issubdtype(values.dtype, numpy.floating):
                raise ValueError(
                    f"The vertex property {key!r} (dtype {values.dtype}) cannot be interpolated at the "
                    "mid-edge vertices: only floating point properties can be propagated. Remove it or "
                    "use propagate_vertices_properties=False."
                )
            vertex_properties[key] = numpy.concatenate((values, values[edges].mean(axis=1)), axis=0)

    # Element properties copied, the UV map interpolated in each element (the seams are respected)
    element_properties = {}
    if propagate_elements_properties:
        for key, values in mesh.connectivity.properties.items():
            if key == "uvmap":
                uv = values.reshape((N_e, -1, 2))  # (N_e, N_c, 2): (u, v) of the corners
                mid_uv = uv[:, local_edges].mean(axis=2)  # (N_e, N_c, 2)
                element_properties[key] = numpy.concatenate((uv, mid_uv), axis=1).reshape((N_e, -1))
            else:
                element_properties[key] = numpy.array(values, copy=True)

    vertices = PointCloud(coordinates, n_dimensions=mesh.n_dimensions, properties=vertex_properties)
    connectivity = Connectivity(elements, properties=element_properties, element_type=quadratic_type)
    return Mesh._from_owned(vertices, connectivity)


def quadrangle_4_to_triangle_3(
    mesh: Mesh,
    diagonal: str = "first",
    *,
    seed: Optional[Union[Integral, numpy.random.Generator]] = None,
    propagate_vertices_properties: bool = True,
    propagate_elements_properties: bool = True,
) -> Mesh:
    r"""
    Convert a ``quadrangle_4`` mesh to a ``triangle_3`` mesh by splitting each
    quadrangle into two triangles.

    The quadrangle :math:`q` with vertices :math:`(v_0, v_1, v_2, v_3)` gives the
    triangles :math:`2q` and :math:`2q + 1`, along one of its diagonals:

    - ``"first"``: diagonal :math:`v_0 - v_2`, triangles :math:`(v_0, v_1, v_2)` and
      :math:`(v_0, v_2, v_3)`.
    - ``"second"``: diagonal :math:`v_1 - v_3`, triangles :math:`(v_0, v_1, v_3)` and
      :math:`(v_1, v_2, v_3)`.
    - ``"alternate"``: ``"first"`` for the even quadrangles and ``"second"`` for the odd
      ones.
    - ``"random"``: a random diagonal for each quadrangle (see ``seed``).

    The triangles keep the orientation of the quadrangles. The vertices are unchanged
    (same order).

    Parameters
    ----------
    mesh: :class:`pysdic.Mesh`
        A mesh with element type ``"quadrangle_4"``.

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed (or the random generator) of ``diagonal="random"``, for reproducible
        results. Ignored for the other diagonals.

    propagate_vertices_properties: bool, optional (default: ``True``)
        If True, the vertex properties are copied to the triangle mesh.

    propagate_elements_properties: bool, optional (default: ``True``)
        If True, the element properties are copied to both triangles of each
        quadrangle. The UV map (element property ``"uvmap"``) is split with the
        quadrangles.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``triangle_3`` mesh, with :math:`2 N_e` elements.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import Mesh, quadrangle_4_to_triangle_3

        mesh = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]], element_type="quadrangle_4")
        triangle_mesh = quadrangle_4_to_triangle_3(mesh, "second")
        print(triangle_mesh.elements)  # [[0 1 3] [1 2 3]]

    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if mesh.element_type != "quadrangle_4":
        raise ValueError(
            f"Input 'mesh' must have the element type 'quadrangle_4'. Got {mesh.element_type!r}."
        )
    diagonal = _validate_choice(diagonal, "diagonal", ("first", "second", "alternate", "random"))
    propagate_vertices_properties = _validate_bool(propagate_vertices_properties, "propagate_vertices_properties")
    propagate_elements_properties = _validate_bool(propagate_elements_properties, "propagate_elements_properties")
    N_e = mesh.n_elements

    # Diagonal of each quadrangle: True for "first", False for "second"
    if diagonal in ("first", "second"):
        use_first = numpy.full(N_e, diagonal == "first")
    elif diagonal == "alternate":
        use_first = numpy.arange(N_e) % 2 == 0
    else:
        if isinstance(seed, bool) or (seed is not None and not isinstance(seed, (Integral, numpy.random.Generator))):
            raise TypeError(
                f"Input 'seed' must be None, an integer or a numpy.random.Generator. Got type {type(seed)}."
            )
        use_first = numpy.random.default_rng(seed).integers(0, 2, size=N_e).astype(bool)

    # Local indices (0..3) of the vertices of the two triangles of each quadrangle: (N_e, 2, 3)
    local = numpy.where(
        use_first[:, numpy.newaxis, numpy.newaxis],
        numpy.array(_QUADRANGLE_SPLITS["first"]),
        numpy.array(_QUADRANGLE_SPLITS["second"]),
    )
    quadrangles = mesh.connectivity._elements  # (N_e, 4)
    triangles = numpy.take_along_axis(
        quadrangles[:, numpy.newaxis, :], local, axis=2
    ).reshape((2 * N_e, 3))  # Triangles 2q and 2q + 1 of the quadrangle q

    element_properties = {}
    if propagate_elements_properties:
        for key, values in mesh.connectivity.properties.items():
            if key == "uvmap":
                uv = values.reshape((N_e, 4, 2))  # (u_i, v_i) of the 4 vertices
                uv = numpy.take_along_axis(uv[:, numpy.newaxis], local[..., numpy.newaxis], axis=2)
                element_properties[key] = uv.reshape((2 * N_e, 6))
            else:
                element_properties[key] = numpy.repeat(values, 2, axis=0)

    vertices = mesh.vertices.copy(copy_properties=propagate_vertices_properties)
    connectivity = Connectivity(triangles, properties=element_properties, element_type="triangle_3")
    return Mesh._from_owned(vertices, connectivity)

def triangle_3_to_triangle_6(
    mesh: Mesh,
    *,
    propagate_vertices_properties: bool = True,
    propagate_elements_properties: bool = True,
) -> Mesh:
    r"""
    Convert a ``triangle_3`` mesh to a ``triangle_6`` mesh by adding a vertex at the middle
    of each edge.

    The triangle :math:`(v_0, v_1, v_2)` gives the element
    :math:`(v_0, v_1, v_2, m_{01}, m_{12}, m_{20})`, where :math:`m_{ab}` is the mid-edge
    vertex of the edge :math:`v_a - v_b`. A mid-edge vertex is shared by the two elements of
    its edge, so the mesh stays conforming.

    The original vertices are unchanged (same order, indices :math:`0` to :math:`N_v - 1`).
    The mid-edge vertices are added after them, one per edge, in the lexicographic order of
    the (sorted) indices of the vertices of their edge. They are at the middle of the
    straight edges: the geometry of the mesh is unchanged, curved edges can be obtained by
    moving them afterwards.

    Parameters
    ----------
    mesh: :class:`pysdic.Mesh`
        A mesh with element type ``"triangle_3"``.

    propagate_vertices_properties: bool, optional (default: ``True``)
        If True, the vertex properties are copied to the original vertices and interpolated
        (mean of the two vertices of the edge) at the mid-edge vertices. Only floating point
        properties can be interpolated.

    propagate_elements_properties: bool, optional (default: ``True``)
        If True, the element properties are copied (same elements). The UV map (element
        property ``"uvmap"``) is interpolated at the mid-edge vertices in each element, so
        the seams of the UV map are respected.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``triangle_6`` mesh, with the same :math:`N_e` elements and
        :math:`N_v + N_{edges}` vertices.

    Raises
    ------
    TypeError
        If ``mesh`` is not a :class:`pysdic.Mesh`.

    ValueError
        If the element type is not ``"triangle_3"``, or if a vertex property to propagate is
        not a floating point property.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import Mesh, triangle_3_to_triangle_6

        mesh = Mesh([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2], [1, 3, 2]], element_type="triangle_3")
        quadratic_mesh = triangle_3_to_triangle_6(mesh)
        print(quadratic_mesh.n_vertices)  # 9: 4 corners and 5 edges
        print(quadratic_mesh.elements)  # [[0 1 2 4 6 5] [1 3 2 7 8 6]]

    """
    return _add_mid_edge_nodes(
        mesh, "triangle_3", "triangle_6", propagate_vertices_properties, propagate_elements_properties
    )


def quadrangle_4_to_quadrangle_8(
    mesh: Mesh,
    *,
    propagate_vertices_properties: bool = True,
    propagate_elements_properties: bool = True,
) -> Mesh:
    r"""
    Convert a ``quadrangle_4`` mesh to a ``quadrangle_8`` mesh by adding a vertex at the
    middle of each edge.

    The quadrangle :math:`(v_0, v_1, v_2, v_3)` gives the element
    :math:`(v_0, v_1, v_2, v_3, m_{01}, m_{12}, m_{23}, m_{30})`, where :math:`m_{ab}` is the
    mid-edge vertex of the edge :math:`v_a - v_b`. A mid-edge vertex is shared by the two
    elements of its edge, so the mesh stays conforming.

    The original vertices are unchanged (same order, indices :math:`0` to :math:`N_v - 1`).
    The mid-edge vertices are added after them, one per edge, in the lexicographic order of
    the (sorted) indices of the vertices of their edge. They are at the middle of the
    straight edges: the geometry of the mesh is unchanged, curved edges can be obtained by
    moving them afterwards.

    Parameters
    ----------
    mesh: :class:`pysdic.Mesh`
        A mesh with element type ``"quadrangle_4"``.

    propagate_vertices_properties: bool, optional (default: ``True``)
        If True, the vertex properties are copied to the original vertices and interpolated
        (mean of the two vertices of the edge) at the mid-edge vertices. Only floating point
        properties can be interpolated.

    propagate_elements_properties: bool, optional (default: ``True``)
        If True, the element properties are copied (same elements). The UV map (element
        property ``"uvmap"``) is interpolated at the mid-edge vertices in each element, so
        the seams of the UV map are respected.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``quadrangle_8`` mesh, with the same :math:`N_e` elements and
        :math:`N_v + N_{edges}` vertices.

    Raises
    ------
    TypeError
        If ``mesh`` is not a :class:`pysdic.Mesh`.

    ValueError
        If the element type is not ``"quadrangle_4"``, or if a vertex property to propagate
        is not a floating point property.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import Mesh, quadrangle_4_to_quadrangle_8

        mesh = Mesh([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]], element_type="quadrangle_4")
        quadratic_mesh = quadrangle_4_to_quadrangle_8(mesh)
        print(quadratic_mesh.elements)  # [[0 1 2 3 4 6 7 5]]

    """
    return _add_mid_edge_nodes(
        mesh, "quadrangle_4", "quadrangle_8", propagate_vertices_properties, propagate_elements_properties
    )