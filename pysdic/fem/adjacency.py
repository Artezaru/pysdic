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
from typing import Dict, Optional, Sequence, Union
from numbers import Integral
from numpy.typing import ArrayLike

import numpy
import scipy.sparse
import scipy.sparse.csgraph

from .._validation import _as_array, _as_integer_array, _validate_bool, _validate_integer
from ._validation import _validate_n_vertices, _valid_points_mask


# =============================================================================
# Internal helpers
# =============================================================================
def _as_graph(graph: Union[ArrayLike, scipy.sparse.spmatrix]) -> scipy.sparse.csr_matrix:
    r"""
    Convert an adjacency graph (sparse or dense square matrix) to a boolean
    ``scipy.sparse.csr_matrix`` with sorted indices.

    .. important::

        Internal use only.

    """
    if scipy.sparse.issparse(graph):
        graph = scipy.sparse.csr_matrix(graph)
    else:
        # Boolean dtype: same meaning as the conversion below, without a float copy of a
        # dense (N, N) matrix
        graph = _as_array(
            graph, "graph", numpy.bool_, shape=(None, None), shape_hint="(N, N)", finite=False
        )
        graph = scipy.sparse.csr_matrix(graph)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(
            f"Input 'graph' must be a square matrix of shape (N, N). "
            f"Got shape {graph.shape}."
        )
    graph = graph.astype(bool)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _validate_start(start: Integral, n_nodes: int) -> int:
    r"""
    Check that ``start`` is a valid node index of a graph with ``n_nodes`` nodes.

    .. important::

        Internal use only.

    """
    start = _validate_integer(start, "start")
    if not 0 <= start < n_nodes:
        raise IndexError(
            f"Input 'start' must be an integer between 0 and {n_nodes - 1}. Got {start}."
        )
    return start


def _reachability(graph: scipy.sparse.csr_matrix, max_distance: int) -> scipy.sparse.csr_matrix:
    r"""
    Compute the boolean reachability matrix :math:`(I + A)^{d}` of a graph, i.e. the
    nodes reachable from each node in at most :math:`d` steps (the diagonal included).

    .. important::

        Internal use only. ``graph`` is a validated boolean csr matrix.

    """
    n_nodes = graph.shape[0]
    one_step = (
        scipy.sparse.identity(n_nodes, dtype=bool, format="csr") + graph
    ).tocsr()  # (I + A), boolean
    reachable = scipy.sparse.identity(n_nodes, dtype=bool, format="csr")
    for _ in range(max_distance):
        new_reachable = (reachable @ one_step).tocsr()  # Boolean product = logical OR
        if new_reachable.nnz == reachable.nnz:
            break  # Converged: all reachable nodes already found
        reachable = new_reachable
    reachable.sort_indices()
    return reachable


# =============================================================================
# Single source searches
# =============================================================================
def bfs_distance(
    graph: Union[ArrayLike, scipy.sparse.spmatrix],
    start: Integral,
    *,
    max_distance: Optional[Integral] = None,
) -> numpy.ndarray:
    r"""
    Compute the shortest path distances from a starting node to all the nodes of an
    adjacency graph.

    .. note::

        The search is performed with ``scipy.sparse.csgraph.dijkstra`` on the unweighted
        graph, which is equivalent to a breadth-first search (BFS).

    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph
        is not symmetric, ...). The behavior of the function is undefined if the input
        graph is not valid.


    Parameters
    ----------
    graph: Union[ArrayLike, scipy.sparse.spmatrix]
        The adjacency graph as a square matrix of shape :math:`(N, N)` (sparse or
        dense), where a non-zero entry at :math:`(i, j)` means that nodes :math:`i`
        and :math:`j` are neighbors. See
        :func:`pysdic.fem.assemble_vertices_adjacency_graph` and
        :func:`pysdic.fem.assemble_elements_adjacency_graph`.

    start: Integral
        The index of the starting node.

    max_distance: Optional[Integral], optional (default: ``None``)
        The maximum distance of the search. If None, all the reachable nodes are
        considered. Otherwise, the nodes farther than ``max_distance`` get a distance
        of -1.


    Returns
    -------
    distances: numpy.ndarray
        An integer array of shape :math:`(N,)` containing the shortest path distances
        from ``start`` to all the nodes of the graph (-1 for unreachable nodes).


    Notes
    -----
    The distance from node :math:`A` to node :math:`T` is defined as follows:

    - :math:`\text{dist}(A, T) = 0` if node :math:`T` is the same as node :math:`A`.
    - :math:`\text{dist}(A, T) = 1` if node :math:`T` is a neighbor of node :math:`A`.
    - :math:`\text{dist}(A, T) = d \geq 1` if the shortest path from node :math:`A`
      to node :math:`T` has length :math:`d`.
    - :math:`\text{dist}(A, T) = -1` if node :math:`T` is not reachable from node
      :math:`A`, or not reachable within the specified ``max_distance``.


    See Also
    --------
    pysdic.fem.bfs_neighborhood:
        Neighborhood of a single node within a given distance.

    pysdic.fem.compute_distance_matrix:
        Distances between all the pairs of nodes of a graph.


    Examples
    --------
    Create a simple disconnected graph of 7 nodes and compute the shortest path
    distances from node 0 to all the other nodes.

    .. figure:: /_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%

       Disconnected graph with 7 nodes for BFS example. Distance from node 0 to
       node 4 is 3 (0 -> 2 -> 3 -> 4).

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import bfs_distance

        edges = numpy.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [5, 6]])
        graph = numpy.zeros((7, 7), dtype=bool)
        graph[edges[:, 0], edges[:, 1]] = True
        graph[edges[:, 1], edges[:, 0]] = True

        distances = bfs_distance(graph, start=0)
        print(distances)

    .. code-block:: console

        [ 0  1  1  2  3 -1 -1]

    """
    graph = _as_graph(graph)
    start = _validate_start(start, graph.shape[0])
    max_distance = _validate_integer(
        max_distance, "max_distance", min_value=0, allow_none=True
    )

    distances = scipy.sparse.csgraph.dijkstra(
        graph,
        unweighted=True,
        indices=start,
        limit=numpy.inf if max_distance is None else float(max_distance),
    )  # (N,), inf for unreachable nodes
    return numpy.where(numpy.isinf(distances), -1, distances).astype(numpy.int64)


def bfs_neighborhood(
    graph: Union[ArrayLike, scipy.sparse.spmatrix],
    start: Integral,
    *,
    max_distance: Integral = 1,
    self_neighborhood: bool = True,
) -> numpy.ndarray:
    r"""
    Compute the neighborhood of a starting node in an adjacency graph, i.e. the nodes
    within a given distance from the starting node.

    .. note::

        The search is performed with ``scipy.sparse.csgraph.dijkstra`` on the unweighted
        graph, which is equivalent to a breadth-first search (BFS).

    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph
        is not symmetric, ...). The behavior of the function is undefined if the input
        graph is not valid.


    Parameters
    ----------
    graph: Union[ArrayLike, scipy.sparse.spmatrix]
        The adjacency graph as a square matrix of shape :math:`(N, N)` (sparse or
        dense), where a non-zero entry at :math:`(i, j)` means that nodes :math:`i`
        and :math:`j` are neighbors.

    start: Integral
        The index of the starting node.

    max_distance: Integral, optional (default: ``1``)
        The maximum distance from ``start`` of the nodes in the neighborhood.

    self_neighborhood: bool, optional (default: ``True``)
        If True, the neighborhood includes the starting node itself.


    Returns
    -------
    neighborhood: numpy.ndarray
        A sorted integer array containing the indices of the nodes in the neighborhood.


    See Also
    --------
    pysdic.fem.bfs_distance:
        Distances from a single node to all the nodes of a graph.

    pysdic.fem.compute_neighborhood:
        Neighborhoods of all the nodes of a graph.


    Examples
    --------
    Create a simple disconnected graph of 7 nodes and compute the neighborhood of
    node 0 within a distance of 2.

    .. figure:: /_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%

       Disconnected graph with 7 nodes for BFS example. Nodes 0, 1, 2 and 3 are
       the only nodes in the neighborhood of node 0 for max_distance = 2.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import bfs_neighborhood

        edges = numpy.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [5, 6]])
        graph = numpy.zeros((7, 7), dtype=bool)
        graph[edges[:, 0], edges[:, 1]] = True
        graph[edges[:, 1], edges[:, 0]] = True

        neighborhood = bfs_neighborhood(graph, start=0, max_distance=2)
        print(neighborhood)

    .. code-block:: console

        [0 1 2 3]

    """
    max_distance = _validate_integer(max_distance, "max_distance", min_value=0)
    self_neighborhood = _validate_bool(self_neighborhood, "self_neighborhood")

    distances = bfs_distance(graph, start, max_distance=max_distance)
    neighborhood = numpy.flatnonzero(distances >= 0)
    if not self_neighborhood:
        neighborhood = neighborhood[neighborhood != start]
    return neighborhood


# =============================================================================
# Adjacency graphs
# =============================================================================
def assemble_vertices_adjacency_graph(
    connectivity_array: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
) -> scipy.sparse.csr_matrix:
    r"""
    Assemble the vertices adjacency graph of a mesh, where two vertices are neighbors
    if they belong to the same element.

    The function is independent of the geometry of the elements: all the vertices of an
    element are neighbors of each other (e.g. the diagonal vertices of a quadrangle,
    or the corner and mid-edge vertices of a 6-node triangle).

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        No tests are performed to check if the input connectivity is valid
        (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined in that case.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it
        will be inferred as the maximum vertex index in ``connectivity_array`` plus one.


    Returns
    -------
    graph: scipy.sparse.csr_matrix
        A symmetric boolean sparse matrix of shape :math:`(N_{v}, N_{v})` with an empty
        diagonal, where ``graph[i, j]`` is True if vertices :math:`i` and :math:`j`
        are neighbors. The neighbors of vertex :math:`i` are ``graph[i].indices``.


    See Also
    --------
    pysdic.fem.assemble_elements_adjacency_graph:
        Elements adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and assemble
    the vertices adjacency graph.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh. The neighboring vertices of vertex 1 are vertices 0, 2,
        4, 5 and 6 since they are part of the same elements in the mesh
        (elements 0 and 1).

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import assemble_vertices_adjacency_graph

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        graph = assemble_vertices_adjacency_graph(connectivity_array)

        print(f"vertices adjacency graph (shape={graph.shape})")
        print(f"neighbors of vertex 1: {graph[1].indices}")

    .. code-block:: console

        vertices adjacency graph (shape=(10, 10))
        neighbors of vertex 1: [0 2 4 5 6]

    """
    connectivity_array = _as_integer_array(
        connectivity_array, "connectivity_array", shape=(None, None), shape_hint="(N_e, N_vpe)"
    )
    n_vertices = _validate_n_vertices(n_vertices, connectivity_array)

    # All the pairs (u, v) of vertices of each element, in both directions
    N_vpe = connectivity_array.shape[1]
    first, second = numpy.triu_indices(N_vpe, k=1)
    u = connectivity_array[:, first].ravel()
    v = connectivity_array[:, second].ravel()
    rows = numpy.concatenate([u, v])
    columns = numpy.concatenate([v, u])
    not_loop = rows != columns  # Degenerated elements with repeated vertices

    graph = scipy.sparse.coo_matrix(
        (numpy.ones(numpy.count_nonzero(not_loop), dtype=bool),
         (rows[not_loop], columns[not_loop])),
        shape=(n_vertices, n_vertices),
    ).tocsr()  # Duplicates are merged (logical OR)
    graph.sort_indices()
    return graph


def assemble_elements_adjacency_graph(
    connectivity_array: ArrayLike,
) -> scipy.sparse.csr_matrix:
    r"""
    Assemble the elements adjacency graph of a mesh, where two elements are neighbors
    if they share at least one vertex.

    The function is independent of the geometry of the elements: two elements sharing
    only a corner vertex are neighbors.

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        No tests are performed to check if the input connectivity is valid
        (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined in that case.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.


    Returns
    -------
    graph: scipy.sparse.csr_matrix
        A symmetric boolean sparse matrix of shape :math:`(N_{e}, N_{e})` with an empty
        diagonal, where ``graph[i, j]`` is True if elements :math:`i` and :math:`j`
        are neighbors. The neighbors of element :math:`i` are ``graph[i].indices``.


    Notes
    -----
    Let :math:`B \in \{0, 1\}^{N_{e} \times N_{v}}` be the incidence matrix of the mesh
    (:math:`B_{e,v} = 1` if vertex :math:`v` belongs to element :math:`e`). The
    elements :math:`i \neq j` are neighbors if :math:`(B B^{T})_{i,j} > 0`.


    See Also
    --------
    pysdic.fem.assemble_vertices_adjacency_graph:
        Vertices adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and assemble
    the elements adjacency graph.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh. The neighboring elements of element 3 are elements 1
        and 2 since they share at least one vertex in the mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import assemble_elements_adjacency_graph

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        graph = assemble_elements_adjacency_graph(connectivity_array)

        print(f"elements adjacency graph (shape={graph.shape})")
        print(f"neighbors of element 3: {graph[3].indices}")

    .. code-block:: console

        elements adjacency graph (shape=(4, 4))
        neighbors of element 3: [1 2]

    """
    connectivity_array = _as_integer_array(
        connectivity_array, "connectivity_array", shape=(None, None), shape_hint="(N_e, N_vpe)"
    )
    N_e, N_vpe = connectivity_array.shape
    n_vertices = _validate_n_vertices(None, connectivity_array)

    # Incidence matrix B (N_e, N_v)
    incidence = scipy.sparse.coo_matrix(
        (numpy.ones(N_e * N_vpe, dtype=bool),
         (numpy.repeat(numpy.arange(N_e), N_vpe), connectivity_array.ravel())),
        shape=(N_e, n_vertices),
    ).tocsr()

    # Elements sharing a vertex: B B^T without the diagonal
    shared = (incidence @ incidence.T).tocoo()
    not_loop = shared.row != shared.col
    graph = scipy.sparse.coo_matrix(
        (shared.data[not_loop], (shared.row[not_loop], shared.col[not_loop])),
        shape=(N_e, N_e),
    ).tocsr()
    graph.sort_indices()
    return graph


# =============================================================================
# Distance matrices
# =============================================================================
def compute_distance_matrix(
    graph: Union[ArrayLike, scipy.sparse.spmatrix],
    *,
    max_distance: Optional[Integral] = None,
    sparse: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Compute the shortest path distances between all the pairs of nodes of an adjacency
    graph.

    .. warning::

        The dense output has shape :math:`(N, N)` and its memory grows quadratically
        with the number of nodes (about 800 MB for :math:`N = 10^4`). For large graphs,
        use ``sparse=True`` with a small ``max_distance``.

    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph
        is not symmetric, ...). The behavior of the function is undefined if the input
        graph is not valid.


    Parameters
    ----------
    graph: Union[ArrayLike, scipy.sparse.spmatrix]
        The adjacency graph as a square matrix of shape :math:`(N, N)` (sparse or
        dense), where a non-zero entry at :math:`(i, j)` means that nodes :math:`i`
        and :math:`j` are neighbors.

    max_distance: Optional[Integral], optional (default: ``None``)
        The maximum distance to compute. Pairs of nodes farther than ``max_distance``
        are considered unreachable. If None, all the distances are computed.

    sparse: bool, optional (default: ``False``)
        If True, a sparse matrix storing only the distances between distinct reachable
        nodes is returned (see below).


    Returns
    -------
    distance_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        A symmetric matrix of shape :math:`(N, N)` where the entry :math:`(i, j)` is the
        distance between nodes :math:`i` and :math:`j`.

        - Dense output: an integer array with 0 on the diagonal and -1 for unreachable
          pairs.
        - Sparse output: an integer ``csr_matrix`` storing only the distances
          :math:`\geq 1`. The diagonal (distance 0) and the unreachable pairs are not
          stored, so ``distance_matrix[i].indices`` are the nodes reachable from
          :math:`i` (excluding :math:`i`).


    Notes
    -----
    The distance between nodes :math:`i` and :math:`j` is the number of edges of the
    shortest path between them (see :func:`pysdic.fem.bfs_distance`).

    The dense output is computed with ``scipy.sparse.csgraph.dijkstra`` on the unweighted
    graph. The sparse output is computed level by level with boolean sparse products:
    the nodes at distance :math:`k` are the nodes reachable in :math:`k` steps
    (:math:`(I + A)^{k}`) but not in :math:`k - 1` steps, so the memory only depends
    on the size of the neighborhoods.


    See Also
    --------
    pysdic.fem.bfs_distance:
        Distances from a single node to all the nodes of a graph.

    pysdic.fem.compute_vertices_distance_matrix:
        Distance matrix of the vertices of a mesh.

    pysdic.fem.compute_elements_distance_matrix:
        Distance matrix of the elements of a mesh.


    Examples
    --------
    Create a simple disconnected graph of 7 nodes and compute the distance matrix.

    .. figure:: /_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%

       Disconnected graph with 7 nodes for distance matrix example.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_distance_matrix

        edges = numpy.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [5, 6]])
        graph = numpy.zeros((7, 7), dtype=bool)
        graph[edges[:, 0], edges[:, 1]] = True
        graph[edges[:, 1], edges[:, 0]] = True

        distance_matrix = compute_distance_matrix(graph)
        print(f"distance matrix (shape={distance_matrix.shape}):")
        print(distance_matrix)

        sparse_distance_matrix = compute_distance_matrix(graph, max_distance=2, sparse=True)
        print(f"nodes within a distance of 2 from node 0: {sparse_distance_matrix[0].indices}")
        print(f"their distances: {sparse_distance_matrix[0].data}")

    .. code-block:: console

        distance matrix (shape=(7, 7)):
        [[ 0  1  1  2  3 -1 -1]
         [ 1  0  1  2  3 -1 -1]
         [ 1  1  0  1  2 -1 -1]
         [ 2  2  1  0  1 -1 -1]
         [ 3  3  2  1  0 -1 -1]
         [-1 -1 -1 -1 -1  0  1]
         [-1 -1 -1 -1 -1  1  0]]
        nodes within a distance of 2 from node 0: [1 2 3]
        their distances: [1 1 2]

    """
    graph = _as_graph(graph)
    max_distance = _validate_integer(
        max_distance, "max_distance", min_value=0, allow_none=True
    )
    sparse = _validate_bool(sparse, "sparse")
    n_nodes = graph.shape[0]

    # Dense output with csgraph
    if not sparse:
        distances = scipy.sparse.csgraph.dijkstra(
            graph,
            unweighted=True,
            limit=numpy.inf if max_distance is None else float(max_distance),
        )  # (N, N), inf for unreachable pairs
        return numpy.where(numpy.isinf(distances), -1, distances).astype(numpy.int64)

    # Sparse output, level by level
    one_step = (
        scipy.sparse.identity(n_nodes, dtype=bool, format="csr") + graph
    ).tocsr()
    reachable = scipy.sparse.identity(n_nodes, dtype=bool, format="csr")
    distance_matrix = scipy.sparse.csr_matrix((n_nodes, n_nodes), dtype=numpy.int64)
    level = 0
    while max_distance is None or level < max_distance:
        level += 1
        new_reachable = (reachable @ one_step).tocsr()
        if new_reachable.nnz == reachable.nnz:
            break  # Converged: all reachable nodes already found
        new_level = (new_reachable.astype(numpy.int64) - reachable.astype(numpy.int64))
        distance_matrix = distance_matrix + level * new_level  # Entries at distance level
        reachable = new_reachable

    distance_matrix = distance_matrix.tocsr()
    distance_matrix.eliminate_zeros()
    distance_matrix.sort_indices()
    return distance_matrix


def compute_vertices_distance_matrix(
    connectivity_array: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
    max_distance: Optional[Integral] = None,
    sparse: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Compute the shortest path distances between all the pairs of vertices of a mesh,
    where two vertices are neighbors (distance 1) if they belong to the same element.

    Convenience function that combines :func:`pysdic.fem.assemble_vertices_adjacency_graph`
    and :func:`pysdic.fem.compute_distance_matrix`.

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        The dense output has shape :math:`(N_{v}, N_{v})`. For large meshes, use
        ``sparse=True`` with a small ``max_distance``.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it
        will be inferred as the maximum vertex index in ``connectivity_array`` plus one.

    max_distance: Optional[Integral], optional (default: ``None``)
        The maximum distance to compute. Pairs of vertices farther than
        ``max_distance`` are considered unreachable. If None, all the distances are
        computed.

    sparse: bool, optional (default: ``False``)
        If True, a sparse matrix storing only the distances between distinct reachable
        vertices is returned (see :func:`pysdic.fem.compute_distance_matrix`).


    Returns
    -------
    distance_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        A symmetric matrix of shape :math:`(N_{v}, N_{v})` containing the distances
        between the vertices (see :func:`pysdic.fem.compute_distance_matrix` for the
        dense and sparse conventions).


    See Also
    --------
    pysdic.fem.compute_distance_matrix:
        Distance matrix of an adjacency graph.

    pysdic.fem.assemble_vertices_adjacency_graph:
        Vertices adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and compute
    the vertices distance matrix.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_vertices_distance_matrix

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        distance_matrix = compute_vertices_distance_matrix(connectivity_array)

        print(f"vertices distance matrix (shape={distance_matrix.shape})")
        print(f"distances from vertex 1: {distance_matrix[1]}")

    .. code-block:: console

        vertices distance matrix (shape=(10, 10))
        distances from vertex 1: [1 0 1 2 1 1 1 2 2 2]

    """
    graph = assemble_vertices_adjacency_graph(connectivity_array, n_vertices=n_vertices)
    return compute_distance_matrix(graph, max_distance=max_distance, sparse=sparse)


def compute_elements_distance_matrix(
    connectivity_array: ArrayLike,
    *,
    max_distance: Optional[Integral] = None,
    sparse: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Compute the shortest path distances between all the pairs of elements of a mesh,
    where two elements are neighbors (distance 1) if they share at least one vertex.

    Convenience function that combines :func:`pysdic.fem.assemble_elements_adjacency_graph`
    and :func:`pysdic.fem.compute_distance_matrix`.

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        The dense output has shape :math:`(N_{e}, N_{e})`. For large meshes, use
        ``sparse=True`` with a small ``max_distance``.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    max_distance: Optional[Integral], optional (default: ``None``)
        The maximum distance to compute. Pairs of elements farther than
        ``max_distance`` are considered unreachable. If None, all the distances are
        computed.

    sparse: bool, optional (default: ``False``)
        If True, a sparse matrix storing only the distances between distinct reachable
        elements is returned (see :func:`pysdic.fem.compute_distance_matrix`).


    Returns
    -------
    distance_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        A symmetric matrix of shape :math:`(N_{e}, N_{e})` containing the distances
        between the elements (see :func:`pysdic.fem.compute_distance_matrix` for the
        dense and sparse conventions).


    See Also
    --------
    pysdic.fem.compute_distance_matrix:
        Distance matrix of an adjacency graph.

    pysdic.fem.assemble_elements_adjacency_graph:
        Elements adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and compute
    the elements distance matrix.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_elements_distance_matrix

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        distance_matrix = compute_elements_distance_matrix(connectivity_array)

        print(f"elements distance matrix (shape={distance_matrix.shape}):")
        print(distance_matrix)

    .. code-block:: console

        elements distance matrix (shape=(4, 4)):
        [[0 1 2 2]
         [1 0 1 1]
         [2 1 0 1]
         [2 1 1 0]]

    """
    graph = assemble_elements_adjacency_graph(connectivity_array)
    return compute_distance_matrix(graph, max_distance=max_distance, sparse=sparse)


# =============================================================================
# Neighborhoods
# =============================================================================
def compute_neighborhood(
    graph: Union[ArrayLike, scipy.sparse.spmatrix],
    *,
    max_distance: Integral = 1,
    self_neighborhood: bool = True,
) -> scipy.sparse.csr_matrix:
    r"""
    Compute the neighborhood of each node of an adjacency graph, i.e. the nodes within
    a given distance from each node.

    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph
        is not symmetric, ...). The behavior of the function is undefined if the input
        graph is not valid.


    Parameters
    ----------
    graph: Union[ArrayLike, scipy.sparse.spmatrix]
        The adjacency graph as a square matrix of shape :math:`(N, N)` (sparse or
        dense), where a non-zero entry at :math:`(i, j)` means that nodes :math:`i`
        and :math:`j` are neighbors.

    max_distance: Integral, optional (default: ``1``)
        The maximum distance of the nodes in the neighborhood. With 1, only the directly
        adjacent nodes are included.

    self_neighborhood: bool, optional (default: ``True``)
        If True, the neighborhood of each node includes the node itself.


    Returns
    -------
    neighborhood: scipy.sparse.csr_matrix
        A symmetric boolean sparse matrix of shape :math:`(N, N)`, where
        ``neighborhood[i, j]`` is True if node :math:`j` is in the neighborhood of node
        :math:`i`. The neighborhood of node :math:`i` is ``neighborhood[i].indices``
        (sorted).


    Notes
    -----
    The neighborhood of node :math:`i` contains all the nodes :math:`j` such that
    :math:`\text{dist}(i, j) \leq d` with :math:`d` the ``max_distance``. It is
    computed with boolean sparse products as the reachability matrix
    :math:`(I + A)^{d}`, where :math:`A` is the adjacency matrix of the graph.

    - For :math:`d = 0`, the neighborhood of each node is the node itself (or empty if
      ``self_neighborhood`` is False).
    - For :math:`d = 1`, the neighborhood is :math:`I + A` (or :math:`A` if
      ``self_neighborhood`` is False).


    See Also
    --------
    pysdic.fem.bfs_neighborhood:
        Neighborhood of a single node.

    pysdic.fem.compute_vertices_neighborhood:
        Neighborhood of each vertex of a mesh.

    pysdic.fem.compute_elements_neighborhood:
        Neighborhood of each element of a mesh.


    Examples
    --------
    Create a simple disconnected graph of 7 nodes and compute the neighborhood of
    each node within a distance of 2.

    .. figure:: /_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%

       Disconnected graph with 7 nodes for BFS example. Neighborhood of node 0
       with max_distance=2 includes nodes 0, 1, 2 and 3.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_neighborhood

        edges = numpy.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [5, 6]])
        graph = numpy.zeros((7, 7), dtype=bool)
        graph[edges[:, 0], edges[:, 1]] = True
        graph[edges[:, 1], edges[:, 0]] = True

        neighborhood = compute_neighborhood(graph, max_distance=2)

        print(f"neighborhood (shape={neighborhood.shape})")
        print(f"neighborhood of node 0: {neighborhood[0].indices}")

    .. code-block:: console

        neighborhood (shape=(7, 7))
        neighborhood of node 0: [0 1 2 3]

    """
    graph = _as_graph(graph)
    max_distance = _validate_integer(max_distance, "max_distance", min_value=0)
    self_neighborhood = _validate_bool(self_neighborhood, "self_neighborhood")

    neighborhood = _reachability(graph, max_distance)  # (N, N), diagonal included
    if not self_neighborhood:
        neighborhood = neighborhood.tocoo()
        not_self = neighborhood.row != neighborhood.col
        neighborhood = scipy.sparse.coo_matrix(
            (neighborhood.data[not_self],
             (neighborhood.row[not_self], neighborhood.col[not_self])),
            shape=neighborhood.shape,
        ).tocsr()
        neighborhood.sort_indices()
    return neighborhood


def compute_vertices_neighborhood(
    connectivity_array: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
    max_distance: Integral = 1,
    self_neighborhood: bool = True,
) -> scipy.sparse.csr_matrix:
    r"""
    Compute the neighborhood of each vertex of a mesh, i.e. the vertices within a given
    distance, where two vertices are neighbors (distance 1) if they belong to the same
    element.

    Convenience function that combines :func:`pysdic.fem.assemble_vertices_adjacency_graph`
    and :func:`pysdic.fem.compute_neighborhood`.

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        No tests are performed to check if the input connectivity is valid
        (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined in that case.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it
        will be inferred as the maximum vertex index in ``connectivity_array`` plus one.

    max_distance: Integral, optional (default: ``1``)
        The maximum distance of the vertices in the neighborhood. With 1, only the
        vertices sharing an element are included.

    self_neighborhood: bool, optional (default: ``True``)
        If True, the neighborhood of each vertex includes the vertex itself.


    Returns
    -------
    neighborhood: scipy.sparse.csr_matrix
        A symmetric boolean sparse matrix of shape :math:`(N_{v}, N_{v})`, where the
        neighborhood of vertex :math:`i` is ``neighborhood[i].indices`` (sorted).


    See Also
    --------
    pysdic.fem.compute_neighborhood:
        Neighborhood of each node of an adjacency graph.

    pysdic.fem.assemble_vertices_adjacency_graph:
        Vertices adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and compute
    the neighborhood of each vertex.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh. The neighborhood of vertex 1 with max_distance=1
        contains vertices 0, 1, 2, 4, 5 and 6.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_vertices_neighborhood

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        neighborhood = compute_vertices_neighborhood(connectivity_array, max_distance=1)

        print(f"vertices neighborhood (shape={neighborhood.shape})")
        print(f"neighborhood of vertex 1: {neighborhood[1].indices}")

    .. code-block:: console

        vertices neighborhood (shape=(10, 10))
        neighborhood of vertex 1: [0 1 2 4 5 6]

    """
    graph = assemble_vertices_adjacency_graph(connectivity_array, n_vertices=n_vertices)
    return compute_neighborhood(
        graph, max_distance=max_distance, self_neighborhood=self_neighborhood
    )


def compute_elements_neighborhood(
    connectivity_array: ArrayLike,
    *,
    max_distance: Integral = 1,
    self_neighborhood: bool = True,
) -> scipy.sparse.csr_matrix:
    r"""
    Compute the neighborhood of each element of a mesh, i.e. the elements within a
    given distance, where two elements are neighbors (distance 1) if they share at
    least one vertex.

    Convenience function that combines :func:`pysdic.fem.assemble_elements_adjacency_graph`
    and :func:`pysdic.fem.compute_neighborhood`.

    .. note::

        - Input ``connectivity_array`` will be converted to ``numpy.int64``.

    .. warning::

        No tests are performed to check if the input connectivity is valid
        (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined in that case.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    max_distance: Integral, optional (default: ``1``)
        The maximum distance of the elements in the neighborhood. With 1, only the
        elements sharing a vertex are included.

    self_neighborhood: bool, optional (default: ``True``)
        If True, the neighborhood of each element includes the element itself.


    Returns
    -------
    neighborhood: scipy.sparse.csr_matrix
        A symmetric boolean sparse matrix of shape :math:`(N_{e}, N_{e})`, where the
        neighborhood of element :math:`i` is ``neighborhood[i].indices`` (sorted).


    See Also
    --------
    pysdic.fem.compute_neighborhood:
        Neighborhood of each node of an adjacency graph.

    pysdic.fem.assemble_elements_adjacency_graph:
        Elements adjacency graph of a mesh.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements and compute
    the neighborhood of each element.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh. The neighborhood of element 0 with max_distance=1
        contains elements 0 and 1.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_elements_neighborhood

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        neighborhood = compute_elements_neighborhood(connectivity_array, max_distance=1)

        print(f"elements neighborhood (shape={neighborhood.shape})")
        print(f"neighborhood of element 0: {neighborhood[0].indices}")

    .. code-block:: console

        elements neighborhood (shape=(4, 4))
        neighborhood of element 0: [0 1]

    """
    graph = assemble_elements_adjacency_graph(connectivity_array)
    return compute_neighborhood(
        graph, max_distance=max_distance, self_neighborhood=self_neighborhood
    )


# =============================================================================
# Integration points neighborhoods
# =============================================================================
def remap_integration_points_neighborhood(
    elements_neighborhood: Union[ArrayLike, scipy.sparse.spmatrix],
    element_indices: ArrayLike,
    *,
    skip_m1: bool = True,
) -> scipy.sparse.csr_matrix:
    r"""
    Remap a neighborhood of the elements of a mesh to the integration points: for each
    element, get the integration points located in the elements of its neighborhood.

    .. note::

        - Input ``element_indices`` will be converted to ``numpy.int64``.

    .. important::

        With ``skip_m1=True``, the points with element index ``-1`` (outside the mesh)
        belong to no neighborhood. With ``skip_m1=False``, negative element indices raise
        an ``IndexError``.


    Parameters
    ----------
    elements_neighborhood: Union[ArrayLike, scipy.sparse.spmatrix]
        The neighborhood of the elements as a square matrix of shape
        :math:`(N_{e}, N_{e})` (sparse or dense), where a non-zero entry at
        :math:`(i, j)` means that element :math:`j` is in the neighborhood of element
        :math:`i` (see :func:`pysdic.fem.compute_elements_neighborhood`).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of the element of each
        of the :math:`N_{p}` integration points.

    skip_m1: bool, optional (default: ``True``)
        If set to True, the points with element index ``-1`` are ignored.


    Returns
    -------
    points_neighborhood: scipy.sparse.csr_matrix
        A boolean sparse matrix of shape :math:`(N_{e}, N_{p})`, where the integration
        points in the neighborhood of element :math:`i` are
        ``points_neighborhood[i].indices`` (sorted).


    Notes
    -----
    Let :math:`\mathcal{V} \in \{0, 1\}^{N_{e} \times N_{e}}` be the elements
    neighborhood and :math:`C \in \{0, 1\}^{N_{e} \times N_{p}}` the matrix such that
    :math:`C_{e,p} = 1` if point :math:`p` is located in element :math:`e`. The points
    neighborhood is the boolean product :math:`\mathcal{V} C`.


    See Also
    --------
    pysdic.fem.compute_elements_neighborhood:
        Neighborhood of each element of a mesh.

    pysdic.fem.compute_integration_points_neighborhood:
        Neighborhood of integration points directly from the mesh connectivity.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements, with two
    integration points per element and one point outside the mesh, and remap the
    neighborhood of the elements to the integration points.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            compute_elements_neighborhood,
            remap_integration_points_neighborhood,
        )

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        element_indices = numpy.array([0, 0, 1, 1, 2, 2, 3, 3, -1])

        elements_neighborhood = compute_elements_neighborhood(connectivity_array)
        points_neighborhood = remap_integration_points_neighborhood(
            elements_neighborhood, element_indices
        )

        print(f"points neighborhood (shape={points_neighborhood.shape})")
        for element in range(points_neighborhood.shape[0]):
            print(f"element {element}: points {points_neighborhood[element].indices}")

    .. code-block:: console

        points neighborhood (shape=(4, 9))
        element 0: points [0 1 2 3]
        element 1: points [0 1 2 3 4 5 6 7]
        element 2: points [2 3 4 5 6 7]
        element 3: points [2 3 4 5 6 7]

    """
    elements_neighborhood = _as_graph(elements_neighborhood)
    element_indices = _as_integer_array(
        element_indices, "element_indices", shape=(None,), shape_hint="(N_p,)"
    )
    skip_m1 = _validate_bool(skip_m1, "skip_m1")

    N_e = elements_neighborhood.shape[0]
    N_p = element_indices.shape[0]
    valid_mask = _valid_points_mask(element_indices, skip_m1)
    if numpy.any(element_indices[valid_mask] >= N_e):
        raise IndexError(
            f"Input 'element_indices' contains indices greater than the number of "
            f"elements ({N_e}) of 'elements_neighborhood'."
        )

    # C[e, p] = True if point p is located in element e
    points = numpy.flatnonzero(valid_mask)
    location = scipy.sparse.coo_matrix(
        (numpy.ones(points.size, dtype=bool), (element_indices[points], points)),
        shape=(N_e, N_p),
    ).tocsr()

    points_neighborhood = (elements_neighborhood @ location).tocsr()  # (N_e, N_p)
    points_neighborhood.sort_indices()
    return points_neighborhood


def compute_integration_points_neighborhood(
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    *,
    max_distance: Integral = 1,
    self_neighborhood: bool = True,
    skip_m1: bool = True,
) -> scipy.sparse.csr_matrix:
    r"""
    Compute, for each element of a mesh, the integration points located in the elements
    of its neighborhood.

    Convenience function that combines :func:`pysdic.fem.compute_elements_neighborhood`
    and :func:`pysdic.fem.remap_integration_points_neighborhood`.

    .. note::

        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. important::

        With ``skip_m1=True``, the points with element index ``-1`` (outside the mesh)
        belong to no neighborhood. With ``skip_m1=False``, negative element indices raise
        an ``IndexError``.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of the element of each
        of the :math:`N_{p}` integration points.

    max_distance: Integral, optional (default: ``1``)
        The maximum distance of the elements in the neighborhood. With 1, only the
        elements sharing a vertex are included.

    self_neighborhood: bool, optional (default: ``True``)
        If True, the neighborhood of each element includes the element itself (and so
        its own integration points).

    skip_m1: bool, optional (default: ``True``)
        If set to True, the points with element index ``-1`` are ignored.


    Returns
    -------
    points_neighborhood: scipy.sparse.csr_matrix
        A boolean sparse matrix of shape :math:`(N_{e}, N_{p})`, where the integration
        points in the neighborhood of element :math:`i` are
        ``points_neighborhood[i].indices`` (sorted).


    See Also
    --------
    pysdic.fem.compute_elements_neighborhood:
        Neighborhood of each element of a mesh.

    pysdic.fem.remap_integration_points_neighborhood:
        Remap a neighborhood of the elements to the integration points.


    Examples
    --------
    Create a simple ``'quadrangle_4'`` mesh with 10 vertices and 4 elements, with two
    integration points per element, and compute the integration points in the
    neighborhood of each element.

    .. figure:: /_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%

        ``'quadrangle_4'`` mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_integration_points_neighborhood

        connectivity_array = numpy.array([
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [2, 3, 6, 7],
            [6, 7, 8, 9]])

        element_indices = numpy.array([0, 0, 1, 1, 2, 2, 3, 3])

        points_neighborhood = compute_integration_points_neighborhood(
            connectivity_array, element_indices, max_distance=1
        )

        print(f"points neighborhood (shape={points_neighborhood.shape})")
        for element in range(points_neighborhood.shape[0]):
            print(f"element {element}: points {points_neighborhood[element].indices}")

    .. code-block:: console

        points neighborhood (shape=(4, 8))
        element 0: points [0 1 2 3]
        element 1: points [0 1 2 3 4 5 6 7]
        element 2: points [2 3 4 5 6 7]
        element 3: points [2 3 4 5 6 7]

    """
    elements_neighborhood = compute_elements_neighborhood(
        connectivity_array,
        max_distance=max_distance,
        self_neighborhood=self_neighborhood,
    )
    return remap_integration_points_neighborhood(
        elements_neighborhood, element_indices, skip_m1=skip_m1
    )

# =============================================================================
# Neighborhood statistics
# =============================================================================
_NEIGHBORHOOD_STATISTICS = ("mean", "median", "min", "max", "std", "var", "Q1", "Q3", "IQR")


def compute_neighborhood_statistics(
    values: ArrayLike,
    neighborhood: Union[ArrayLike, scipy.sparse.spmatrix],
    statistics: Union[str, Sequence[str]],
) -> Dict[str, numpy.ndarray]:
    r"""
    Compute statistics of values over the neighborhood of each node of a graph (vertices,
    elements, ...).

    The neighborhood is a matrix of shape :math:`(N, M)` whose row :math:`i` lists the
    indices of the values used for the node :math:`i` (its non-zero columns), as returned by
    :func:`pysdic.fem.compute_neighborhood`, :func:`pysdic.fem.compute_vertices_neighborhood`
    or :func:`pysdic.fem.compute_elements_neighborhood` (:math:`M = N`), or by
    :func:`pysdic.fem.compute_integration_points_neighborhood` (values defined at :math:`M`
    integration points, statistics computed for each element).

    Implemented statistics:

    +------------+-------------------------------------------------------------------+
    | Statistic  | Description                                                       |
    +============+===================================================================+
    | ``mean``   | Mean of the values of the neighborhood.                           |
    +------------+-------------------------------------------------------------------+
    | ``median`` | Median of the values of the neighborhood.                         |
    +------------+-------------------------------------------------------------------+
    | ``min``    | Minimum of the values of the neighborhood.                        |
    +------------+-------------------------------------------------------------------+
    | ``max``    | Maximum of the values of the neighborhood.                        |
    +------------+-------------------------------------------------------------------+
    | ``std``    | Standard deviation (population, ``ddof=0``) of the values.        |
    +------------+-------------------------------------------------------------------+
    | ``var``    | Variance (population, ``ddof=0``) of the values.                  |
    +------------+-------------------------------------------------------------------+
    | ``Q1``     | First quartile (25th percentile) of the values.                   |
    +------------+-------------------------------------------------------------------+
    | ``Q3``     | Third quartile (75th percentile) of the values.                   |
    +------------+-------------------------------------------------------------------+
    | ``IQR``    | Interquartile range ``Q3 - Q1``.                                  |
    +------------+-------------------------------------------------------------------+

    .. note::

        - Input ``values`` will be converted to ``numpy.float64``. A 1D array of shape
          :math:`(M,)` is treated as :math:`(M, 1)`. The statistics are computed
          independently for each of the :math:`P` components.
        - The non-finite values are not ignored: a NaN in a neighborhood gives NaN
          statistics for this node (``numpy`` semantics).
        - A node with an empty neighborhood gets NaN statistics.

    Parameters
    ----------
    values: ArrayLike
        An array of shape :math:`(M,)` or :math:`(M, P)` with the values.

    neighborhood: Union[ArrayLike, scipy.sparse.spmatrix]
        A matrix of shape :math:`(N, M)`, the non-zero entries of the row :math:`i` being
        the indices of the values in the neighborhood of the node :math:`i`.

    statistics: Union[str, Sequence[str]]
        The name of the statistics to compute, or a sequence of names.

    Returns
    -------
    Dict[str, numpy.ndarray]
        For each requested statistic, an array of shape :math:`(N, P)`.

    Raises
    ------
    TypeError
        If ``statistics`` is not a string or a sequence of strings.

    ValueError
        If a statistic is not implemented, or if the shapes are not consistent.

    Examples
    --------
    Statistics of a temperature defined at the 7 vertices of a disconnected graph, over
    the vertices within a distance 2.

    .. figure:: /_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%

       Disconnected graph with 7 vertices.

    .. code-block:: python
        :linenos:

        import numpy
        import scipy.sparse
        from pysdic.fem import compute_neighborhood, compute_neighborhood_statistics

        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (5, 6)]
        rows, columns = numpy.array(edges).T
        graph = scipy.sparse.coo_matrix((numpy.ones(len(edges)), (rows, columns)), shape=(7, 7))
        graph = graph + graph.T

        neighborhood = compute_neighborhood(graph, max_distance=2, self_neighborhood=True)
        temperature = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

        statistics = compute_neighborhood_statistics(temperature, neighborhood, ["min", "max", "mean"])
        print(statistics["min"].ravel())
        print(statistics["max"].ravel())
        print(statistics["mean"].ravel())

    .. code-block:: console

        [10. 10. 10. 10. 30. 60. 60.]
        [40. 40. 50. 50. 50. 70. 70.]
        [25. 25. 30. 30. 40. 65. 65.]

    """
    values = _as_array(values, "values", numpy.float64, shape=(None, None), finite=False, accept_1d_as_column=True)
    if not scipy.sparse.issparse(neighborhood):
        neighborhood = _as_array(neighborhood, "neighborhood", numpy.bool_, shape=(None, None), finite=False)
    neighborhood = scipy.sparse.csr_matrix(neighborhood, dtype=bool)
    neighborhood.eliminate_zeros()
    neighborhood.sort_indices()
    if neighborhood.shape[1] != values.shape[0]:
        raise ValueError(
            f"Input 'neighborhood' must have one column per value ({values.shape[0]}). "
            f"Got shape {neighborhood.shape}."
        )
    if isinstance(statistics, str):
        statistics = [statistics]
    elif not isinstance(statistics, (list, tuple)) or not all(isinstance(s, str) for s in statistics):
        raise TypeError(
            f"Input 'statistics' must be a string or a sequence of strings. Got {statistics!r}."
        )
    for statistic in statistics:
        if statistic not in _NEIGHBORHOOD_STATISTICS:
            raise ValueError(
                f"Unsupported statistic {statistic!r}. Supported statistics: {_NEIGHBORHOOD_STATISTICS}."
            )

    N, P = neighborhood.shape[0], values.shape[1]
    functions = {
        "mean": lambda v: numpy.mean(v, axis=0),
        "median": lambda v: numpy.median(v, axis=0),
        "min": lambda v: numpy.min(v, axis=0),
        "max": lambda v: numpy.max(v, axis=0),
        "std": lambda v: numpy.std(v, axis=0),
        "var": lambda v: numpy.var(v, axis=0),
        "Q1": lambda v: numpy.percentile(v, 25, axis=0),
        "Q3": lambda v: numpy.percentile(v, 75, axis=0),
        "IQR": lambda v: numpy.percentile(v, 75, axis=0) - numpy.percentile(v, 25, axis=0),
    }
    output = {statistic: numpy.full((N, P), numpy.nan, dtype=numpy.float64) for statistic in statistics}
    for node in range(N):
        indices = neighborhood.indices[neighborhood.indptr[node] : neighborhood.indptr[node + 1]]
        if indices.size == 0:
            continue  # Empty neighborhood: NaN
        neighbor_values = values[indices]
        for statistic in statistics:
            output[statistic][node] = functions[statistic](neighbor_values)
    return output