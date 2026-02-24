# Copyright 2026 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence
from typing import List, Optional, Iterable, Set, Union, Dict
from numbers import Integral
from numpy.typing import ArrayLike

import numpy
from collections import deque, defaultdict


def bfs_distance(
    graph: Sequence[Iterable[Integral]], 
    start: Integral, 
    max_distance: Optional[Integral] = None,
    *,
    _skip_type_check: bool = False
) -> List[Integral]:
    r"""
    Perform a breadth-first search (BFS) to compute the shortest path distances from a starting vertex to all other vertices in the graph.
    
    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph contains self-loops, if the graph contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input graph is not valid.

    
    Parameters
    ----------
    graph: Sequence[Iterable[Integral]]
        A sequence of iterables representing the adjacency list of the graph, where each index corresponds to a vertex and the iterable at that index contains the adjacent vertex indices.
        The length of the input sequence is equal to the number of vertices in the graph.

    start: Integral
        The starting vertex index for the BFS.
    
    max_distance : Optional[Integral], optional
        The maximum distance to search for in the BFS. If :obj:`None`, there is no maximum distance and all reachable vertices will be included in the distance computation. 
        If an integer is provided, the BFS will only consider vertices that are within this distance from the starting vertex, and any vertex that is not reachable within this distance will have a distance of -1. By default :obj:`None`.


    Returns
    -------
    List[:class:`int`]
        A list of integers representing the shortest path distances from the starting vertex to all other vertices in the graph. 
        The length of the output list is equal to the number of vertices in the graph.


    Raises
    ------
    TypeError
        If the input graph is not a sequence of iterables.
        If the start vertex index is not an integer.
        If maximum distance is not an integer or None.

    ValueError
        If the start vertex index is out of bounds.
        If maximum distance is a negative integer.
    
    
    Notes
    -----
    Lets note :math:`A` the starting vertex of the graph and :math:`T` the target vertex of the graph. 
    The BFS algorithm explores the graph in layers, starting from vertex :math:`A` and visiting all its neighbors (vertices directly connected to :math:`A`) before moving on to the neighbors' neighbors, and so on.
    The distance from vertex :math:`A` to vertex :math:`T` is determined by the layer at which vertex :math:`T` is first encountered during the BFS traversal.

    The distance from vertex :math:`A` to vertex :math:`T` is defined as follows:
    
    - :math:`\text{dist}(A, T) = 0` if vertex :math:`T` is the same as vertex :math:`A`, since the distance from a vertex to itself is zero.
    - :math:`\text{dist}(A, T) = 1` if vertex :math:`T` is a neighbor of vertex :math:`A`, meaning that there is a direct edge connecting vertex :math:`A` to vertex :math:`T`.
    - :math:`\text{dist}(A, T) = d \geq 1` if the shortest path from vertex :math:`A` to vertex :math:`T` has length :math:`d`.
    - :math:`\text{dist}(A, T) = -1` if vertex :math:`T` is not reachable from vertex :math:`A`, meaning that there is no path connecting vertex :math:`A` to vertex :math:`T` in the graph.
    - :math:`\text{dist}(A, T) = -1` if vertex :math:`T` is not reachable from vertex :math:`A` within the specified ``max_distance``.


    See Also
    --------
    :func:`bfs_neighborhood`
        Perform a breadth-first search (BFS) to compute the neighborhood of a starting vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.


   
    Examples
    --------
    Create a simple disconnected graph of 7 vertices and compute the shortest path distances from a starting vertex (0) to all other vertices in the graph.
    
    .. figure:: ../_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%
       
       Disconnected graph with 7 vertices for BFS example. Distance from vertex 0 to vertex 4 is 3 (0 -> 2 -> 3 -> 4).
       
    
    .. code-block:: python
        :linenos:
        
        from pysdic import bfs_distance
        
        graph = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3], [6], [5]]
        start_vertex = 0
        max_distance = None
        
        distances = bfs_distance(graph, start_vertex, max_distance)
        print(distances)

    .. code-block:: console
    
        [0, 1, 1, 2, 3, -1, -1]

    """
    if not _skip_type_check:
        if not isinstance(graph, Sequence):
            raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)}.")
        if not all(isinstance(neighbors, Iterable) for neighbors in graph):
            raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)} with elements of type {set(type(neighbors) for neighbors in graph)}.") 
        
        if not isinstance(start, Integral):
            raise TypeError(f"start must be an integer. Got {type(start)}.")
        if start < 0 or start >= len(graph):
            raise ValueError(f"start must be an integer between 0 and n_vertices-1. Got {start}.")
        
        if max_distance is not None and (not isinstance(max_distance, Integral)):
            raise TypeError(f"max_distance must be an integer or None. Got {type(max_distance)}.")    
        if max_distance is not None and max_distance < 0:
            raise ValueError(f"max_distance must be a non-negative integer or None. Got {max_distance}.")
    
    # Breadth-first search (BFS) to find shortest paths from vertex i
    n_vertices = len(graph)
    distances = [-1] * n_vertices  # -1 indicates unvisited vertices
    distances[start] = 0
    queue = deque([start])

    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        
        if max_distance is not None and current_distance >= max_distance:
            continue
        
        for neighbor in graph[node]:
            if distances[neighbor] == -1: 
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distances



def bfs_neighborhood(
    graph:  Sequence[Iterable[Integral]],
    start: Integral, 
    max_distance: Integral, 
    self_neighborhood: bool = True,
    *,
    _skip_type_check: bool = False
) -> List[int]:
    r"""
    Perform a breadth-first search (BFS) to compute the neighborhood of a starting vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.

    .. warning::
    
        No tests are performed to check if the input graph is valid (e.g., if the graph contains self-loops, if the graph contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input graph is not valid.
    
    Parameters
    ----------
    graph : Sequence[Iterable[Integral]]
        A sequence of iterables representing the adjacency list of the graph, where each index corresponds to a vertex and the iterable at that index contains the adjacent vertex indices.
        The length of the input sequence is equal to the number of vertices in the graph.
        
    start : Integral
        The starting vertex index for the BFS.
        
    max_distance : Optional[Integral], optional
        The maximum distance to search for in the BFS. The neighborhood will include all vertices that are within this distance from the vertex in the connectivity of the mesh.
        
    self_neighborhood : :class:`bool`, optional
        If :obj:`True`, the neighborhood of the vertex will include the vertex itself (i.e., the vertex will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of the vertex will not include the vertex itself. By default :obj:`True`.
        
    
    Returns
    -------
    List[:class:`int`]
        A list of vertex indices representing the neighborhood of the starting vertex, where the neighborhood includes all vertices that are within the specified distance from the vertex in the connectivity of the mesh.
        
        
    Raises
    ------
    TypeError
        If the input graph is not a sequence of iterables.
        If the start vertex index is not an integer.
        If maximum distance is not an integer or None.
        If self_neighborhood is not a boolean.

    ValueError
        If the start vertex index is out of bounds.
        If maximum distance is a negative integer.


    Notes
    -----
    Lets note :math:`A` the starting vertex of the graph and :math:`T` the target vertex of the graph.
    The BFS algorithm explores the graph in layers, starting from vertex :math:`A` and visiting all its neighbors (vertices directly connected to :math:`A`) before moving on to the neighbors' neighbors, and so on.
    At the max_distance layer, the BFS will stop and the neighborhood of vertex :math:`A` will include all vertices that were visited during the BFS traversal up to and including the max_distance layer.

    There is some particular cases for the neighborhood of vertex :math:`A` depending on the value of :obj:`max_distance`:
    
    - For max_distance = 0, the neighborhood of vertex :math:`A` will include only vertex :math:`A` itself (if self_neighborhood is True).
    - For max_distance = 1, the neighborhood of vertex :math:`A` will include vertex :math:`A` itself (if self_neighborhood is True) and all its neighbors (directly extracted from the graph).


    See Also
    --------
    :func:`bfs_distance`
        Perform a breadth-first search (BFS) to compute the shortest path distances from a starting vertex to all other vertices in the graph, where the distance is defined as the number of edges in the shortest path between the vertices.
        

    Examples
    --------
    Create a simple disconnected graph of 7 vertices and compute the shortest path distances from a starting vertex (0) to all other vertices in the graph.
    
    .. figure:: ../_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%
       
       Disconnected graph with 7 vertices for BFS example. Vertices 0, 1, 2 and 3 are the only vertices in the neighborhood of vertex 0 for max_distance = 2.
    
    .. code-block:: python
        :linenos:
        
        from pysdic import bfs_neighborhood
        
        graph = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3], [6], [5]]
        start_vertex = 0
        max_distance = 2
        
        neighborhood = bfs_neighborhood(graph, start_vertex, max_distance, self_neighborhood=True)
        print(neighborhood)
        
    .. code-block:: console
    
        [0, 1, 2, 3]
        
    """
    if not _skip_type_check:
        if not isinstance(graph, Sequence):
            raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)}.")
        if not all(isinstance(neighbors, Iterable) for neighbors in graph):
            raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)} with elements of type {set(type(neighbors) for neighbors in graph)}.") 
        
        if not isinstance(start, int):
            raise TypeError(f"start must be an integer. Got {type(start)}.")
        if start < 0 or start >= len(graph):
            raise ValueError(f"start must be an integer between 0 and n_vertices-1. Got {start}.")
        
        if max_distance is not None and (not isinstance(max_distance, int)):
            raise TypeError(f"max_distance must be an integer or None. Got {type(max_distance)}.")    
        if max_distance is not None and max_distance < 0:
            raise ValueError(f"max_distance must be a non-negative integer or None. Got {max_distance}.")
        
        if not isinstance(self_neighborhood, bool):
            raise TypeError(f"self_neighborhood must be a boolean. Got {type(self_neighborhood)}.")
    
    # Breadth-first search (BFS) to find the neighborhood of vertex i
    visited = {start: 0}  # key: vertex index, value: distance from start
    queue = deque([start])

    while queue:
        node = queue.popleft()
        d = visited[node]

        # Stop if max distance reached
        if max_distance is not None and d >= max_distance:
            continue

        for nbr in graph[node]:
            if nbr not in visited:
                visited[nbr] = d + 1
                queue.append(nbr)

    if self_neighborhood:
        return [int(v) for v in visited]
    else:
        return [int(v) for v in visited if v != start]
    

def create_vertices_adjacency_graph(
    connectivity: ArrayLike,
    n_vertices: Optional[Integral] = None
) -> List[Set[Integral]]:
    r"""
    Create the vertices adjacency graph from the connectivity of the mesh, where two vertices are neighbors if there is an element in the mesh containing both vertices.

    Note that the function is indepedent of the geometry of the element, thus two vertices will be considered as neighbors if they are part of the same element, regardless of the shape of the element (e.g., triangle, quadrilateral, tetrahedron, hexahedron, ...).
    
    .. note::
    
        The input :obj:`connectivity` must be integral-array like. The output integral values will be of the same type as the input :obj:`connectivity` (e.g., if the input :obj:`connectivity` is of type :obj:`numpy.int64`, the output integral values will be of type :obj:`numpy.int64`).

    .. warning::
    
        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...).

    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
        
    n_vertices : Optional[Integral], optional
        The total number of vertices in the mesh. If not provided, it will be inferred from the connectivity array.
        
    
    Returns
    -------
    List[Set[Integral]]
        A list of sets representing the adjacency list of the graph, where each index corresponds to a vertex and each set at that index contains the neighboring vertex indices.
    
        
    Raises
    ------
    TypeError
        If n_vertices is not an integer or None.

    ValueError
        If n_vertices is a non-positive integer.
        If the connectivity array is not 2D.
        
        
    See Also
    --------
    :func:`create_elements_adjacency_graph`
        Create the elements adjacency graph from the connectivity of the mesh.
        
    
    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the vertices adjacency graph from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh. The neighboring vertices of vertex 1 are vertices 0, 2, 4, 5 and 6 since they are part of the same elements in the mesh (elements 0 and 1).
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import create_vertices_adjacency_graph
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]], dtype=np.int64)
        n_vertices = 10
        
        adjacency_graph = create_vertices_adjacency_graph(
            connectivity=connectivity,
            n_vertices=n_vertices
        )
        
        print(f"vertices adjacency graph with length: {len(adjacency_graph)}")
        subset_1 = adjacency_graph[1]
        print(f"vertices adjacency graph (neighbors of vertex 1): {subset_1}")
        
    .. code-block:: console

        vertices adjacency graph with length: 10
        vertices adjacency graph (neighbors of vertex 1): {np.int64(0), np.int64(2), np.int64(4), np.int64(5), np.int64(6)}
        
    """
    connectivity = numpy.asarray(connectivity)
    
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        raise TypeError(f"connectivity must be an integral array. Got array of type {connectivity.dtype}.")
    if connectivity.ndim != 2:
        raise ValueError(f"'connectivity' must be a 2D array of shape (N_e, N_vpe). Got {connectivity.ndim}D array instead.")
    
    # Extract number of vertices
    if n_vertices is None:
        n_vertices = numpy.max(connectivity) + 1
    
    if not isinstance(n_vertices, Integral):
        raise TypeError(f"n_vertices must be an integer or None. Got {type(n_vertices)}.")
    if n_vertices <= 0:
        raise ValueError(f"n_vertices must be a positive integer. Got {n_vertices} instead.")
    
    # Create the edges from the connectivity
    # For Nvpe vertices per element, we have Nvpe*(Nvpe-1)/2 edges per element
    Ne, Nvpe = connectivity.shape
    Ned = Ne * Nvpe * (Nvpe - 1) // 2
    
    irow, icol = numpy.triu_indices(Nvpe, k=1)
    row = connectivity[:, irow].ravel()
    col = connectivity[:, icol].ravel()
    
    # Build the graph as an adjacency list for the BFS algorithm
    graph = [set() for _ in range(n_vertices)]
    for u, v in zip(row, col):
        graph[u].add(v)
        graph[v].add(u)
    
    return graph


def create_elements_adjacency_graph(
    connectivity: ArrayLike
) -> List[Set[int]]:
    r"""
    Create the elements adjacency graph from the connectivity of the mesh, where two elements are neighbors if they share at least one vertex in the mesh.

    Note that the function is indepedent of the geometry of the element, thus two elements will be considered as neighbors if they share at least one vertex, regardless of the shape of the elements (e.g., triangle, quadrilateral, tetrahedron, hexahedron, ...).
    
    .. note::
    
        The input :obj:`connectivity` must be integral-array like.
        
    .. warning::
    
        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...).
        
    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
        
        
    Returns
    -------
    List[Set[:class:`int`]]
        A list of sets representing the adjacency list of the graph, where each index corresponds to an element and each set at that index contains the neighboring element indices.
        
    
    Raises
    ------
    ValueError
        If the connectivity array is not 2D.
        
    
    See Also
    --------
    :func:`create_vertices_adjacency_graph`
        Create the vertices adjacency graph from the connectivity of the mesh.
        
        
    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the elements adjacency graph from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh. The neighboring elements of element 3 are elements 1 and 2 since they share at least one vertex in the mesh.
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import create_elements_adjacency_graph
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]], dtype=np.int64)
        n_vertices = 10
        
        adjacency_graph = create_elements_adjacency_graph(connectivity=connectivity)
        
        print(f"elements adjacency graph with length: {len(adjacency_graph)}")
        subset_1 = adjacency_graph[3]
        print(f"elements adjacency graph (neighbors of element 3): {subset_1}")
        
    .. code-block:: console

        elements adjacency graph with length: 4
        elements adjacency graph (neighbors of element 3): {1, 2}
        
    """
    connectivity = numpy.asarray(connectivity)
    
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        raise TypeError(f"connectivity must be an integral array. Got array of type {connectivity.dtype}.")
    if connectivity.ndim != 2:
        raise ValueError(f"'connectivity' must be a 2D array of shape (N_e, N_vpe). Got {connectivity.ndim}D array instead.")
    
    # Build the vertex to element mapping
    vertex_to_elements = defaultdict(list)
    
    for element_index, vertices in enumerate(connectivity):
        for vertex in vertices:
            vertex_to_elements[vertex].append(element_index)
            
    # Build the graph adjacency matrix for elements
    graph = [set() for _ in range(connectivity.shape[0])]
    for vertex, elements in vertex_to_elements.items():
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                graph[elements[i]].add(elements[j])
                graph[elements[j]].add(elements[i])
    
    return graph


def compute_adjacency_matrix(
    graph: Sequence[Iterable[Integral]],
    max_distance: Optional[Integral] = None
) -> numpy.ndarray:
    r"""
    Compute the adjacency matrix from the adjacency graph of the vertices, where the distance between two vertices is defined as the minimum number of elements that must be traversed to go from one vertex to another.

    The output is a symmetric square matrix of shape (:math:`N_v`, :math:`N_v`), where :math:`N_v` is the number of vertices, and the value at position (i, j) represents the distance between vertex i and vertex j in the graph.

    .. note::
    
        The computation of the adjacency matrix is performed using a breadth-first search (BFS) algorithm for each vertex in the graph.
        
    .. warning::
    
        No tests are performed to check if the input graph is valid (e.g., if the graph contains self-loops, if the graph contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input graph is not valid.
        
    Parameters
    ----------
    graph : Sequence[Iterable[Integral]]
        A sequence of iterables representing the adjacency list of the graph, where each index corresponds to a vertex and the iterable at that index contains the adjacent vertex indices.
        
    max_distance : Optional[Integral], optional
        The maximum distance to consider between vertices. If the distance between two vertices exceeds this value,
        the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.
        
        
    Returns
    -------
    :class:`numpy.ndarray`
        A square array of shape (:math:`N_v`, :math:`N_v`) representing the path distance matrix between vertices, where the value at position (i, j) represents the distance between vertex i and vertex j in the graph.
        
    
    Raises
    ------
    TypeError
        If the input graph is not a sequence of iterables.
        If maximum distance is not an integer or None.
        
    ValueError
        If maximum distance is a negative integer.
        If n_vertices is a non-positive integer.


    See Also
    --------
    :func:`bfs_distance`
        Perform a breadth-first search (BFS) to compute the shortest path distances from a starting vertex to all other vertices in the graph, where the distance is defined as the number of edges in
        the shortest path between the vertices.
        
    :func:`compute_vertices_adjacency_matrix`
        Convenience function to compute the vertices path distance matrix (adjacency matrix) from the connectivity of the mesh.
        
    :func:`compute_elements_adjacency_matrix`
        Convenience function to compute the elements path distance matrix (adjacency matrix) from the connectivity of the mesh.
        
        
    Notes
    -----
    The distance between two vertices i and j in the graph is defined as follows:
    
    - :math:`\text{dist}(i, j) = 0` if vertex :math:`j` is the same as vertex :math:`i`, since the distance from a vertex to itself is zero.
    - :math:`\text{dist}(i, j) = 1` if vertex :math:`j` is a neighbor of vertex :math:`i`, meaning that there is a direct edge connecting vertex :math:`i` to vertex :math:`j`.
    - :math:`\text{dist}(i, j) = d \geq 1` if the shortest path from vertex :math:`i` to vertex :math:`j` has length :math:`d`.
    - :math:`\text{dist}(i, j) = -1` if vertex :math:`j` is not reachable from vertex :math:`i`, meaning that there is no path connecting vertex :math:`i` to vertex :math:`j` in the graph.
    - :math:`\text{dist}(i, j) = -1` if vertex :math:`j` is not reachable from vertex :math:`i` within the specified ``max_distance``.
    
    
    Examples
    --------
    Create a simple disconnected graph of 7 vertices and build the adjacency matrix from the adjacency graph of the vertices, where the distance between two vertices is defined as the minimum number of elements that must be traversed to go from one vertex to another.
    
    .. figure:: ../_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%
       
       Disconnected graph with 7 vertices for adjacency matrix example.
    
    .. code-block:: python
        :linenos:
        
        from pysdic import compute_adjacency_matrix
        
        graph = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3], [6], [5]]
        start_vertex = 0
        max_distance = None
        
        adjacency_matrix = compute_adjacency_matrix(graph, max_distance=max_distance)
        print(f"adjacency_matrix (shape={adjacency_matrix.shape}):")
        print(adjacency_matrix)
        
    .. code-block:: console
    
        adjacency_matrix (shape=(7, 7)):
        [[ 0  1  1  2  3 -1 -1]
         [ 1  0  1  2  3 -1 -1]
         [ 1  1  0  1  2 -1 -1]
         [ 2  2  1  0  1 -1 -1]
         [ 3  3  2  1  0 -1 -1]
         [-1 -1 -1 -1 -1  0  1]
         [-1 -1 -1 -1 -1  1  0]]

    """
    if not isinstance(graph, Sequence):
        raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)}.")
    if not all(isinstance(neighbors, Iterable) for neighbors in graph):
        raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)} with elements of type {set(type(neighbors) for neighbors in graph)}.")
    
    if max_distance is not None and (not isinstance(max_distance, Integral)):
        raise TypeError(f"max_distance must be an integer or None. Got {type(max_distance)}.")    
    if max_distance is not None and max_distance < 0:
        raise ValueError(f"max_distance must be a non-negative integer or None. Got {max_distance}.")
    
    n_vertices = len(graph)
    adjacency_matrix = numpy.full((n_vertices, n_vertices), -1, dtype=numpy.int64)
    for i in range(n_vertices):
        distances = bfs_distance(graph, start=i, max_distance=max_distance, _skip_type_check=True)
        adjacency_matrix[i] = distances
    
    return adjacency_matrix
    
    
def compute_vertices_adjacency_matrix(
    connectivity: ArrayLike,
    n_vertices: Optional[Integral] = None,
    max_distance: Optional[Integral] = None
) -> numpy.ndarray:
    r"""
    Compute the vertices path distance matrix (adjacency matrix) from the connectivity of the mesh, where the distance between two vertices is defined as the minimum number of elements that must be traversed to go from one vertex to another.
    
    Convenience function that combines the creation of the vertices adjacency graph from the connectivity of the mesh and the computation of the adjacency matrix from the vertices adjacency graph.
    
    .. note::
    
        The input :obj:`connectivity` must be integral-array like.
        
    .. warning::
    
        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input connectivity is not valid.


    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
        
    n_vertices : Optional[Integral], optional
        The total number of vertices in the mesh. If not provided, it will be inferred from the connectivity array.
    
    max_distance : Optional[Integral], optional
        The maximum distance to consider between vertices. If the distance between two vertices exceeds this value,
        the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.


    Returns
    -------
    :class:`numpy.ndarray`
        A square array of shape (:math:`N_v`, :math:`N_v`) representing the path distance matrix between vertices.
        
        
    Raises
    ------
    TypeError
        If max_distance is not an integer or None.
        If n_vertices is not an integer or None.
        
    ValueError
        If max_distance is a negative integer.
        If n_vertices is a non-positive integer.
        If the connectivity array is not 2D.
        
        
    See Also
    --------
    :func:`compute_adjacency_matrix`
        Compute the adjacency matrix from the adjacency graph of the vertices, where the distance between two vertices is defined as the minimum number of elements that must be traversed to go from one vertex to another.
        
    :func:`create_vertices_adjacency_graph`
        Create the vertices adjacency graph from the connectivity of the mesh, where two vertices are neighbors if there is an element in the mesh containing both vertices.
    

    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the vertices adjacency matrix from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh.
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import compute_vertices_adjacency_matrix
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]])
        n_vertices = 10
        
        adjacency_matrix = compute_vertices_adjacency_matrix(connectivity=connectivity, n_vertices=n_vertices)
        
        print(f"vertices adjacency matrix with shape: {adjacency_matrix.shape}")
        extracted_subset = adjacency_matrix[1, :]
        print(f"vertices adjacency matrix (distances from vertex 1 to all vertices): {extracted_subset}")
        
    .. code-block:: console

        vertices adjacency matrix with shape: (10, 10)
        vertices adjacency matrix (distances from vertex 1 to all vertices): [1, 0, 1, 2, 1, 1, 1, 2, 3, 3]
    
    """
    # Extract the graph as an adjacency list from the connectivity
    graph = create_vertices_adjacency_graph(connectivity=connectivity, n_vertices=n_vertices)
    
    # Compute the adjacency matrix from the graph
    adjacency_matrix = compute_adjacency_matrix(graph=graph, max_distance=max_distance)
    
    return adjacency_matrix


def compute_elements_adjacency_matrix(
    connectivity: ArrayLike,
    max_distance: Optional[Integral] = None
) -> numpy.ndarray:
    r"""
    Compute the elements path distance matrix (adjacency matrix) from the connectivity of the mesh, where the distance between two elements is defined as the minimum number of elements that must be traversed to go from one element to another.
    
    Convenience function that combines the creation of the elements adjacency graph from the connectivity of the mesh and the computation of the adjacency matrix from the elements adjacency graph.
    
    .. note::
    
        The input :obj:`connectivity` must be integral-array like.
        
    .. warning::
    
        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input connectivity is not valid.


    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
        
    max_distance : Optional[Integral], optional
        The maximum distance to consider between elements. If the distance between two elements exceeds this value,
        the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.


    Returns
    -------
    :class:`numpy.ndarray`
        A square array of shape (:math:`N_e`, :math:`N_e`) representing the path distance matrix between elements.
        
        
    Raises
    ------
    TypeError
        If max_distance is not an integer or None.
        
    ValueError
        If max_distance is a negative integer.
        If the connectivity array is not 2D.
        
        
    See Also
    --------
    :func:`compute_adjacency_matrix`
        Compute the adjacency matrix from the adjacency graph of the elements, where the distance between two elements is defined as the minimum number of elements that must be traversed to go from one element to another.
        
    :func:`create_elements_adjacency_graph`
        Create the elements adjacency graph from the connectivity of the mesh, where two elements are neighbors if they share a common vertex.
    

    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the vertices adjacency matrix from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh.
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import compute_elements_adjacency_matrix
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]])
        n_elements = connectivity.shape[0] # 4 elements
        
        adjacency_matrix = compute_elements_adjacency_matrix(connectivity=connectivity)
        
        print(f"elements adjacency matrix with shape: {adjacency_matrix.shape}")
        print(adjacency_matrix)
        
    .. code-block:: console

        elements adjacency matrix with shape: (4, 4)
        [[ 0  1  2  2]
         [ 1  0  1  2]
         [ 2  1  0  1]
         [ 2  2  1  0]]
    
    """
    # Extract the graph as an adjacency list from the connectivity
    graph = create_elements_adjacency_graph(connectivity=connectivity)
    
    # Compute the adjacency matrix from the graph
    adjacency_matrix = compute_adjacency_matrix(graph=graph, max_distance=max_distance)
    
    return adjacency_matrix


def compute_neighborhood(
    graph: Sequence[Iterable[Integral]],
    max_distance: Integral = 1,
    self_neighborhood: bool = True
) -> List[List[int]]:
    r"""
    Compute the neighborhood of each vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.

    .. note::
    
        The neighborhood of a vertex is computed using a breadth-first search (BFS) algorithm for each vertex in the graph, where the BFS is limited to a maximum distance specified by :obj:`max_distance`.
        
    .. warning::

        No tests are performed to check if the input graph is valid (e.g., if the graph contains self-loops, if the graph contains invalid vertex indices, ...). 
        The behavior of the function is undefined if the input graph is not valid.
        
    Parameters
    ----------
    graph : Sequence[Iterable[Integral]]
        A sequence of iterables representing the adjacency list of the graph, where each index corresponds to a vertex and the iterable at that index contains the adjacent vertex indices.
        
    max_distance : Integral, optional
        The maximum distance to consider for the neighborhood of each vertex. The neighborhood will include all vertices that are within this distance from the vertex in the connectivity of the mesh. By default 1,
        which means that only directly adjacent vertices will be included in the neighborhood.
        
    self_neighborhood : :class:`bool`, optional
        If :obj:`True`, the neighborhood of each vertex will include the vertex itself (i.e., the vertex will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of each vertex will not include the vertex itself. By default :obj:`True`.
        
        
    Returns
    -------
    List[List[:class:`int`]]
        A list where each element is a list containing the indices of the neighboring vertices for the corresponding vertex in the graph.
        
    
    Raises
    ------
    TypeError
        If the input graph is not a sequence of iterables.
        If maximum distance is not an integer or None.
        If self_neighborhood is not a boolean.
        
    ValueError
        If maximum distance is a negative integer.
        
    
    Notes
    -----
    The neighborhood of a vertex i in the graph is defined as all vertices j such that the distance between vertex i and vertex j in the graph is less than or equal to the specified :obj:`max_distance`.
    
    
    See Also
    --------
    :func:`bfs_neighborhood`
        Perform a breadth-first search (BFS) to compute the neighborhood of a starting vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.
        
    :func:`compute_vertices_neighborhood`
        Compute the neighborhood of each vertex in the mesh based on the connectivity and a specified maximum distance.
        
    :func:`compute_elements_neighborhood`
        Compute the neighborhood of each element in the mesh based on the connectivity and a specified maximum distance.
        
        
    Examples
    --------
    Create a simple disconnected graph of 7 vertices and compute the neighborhood of each vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.
    
    .. figure:: ../_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%
       
       Disconnected graph with 7 vertices for BFS example. Neighborhood of vertex 0 with max_distance=2 includes vertices 0, 1, 2 and 3 since they are within a distance of 2 from vertex 0 in the graph.
    
    .. code-block:: python
        :linenos:
        
        from pysdic import compute_neighborhood
        
        graph = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3], [6], [5]]
        start_vertex = 0
        max_distance = 2
        
        neighborhoods = compute_neighborhood(graph, max_distance, self_neighborhood=True)
        
        print(f"neighborhood length: {len(neighborhoods)}")
        subneighborhood = neighborhoods[0]
        print(f"neighborhood of vertex 0 with max_distance={max_distance}: {subneighborhood}")
        
    .. code-block:: console
    
        neighborhood length: 7
        neighborhood of vertex 0 with max_distance=2: [0, 1, 2, 3]
        
    """
    if not isinstance(graph, Sequence):
        raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)}.")
    if not all(isinstance(neighbors, Iterable) for neighbors in graph):
        raise TypeError(f"graph must be a sequence of iterables representing the adjacency list of the graph. Got {type(graph)} with elements of type {set(type(neighbors) for neighbors in graph)}.")
    
    if max_distance is not None and (not isinstance(max_distance, int)):
        raise TypeError(f"max_distance must be an integer or None. Got {type(max_distance)}.")
    if max_distance is not None and max_distance < 0:
        raise ValueError(f"max_distance must be a non-negative integer or None. Got {max_distance}.")
    
    if not isinstance(self_neighborhood, bool):
        raise TypeError(f"self_neighborhood must be a boolean. Got {type(self_neighborhood)}.")
    
    n_vertices = len(graph)
    vertices_neighborhood = []
    for i in range(n_vertices):
        neighborhood = bfs_neighborhood(graph, i, max_distance=max_distance, self_neighborhood=self_neighborhood) # List[int] of the indices of the neighboring vertices for vertex i
        vertices_neighborhood.append(neighborhood)
        
    return vertices_neighborhood
    

def compute_vertices_neighborhood(
    connectivity: ArrayLike,
    n_vertices: Optional[int] = None,
    max_distance: int = 1,
    self_neighborhood: bool = True
) -> List[List[int]]:    
    r"""
    Compute the neighborhood of each vertex in the mesh, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.
    
    Convenience function that combines the creation of the vertices adjacency graph from the connectivity of the mesh and the computation of the neighborhood of each vertex in the mesh based on the connectivity and a specified maximum distance.
        
    .. note::
    
        The input :obj:`connectivity` must be integral-array like.
        
    .. warning::

        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined if the input connectivity is not valid.
        
    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
        
    n_vertices : Optional[Integral], optional
        The total number of vertices in the mesh. If not provided, it will be inferred from the connectivity array.
    
    max_distance : Optional[Integral], optional
        The maximum distance to consider between vertices. If the distance between two vertices exceeds this value,
        the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.
        
    self_neighborhood : :class:`bool`, optional
        If :obj:`True`, the neighborhood of each vertex will include the vertex itself (i.e., the vertex will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of each vertex will not include the vertex itself. By default :obj:`True`.


    Returns
    -------       
    List[List[:class:`int`]]
        A list where each element is a list containing the indices of the neighboring vertices for the corresponding vertex in the mesh.
        
    
    Raises
    ------
    TypeError
        If max_distance is not an integer or None.
        If n_vertices is not an integer or None.
        If self_neighborhood is not a boolean.
        
    ValueError
        If max_distance is a negative integer.
        If n_vertices is a non-positive integer.
        If the connectivity array is not 2D.
        
        
    See Also
    --------
    :func:`compute_neighborhood`
        Compute the neighborhood of each vertex in the graph, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.
        
    :func:`create_vertices_adjacency_graph`
        Create the vertices adjacency graph from the connectivity of the mesh, where two vertices are neighbors if there is an element in the mesh containing both vertices.        
    
    
    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the vertices adjacency matrix from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh. The neighboring vertices of vertex 1 with max_distance=1 are vertices 0, 1, 2, 4, 5, 6 and 7 since they are within a distance of 1 from vertex 1 in the graph.
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import compute_vertices_neighborhood
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]], dtype=np.int64)
        n_vertices = 10
        
        neighborhood = compute_vertices_neighborhood(connectivity=connectivity, n_vertices=n_vertices, max_distance=1, self_neighborhood=True)
        
        print(f"vertices neighborhood (length={len(neighborhood)}):")
        subset_1 = neighborhood[1]
        print(f"vertices neighborhood (neighbors of vertex 1 with max_distance=1): {subset_1}")
        
    .. code-block:: console

        vertices neighborhood (length=10):
        vertices neighborhood (neighbors of vertex 1 with max_distance=1): [1, 0, 2, 4, 5, 6]
        
    """
    # Extract the graph as an adjacency list from the connectivity
    graph = create_vertices_adjacency_graph(connectivity=connectivity, n_vertices=n_vertices)
    
    # Compute the neighborhood of each vertex in the graph
    vertices_neighborhood = compute_neighborhood(graph=graph, max_distance=max_distance, self_neighborhood=self_neighborhood)
    
    return vertices_neighborhood



def compute_elements_neighborhood(
    connectivity: ArrayLike,
    max_distance: Integral = 1,
    self_neighborhood: bool = True
) -> List[List[int]]:    
    r"""
    Compute the neighborhood of each element in the mesh, where the neighborhood of an element is defined as the set of elements that are within a specified distance from the element in the connectivity of the mesh.
    
    Convenience function that combines the creation of the elements adjacency graph from the connectivity of the mesh and the computation of the neighborhood of each element in the mesh based on the connectivity and a specified maximum distance.
        
    .. note::
    
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the input connectivity is valid (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined if the input connectivity is not valid.
        
    Parameters
    ----------
    connectivity : ArrayLike
        An array of shape :math:`(N_e, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.
    
    max_distance : Optional[Integral], optional
        The maximum distance to consider between elements. If the distance between two elements exceeds this value,
        the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.
        
    self_neighborhood : :class:`bool`, optional
        If :obj:`True`, the neighborhood of each element will include the element itself (i.e., the element will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of each element will not include the element itself. By default :obj:`True`.


    Returns
    -------       
    List[List[:class:`int`]]
        A list where each element is a list containing the indices of the neighboring elements for the corresponding element in the mesh.
        
    
    Raises
    ------
    TypeError
        If max_distance is not an integer or None.
        If self_neighborhood is not a boolean.
        
    ValueError
        If max_distance is a negative integer.
        If the connectivity array is not 2D.
        
        
    See Also
    --------
    :func:`compute_neighborhood`
        Compute the neighborhood of each element in the graph, where the neighborhood of an element is defined as the set of elements that are within a specified distance from the element in the connectivity of the mesh.
        
    :func:`create_elements_adjacency_graph`
        Create the elements adjacency graph from the connectivity of the mesh, where two elements are neighbors if they share a common vertex in the mesh.        
    
    
    Examples
    --------
    Create a simple ``quadrangle 4`` mesh with 10 vertices and 4 elements and compute the elements adjacency matrix from the connectivity.
    
    .. figure:: ../_static/adjacency/graph_mesh.png
        :align: center
        :width: 50%
        
        ``Quadrangle 4`` mesh. The neighboring elements of element 0 with max_distance=1 are elements 0, 1 since they are within a distance of 1 from element 0 in the graph.
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import compute_elements_neighborhood
        
        connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]])
        
        neighborhood = compute_elements_neighborhood(connectivity=connectivity, max_distance=1, self_neighborhood=True)
        
        print(f"elements neighborhood (length={len(neighborhood)}):")
        subset_1 = neighborhood[0]
        print(f"elements neighborhood (neighbors of element 0 with max_distance=1): {subset_1}")
        
    .. code-block:: console

        elements neighborhood (length=10):
        elements neighborhood (neighbors of element 0 with max_distance=1): [0, 1]
        
    """
    # Extract the graph as an adjacency list from the connectivity
    graph = create_elements_adjacency_graph(connectivity=connectivity)
    
    # Compute the neighborhood of each element in the graph
    elements_neighborhood = compute_neighborhood(graph=graph, max_distance=max_distance, self_neighborhood=self_neighborhood)
    
    return elements_neighborhood



def compute_neighborhood_statistics(
    property_array: ArrayLike,
    neighborhood: List[List[int]],
    statistics: Union[str, Sequence[str]],
    integrated_property: bool = False
) -> Dict[str, numpy.ndarray]:
    """
    Compute the specified statistics for the given property array defined at the vertices of a graph (or integration points) based on the neighborhood information for each vertex.
    
    This function allows you to compute various statistics (e.g., mean, median, min, max, std) for a property defined at the vertices/elements of a mesh, using the values of the property from neighboring vertices/elements as defined in the neighborhood list. 
    For each vertex/element, we will gather the values of the specified property from its neighboring vertices/elements and then compute the specified statistics to obtain a new array containing the computed statistics for each vertex/element.
    
    Implemented statistics include:
     
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | Statistic      | Description                                                                                                  |
    +================+==============================================================================================================+
    | ``mean``       | Compute the mean of the property values from the neighbors.                                                  |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``median``     | Compute the median of the property values from the neighbors.                                                |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``min``        | Compute the minimum of the property values from the neighbors.                                               |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``max``        | Compute the maximum of the property values from the neighbors.                                               |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``std``        | Compute the standard deviation of the property values from the neighbors.                                    |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``var``        | Compute the variance of the property values from the neighbors.                                              |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``Q1``         | Compute the first quartile (25th percentile) of the property values from the neighbors.                      |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``Q3``         | Compute the third quartile (75th percentile) of the property values from the neighbors.                      |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    | ``IQR``        | Compute the interquartile range (IQR) of the property values from the neighbors, defined as Q3 - Q1.         |
    +----------------+--------------------------------------------------------------------------------------------------------------+
    
    .. note::
    
        The property array is converted to a 2D array of shape :math:`(N, P)` with type :obj:`numpy.float64` to ensure consistent processing, where :math:`N` is the number of vertices/elements and :math:`P` is the number of properties per vertex/element. 
        The computation will be done independently for each property if multiple properties are provided (i.e., if :math:`P > 1`), and the resulting array will have the same shape :math:`(N, P)` containing the computed statistics values for each vertex/element and each property.
        The neighborhood list will be treated as a list of sequences of type :obj:`numpy.int64` containing the indices of neighboring vertices/elements for each vertex/element in the mesh.
        
    .. note::
    
        If a vertex/element has no neighbors (i.e., the corresponding sequence in the neighborhood list is empty), the computed statistics for that vertex/element will be set to :obj:`numpy.nan` for all properties, since there are no values to compute the statistics from.
    
    
    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N, P)` containing the property values defined at the vertices/elements of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N, 1)`.
        
    neighborhood : List[List[:class:`int`]]]
        A list of sequences, where each sequence contains the indices of neighboring vertices/elements for each vertex/element in the mesh. The length of the list should be equal to the number of vertices/elements, and each sequence should contain the indices of the neighbors for the corresponding vertex/element.
        
    statistics : Union[:class:`str`, Sequence[:class:`str`]]
        A string or a sequence of strings specifying the statistics(s) to compute for each vertex/element based on its neighbors.
        
    integrated_property : :class:`bool`, optional
        If :obj:`True`, enlarge the function to accept properties defined at the integration points of a mesh and compute the specified statistics for each vertex/element based on the values of the property at the integration points of its neighboring elements. 
        See the note below for more details. By default :obj:`False`.
    
    
    Returns
    -------
    Dict[:class:`str`, :class:`numpy.ndarray`]
        A dictionary where the keys are the names of the computed statistics and the values are arrays of shape :math:`(N, P)` containing the computed statistics values for each vertex/element and each property.
        
    
    Raises
    ------
    TypeError
        If the input property array is not 1D or 2D
        If the neighborhood list does not match the number of vertices/elements in the property array.
        If the statistics parameter is not a string or a sequence of strings.
    
    ValueError
        If the statistics parameter contains an unsupported statistics name.
        
        
    Notes
    -----
    By default, the function is designed to compute statistics for properties defined at the vertices/elements of a mesh based on the neighborhood information for vertices/elements.
    In this case, the property array should have a shape of :math:`(N, P)`, where :math:`N` is the number of vertices/elements and :math:`P` is the number of properties per vertex/element. 
    The neighborhood list should be a list of sequences containing the indices of neighboring vertices/elements for each vertex/element in the mesh with a length equal to :math:`N`.
    The output will be a dictionary where each key is the name of the computed statistics and the corresponding value is an array of shape :math:`(N, P)` containing the computed statistics values for each vertex/element and each property.
    
    However, if the parameter :obj:`integrated_property` is set to :obj:`True`, the function can also accept properties defined at the integration points of a mesh and compute the specified statistics for each vertex/element based on the values of the property at the integration points of its neighboring elements.
    In this case, the property array should have a shape of :math:`(N_{ip}, P)`, where :math:`N_{ip}` is the total number of integration points in the mesh and :math:`P` is the number of properties per integration point.
    And the neighborhood list should be a list of sequences containing the indices of the integration points that belong to the neighboring elements for each vertex/element in the mesh with a length equal to :math:`N`, where :math:`N` is the number of vertices/elements in the mesh.
    The output will still be a dictionary where each key is the name of the computed statistics and the corresponding value is an array of shape :math:`(N, P)` containing the computed statistics values for each vertex/element and each property, but the computation will be based on the values of the property at the integration points of the neighboring elements for each vertex/element in the mesh.
        
    
    Examples
    --------
    Create a simple disconnected graph of 7 vertices and compute the mean value of a property defined at the vertices of the graph.
    
    .. figure:: ../_static/adjacency/graph_bfs.png
       :align: center
       :width: 50%
       
       Disconnected graph with 7 vertices for BFS example.
    
    .. code-block:: python
        :linenos:
        
        from pysdic import (
            compute_neighborhood,
            compute_neighborhood_statistics
        )
        
        graph = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3], [6], [5]]
        vertices_neighborhood = compute_neighborhood(graph, max_distance=2, self_neighborhood=True)
        
        # Define a property array with values for each vertex in the graph
        temperature = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        
        statistics = compute_neighborhood_statistics(
            property_array=temperature,
            neighborhood=vertices_neighborhood,
            statistics=["mean", "median", "min", "max", "std"]
        )
        
        print("Minimum temperature in the neighborhood of each vertex:")
        print(statistics["min"].reshape(-1)) # Reshape to 1D array for better display
        print("Maximum temperature in the neighborhood of each vertex:")
        print(statistics["max"].reshape(-1)) # Reshape to 1D array for better display
        print("Mean temperature in the neighborhood of each vertex:")
        print(statistics["mean"].reshape(-1)) # Reshape to 1D array for better display
        
    .. code-block:: console
    
        Minimum temperature in the neighborhood of each vertex:
        [10. 10. 10. 10. 30. 60. 60.]
        Maximum temperature in the neighborhood of each vertex:
        [40. 40. 50. 50. 50. 70. 70.]
        Mean temperature in the neighborhood of each vertex:
        [25. 25. 30. 30. 40. 65. 65.]

    """
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis]
    elif property_array.ndim != 2:
        raise ValueError("Input property array must be 1D or 2D.")
    
    if not isinstance(integrated_property, bool):
        raise TypeError(f"integrated_property must be a boolean. Got {type(integrated_property)}.")
    
    if not isinstance(neighborhood, list):
        raise TypeError("Neighborhood must be a list of sequences containing neighbor indices.")
    if not integrated_property and len(neighborhood) != property_array.shape[0]:
        raise ValueError("Neighborhood list length must match the number of vertices/elements in the property array.")
    
    if isinstance(statistics, str):
        statistics = [statistics]
    elif not isinstance(statistics, Sequence) or isinstance(statistics, str):
        raise TypeError("Statistic parameter must be a string or a sequence of strings specifying the statistics(s) to compute.")
    
    supported_statistics = {"mean", "median", "min", "max", "std", "var", "Q1", "Q3", "IQR"}
    for stat in statistics:
        if stat not in supported_statistics:
            raise ValueError(f"Unsupported statistics '{stat}'. Supported statistics are: {supported_statistics}.")
    
    # Compute the specified statistics for each vertex/element based on its neighbors
    n_points, n_properties = property_array.shape
    n_vertices = len(neighborhood)
    
    statistics_dict = {key: numpy.empty((n_vertices, n_properties), dtype=numpy.float64) for key in statistics}
    for i_vertex in range(n_vertices):
        neighbor_indices = neighborhood[i_vertex]
        
        if len(neighbor_indices) == 0:
            # If there are no neighbors, set the statistics to NaN for this vertex/element
            for key in statistics:
                statistics_dict[key][i_vertex, :] = numpy.nan
            continue
        
        neighbor_values = property_array[neighbor_indices, :]
        
        # Compute the specified statistics for the current vertex/element based on its neighbors
        if "mean" in statistics:
            statistics_dict["mean"][i_vertex, :] = numpy.mean(neighbor_values, axis=0)
        if "median" in statistics:
            statistics_dict["median"][i_vertex, :] = numpy.median(neighbor_values, axis=0)
        if "min" in statistics:
            statistics_dict["min"][i_vertex, :] = numpy.min(neighbor_values, axis=0)
        if "max" in statistics:
            statistics_dict["max"][i_vertex, :] = numpy.max(neighbor_values, axis=0)
        if "std" in statistics:
            statistics_dict["std"][i_vertex, :] = numpy.std(neighbor_values, axis=0)
        if "var" in statistics:
            statistics_dict["var"][i_vertex, :] = numpy.var(neighbor_values, axis=0)
        if "Q1" in statistics:
            statistics_dict["Q1"][i_vertex, :] = numpy.percentile(neighbor_values, 25, axis=0)
        if "Q3" in statistics:
            statistics_dict["Q3"][i_vertex, :] = numpy.percentile(neighbor_values, 75, axis=0)
        if "IQR" in statistics:
            Q1 = numpy.percentile(neighbor_values, 25, axis=0)
            Q3 = numpy.percentile(neighbor_values, 75, axis=0)
            statistics_dict["IQR"][i_vertex, :] = Q3 - Q1
    
    return statistics_dict
    


