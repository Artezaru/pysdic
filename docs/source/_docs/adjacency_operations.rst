.. currentmodule:: pysdic

Operations on connectivity and adjacency graphs
==========================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: none

The package ``pysdic`` provides functions to handle connectivity and adjacency graphs commonly used in SDIC analyses.
This allows users to perform operations such as computing adjacency matrices, finding neighboring vertices or elements, and manipulating connectivity information for meshes.

A graph is a mathematical structure used to model pairwise relations between objects constructed from a set of vertices and edges (connections between vertices).
The distance between two vertices in a graph is defined as the length of the shortest path connecting them, where the length is measured in terms of the number of edges traversed.
The neighbors of a vertex are the vertices that are directly connected to it by an edge, while the neighborhood of a vertex includes all vertices that can be reached from it within a certain distance.

In this section:

- ``graph`` is a ``List[List[int]]`` containing for each vertex the list of its neighbors.
- ``adjacency_matrix`` is a Numpy or Scipy sparse square matrix of integers storing the distance between vertices in the graph.
- ``neighborhood`` is a list of Numpy arrays containing for each vertex the list of its neighbors within a certain distance.

Core algorithms (BFS)
-------------------------

The core algorithms for handling connectivity and adjacency graphs in ``pysdic`` are based on Breadth-First Search (BFS) techniques.
BFS is a graph traversal algorithm that explores the vertices of a graph in layers, starting from a given source vertex and visiting all its neighbors before moving on to the neighbors' neighbors.
This approach is efficient for finding neighboring vertices or elements in a mesh, as well as for constructing adjacency matrices.

.. autosummary::
   :toctree: ../_autosummary/

   bfs_distance
   bfs_neighborhood
   compute_adjacency_matrix
   compute_neighborhood


Convenience functions for adjacency and neighborhood in a mesh
---------------------------------------------------------------

Some functions in ``pysdic`` allow users to compute adjacency matrices and neighborhood from the connectivity of a mesh, which is a common representation in SDIC analyses.To 

.. autosummary::
   :toctree: ../_autosummary/

   create_vertices_adjacency_graph
   create_elements_adjacency_graph
   compute_vertices_adjacency_matrix
   compute_elements_adjacency_matrix
   compute_vertices_neighborhood
   compute_elements_neighborhood


Compute the statistics of a property according to the adjacency (mean, median, std, ...)
--------------------------------------------------------------------------------------------------

The property can be defined :

- at the vertices/elements of the mesh (ie the graph) and the statistics are computed according to the neighborhood of each vertex/element in the graph.
- at integration points inside the elements of the mesh and the statistics are computed according to the neighborhood of each element in the graph.

.. autosummary::
   :toctree: ../_autosummary/

   compute_neighborhood_statistics
