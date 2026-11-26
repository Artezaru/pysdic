r"""
.. _example_mesh_adjacency:

Adjacency and neighborhoods of a mesh
=====================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

The connectivity of a mesh defines which vertices and elements are neighbors:

- two vertices are neighbors if they belong to the same element;
- two elements are neighbors if they share at least one vertex.

The methods of :class:`pysdic.Mesh` give these relations as boolean sparse matrices, where
the neighbors of the vertex (or element) :math:`i` are ``matrix[i].indices``.

.. seealso::

    :func:`pysdic.fem.compute_vertices_distance_matrix` and
    :func:`pysdic.fem.compute_elements_distance_matrix` for the distances between all the
    pairs of vertices or elements.

"""

# %%
# A small mesh with special cases
# -------------------------------
#
# The mesh contains a closed tetrahedron (elements 0 to 3), an element sharing the edge
# 1-2 with it (a non-manifold edge), an isolated element and a vertex used by no element.

import numpy
from pysdic import Mesh

points = numpy.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  # Tetrahedron
    [1.0, 1.0, 1.0],                                                     # Non-manifold element
    [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0],                   # Isolated element
    [2.0, 2.0, 1.0],                                                     # Isolated vertex
])
elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4], [5, 6, 7]])
mesh = Mesh(points, elements, element_type="triangle_3")

mesh.quick_visualize(edge_color="black", vertex_size=10, vertex_color="red", face_opacity=0.2, show_bounds=False)

# %%
# Adjacency graphs
# ----------------
#
# The isolated vertex 8 has no neighbor, and the isolated element 5 has no neighbor.

vertices_graph = mesh.assemble_vertices_adjacency_graph()  # (N_v, N_v)
elements_graph = mesh.assemble_elements_adjacency_graph()  # (N_e, N_e)

for vertex in (1, 4, 8):
    print(f"Neighbors of vertex {vertex}: {vertices_graph[vertex].indices}")
for element in (4, 5):
    print(f"Neighbors of element {element}: {elements_graph[element].indices}")

# %%
# Neighborhoods
# -------------
#
# The neighborhood at a distance :math:`d` contains the vertices (or elements) reached in at
# most :math:`d` steps of the adjacency graph, here on a larger 2D mesh.

from pysdic import Plotter3D, create_2d_triangle_3_rectangle

grid = create_2d_triangle_3_rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), n_x=11, n_y=11)
vertex, element = 5 * 11 + 5, 100  # The central vertex and an element

vertices_neighborhood = grid.compute_vertices_neighborhood(max_distance=2)  # (N_v, N_v)
elements_neighborhood = grid.compute_elements_neighborhood(max_distance=1)  # (N_e, N_e)

vertex_mask = numpy.zeros(grid.n_vertices, dtype=bool)
vertex_mask[vertices_neighborhood[vertex].indices] = True
face_mask = numpy.zeros(grid.n_elements, dtype=bool)
face_mask[elements_neighborhood[element].indices] = True
print(f"{vertex_mask.sum()} vertices at a distance <= 2 of vertex {vertex}")
print(f"{face_mask.sum()} elements at a distance <= 1 of element {element}")

plotter = Plotter3D()
plotter.add_mesh(grid, face_color="white", edge_color="gray", show_vertices=False)
plotter.add_mesh(grid, face_mask=face_mask, face_color="lightblue", show_edges=False, show_vertices=False)
plotter.add_mesh(grid, vertex_mask=vertex_mask, vertex_color="red", vertex_size=10, show_faces=False, show_edges=False)
plotter.set_title("Neighborhoods of a vertex (red, d <= 2) and of an element (blue, d <= 1)")
plotter.plotter.view_xy()
plotter.show()