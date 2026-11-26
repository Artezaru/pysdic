Finite Element Tools (``pysdic.fem``)
=====================================

.. currentmodule:: pysdic.fem

.. contents:: Table of Contents
   :local:
   :depth: 1

The submodule ``pysdic.fem`` provides low-level, array-based functions to work with finite element meshes:
shape functions, Gauss quadrature, interpolation and differentiation of nodal properties, assembly of
operators, geometric quantities, ray casting and adjacency of the elements.

All the functions work directly on numpy arrays (no mesh object is needed), so they can be reused with
any mesh data structure.


Definition of Notations
-----------------------

A mesh is defined in a space of dimension :math:`E` by :math:`N_v` vertices and :math:`N_e` elements of
topological dimension :math:`K \leq E`, each element having :math:`N_{vpe}` vertices:

- ``vertices_coordinates``: The coordinates of the vertices with shape :math:`(N_v, E)`.
- ``connectivity_array``: The vertex indices of each element with shape :math:`(N_e, N_{vpe})`.
- ``element_type``: The type of the elements (see the table below).

The points inside the elements (integration points) are given by:

- ``natural_coordinates``: The coordinates :math:`\Xi = (\xi, \eta, ...)` of the points in the reference element with shape :math:`(N_p, K)`.
- ``element_indices``: The index of the element containing each point with shape :math:`(N_p,)`.
  The index ``-1`` marks a point not located in any element (accepted with ``allow_unlocated_points=True``).

A property defined at the vertices (displacements, gray levels, ...) has shape :math:`(N_v, P)`.
Its value at a point :math:`\Xi` of an element is interpolated with the shape functions :math:`N_i(\Xi)`:

.. math::

   f(\Xi) = \sum_{i=1}^{N_{vpe}} N_i(\Xi) \, f_i


Supported element types
-----------------------

+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``element_type`` | Description           | :math:`K`   | :math:`N_{vpe}`       | ``meshio`` type   | VTK cell type     |
+==================+=======================+=============+=======================+===================+===================+
| ``segment_2``    | Linear segment        | 1           | 2                     | ``line``          | 3                 |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``segment_3``    | Quadratic segment     | 1           | 3                     | ``line3``         | 21                |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``triangle_3``   | Linear triangle       | 2           | 3                     | ``triangle``      | 5                 |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``triangle_6``   | Quadratic triangle    | 2           | 6                     | ``triangle6``     | 22                |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``quadrangle_4`` | Bilinear quadrangle   | 2           | 4                     | ``quad``          | 9                 |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+
| ``quadrangle_8`` | Serendipity quadrangle| 2           | 8                     | ``quad8``         | 23                |
+------------------+-----------------------+-------------+-----------------------+-------------------+-------------------+

The ``element_type`` arguments are case insensitive.


Shape functions
---------------

Compute the shape functions :math:`N_i(\Xi)` (and optionally their derivatives :math:`\frac{\partial N_i}{\partial \Xi}`)
at given natural coordinates in the reference element.

- :doc:`compute_shape_functions <fem/compute_shape_functions>`: Shape functions for any supported element type.
- :doc:`compute_segment_2_shape_functions <fem/compute_segment_2_shape_functions>`: Shape functions of the 2-node segment.
- :doc:`compute_segment_3_shape_functions <fem/compute_segment_3_shape_functions>`: Shape functions of the 3-node segment.
- :doc:`compute_triangle_3_shape_functions <fem/compute_triangle_3_shape_functions>`: Shape functions of the 3-node triangle.
- :doc:`compute_triangle_6_shape_functions <fem/compute_triangle_6_shape_functions>`: Shape functions of the 6-node triangle.
- :doc:`compute_quadrangle_4_shape_functions <fem/compute_quadrangle_4_shape_functions>`: Shape functions of the 4-node quadrangle.
- :doc:`compute_quadrangle_8_shape_functions <fem/compute_quadrangle_8_shape_functions>`: Shape functions of the 8-node (serendipity) quadrangle.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/compute_shape_functions.rst
   ./fem/compute_segment_2_shape_functions.rst
   ./fem/compute_segment_3_shape_functions.rst
   ./fem/compute_triangle_3_shape_functions.rst
   ./fem/compute_triangle_6_shape_functions.rst
   ./fem/compute_quadrangle_4_shape_functions.rst
   ./fem/compute_quadrangle_8_shape_functions.rst


Gauss quadrature
----------------

Get the natural coordinates :math:`\Xi_g` and the weights :math:`w_g` of the Gauss points of an element type,
to integrate a function over the reference element:
:math:`\int_{\hat{\Omega}} f(\Xi) \, d\Xi \approx \sum_g w_g f(\Xi_g)`.

- :doc:`get_gauss_points <fem/get_gauss_points>`: Gauss points for any supported element type.
- :doc:`get_segment_2_gauss_points <fem/get_segment_2_gauss_points>`: 1-point rule of the 2-node segment.
- :doc:`get_segment_3_gauss_points <fem/get_segment_3_gauss_points>`: 2-point rule of the 3-node segment.
- :doc:`get_triangle_3_gauss_points <fem/get_triangle_3_gauss_points>`: 1-point rule of the 3-node triangle.
- :doc:`get_triangle_6_gauss_points <fem/get_triangle_6_gauss_points>`: 3-point rule of the 6-node triangle.
- :doc:`get_quadrangle_4_gauss_points <fem/get_quadrangle_4_gauss_points>`: :math:`2 \times 2` rule of the 4-node quadrangle.
- :doc:`get_quadrangle_8_gauss_points <fem/get_quadrangle_8_gauss_points>`: :math:`3 \times 3` rule of the 8-node quadrangle.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/get_gauss_points.rst
   ./fem/get_segment_2_gauss_points.rst
   ./fem/get_segment_3_gauss_points.rst
   ./fem/get_triangle_3_gauss_points.rst
   ./fem/get_triangle_6_gauss_points.rst
   ./fem/get_quadrangle_4_gauss_points.rst
   ./fem/get_quadrangle_8_gauss_points.rst


Operations on the mesh
----------------------

Interpolate, differentiate and project the properties of the mesh at integration points.
Each operation comes in two versions:

- ``assemble_*``: from precomputed shape functions (and derivatives), to reuse them for several properties.
- ``compute_*``: directly from the mesh and the integration points (the shape functions are computed internally).

Shape function matrix and Jacobian of the elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`assemble_shape_function_matrix <fem/assemble_shape_function_matrix>`: Assemble the shape function matrix :math:`\mathcal{N}` with shape :math:`(N_p, N_v)` (dense or sparse).
- :doc:`compute_shape_function_matrix <fem/compute_shape_function_matrix>`: Compute and assemble the shape function matrix :math:`\mathcal{N}`.
- :doc:`remap_vertices_coordinates <fem/remap_vertices_coordinates>`: Coordinates of the vertices of the element of each integration point, with shape :math:`(N_p, N_{vpe}, E)`.
- :doc:`assemble_jacobian_matrix <fem/assemble_jacobian_matrix>`: Assemble the Jacobian :math:`J = \frac{\partial X}{\partial \Xi}` with shape :math:`(N_p, E, K)`.
- :doc:`compute_jacobian_matrix <fem/compute_jacobian_matrix>`: Compute and assemble the Jacobian :math:`J = \frac{\partial X}{\partial \Xi}`.

Interpolation, derivative and projection of a property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`assemble_property_interpolation <fem/assemble_property_interpolation>`: Interpolate a nodal property at integration points from precomputed shape functions.
- :doc:`compute_property_interpolation <fem/compute_property_interpolation>`: Interpolate a nodal property at integration points.
- :doc:`assemble_property_derivative <fem/assemble_property_derivative>`: Derivative of a nodal property with respect to the global coordinates from precomputed derivatives and Jacobians.
- :doc:`compute_property_derivative <fem/compute_property_derivative>`: Derivative of a nodal property with respect to the global coordinates at integration points.
- :doc:`assemble_property_projection <fem/assemble_property_projection>`: Project a property defined at integration points back to the vertices from a precomputed shape function matrix.
- :doc:`compute_property_projection <fem/compute_property_projection>`: Project a property defined at integration points back to the vertices.

Operators for the correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`assemble_displacement_jacobian_matrix <fem/assemble_displacement_jacobian_matrix>`: Assemble the operator :math:`\mathbf{J}` mapping the nodal displacements to the variations of a property (e.g. gray levels) at integration points, :math:`\mathbf{J} \, dU = dR`.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/assemble_shape_function_matrix.rst
   ./fem/compute_shape_function_matrix.rst
   ./fem/remap_vertices_coordinates.rst
   ./fem/assemble_jacobian_matrix.rst
   ./fem/compute_jacobian_matrix.rst
   ./fem/assemble_property_interpolation.rst
   ./fem/compute_property_interpolation.rst
   ./fem/assemble_property_derivative.rst
   ./fem/compute_property_derivative.rst
   ./fem/assemble_property_projection.rst
   ./fem/compute_property_projection.rst
   ./fem/assemble_displacement_jacobian_matrix.rst


Geometry and mechanics
----------------------

Compute geometric and mechanical quantities of the elements from their Jacobian matrices
(any element type, linear or quadratic).

- :doc:`compute_elements_centers <fem/compute_elements_centers>`: Center of each element.
- :doc:`compute_elements_areas <fem/compute_elements_areas>`: Area of each surface element (:math:`K = 2`), integrated with the Gauss points.
- :doc:`compute_integration_points_normals <fem/compute_integration_points_normals>`: Unit normal of a 3D surface mesh at integration points.
- :doc:`compute_elements_normals <fem/compute_elements_normals>`: Unit normal of each element of a 3D surface mesh, at its center.
- :doc:`compute_vertices_normals <fem/compute_vertices_normals>`: Unit normal at each vertex of a 3D surface mesh (mean of the normals of the elements weighted by their areas).
- :doc:`compute_strains <fem/compute_strains>`: Linear strain tensor :math:`\boldsymbol{\varepsilon} = \frac{1}{2} (\nabla U + \nabla U^T)` of a displacement field at integration points.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/compute_elements_centers.rst
   ./fem/compute_elements_areas.rst
   ./fem/compute_integration_points_normals.rst
   ./fem/compute_elements_normals.rst
   ./fem/compute_vertices_normals.rst
   ./fem/compute_strains.rst


Location of points and ray casting
----------------------------------

Locate points in a 2D surface mesh (:math:`K = 2`, :math:`E = 2`), or compute the first intersection of rays with a 3D surface mesh (:math:`K = 2`, :math:`E = 3`), for example the rays
of a camera computed with :func:`pysdic.compute_rays`. The intersections are returned as natural coordinates
and element indices (``-1`` for the rays without intersection).

.. important::

   The ray casting requires the optional package ``open3d`` (``pip install open3d``). The location of points in a 2D mesh
   has no optional dependency.

- :doc:`locate_points <fem/locate_points>`: Element and natural coordinates of points in a 2D surface mesh of any element type (exact for ``triangle_3``, first approximation on subdivided elements refined with Newton iterations otherwise).
- :doc:`cast_rays <fem/cast_rays>`: Intersection with a surface mesh of any element type (first approximation on subdivided elements, refined with Newton iterations on the exact surface).
- :doc:`triangle_3_cast_rays <fem/triangle_3_cast_rays>`: Intersection with a ``triangle_3`` mesh.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/locate_points.rst
   ./fem/cast_rays.rst
   ./fem/triangle_3_cast_rays.rst


Adjacency and neighborhoods
---------------------------

Build the adjacency graphs of a mesh and compute the shortest path distances and the neighborhoods of
its vertices, elements and integration points:

- two vertices are neighbors (distance 1) if they belong to the same element;
- two elements are neighbors (distance 1) if they share at least one vertex.

Adjacency graphs
^^^^^^^^^^^^^^^^

- :doc:`assemble_vertices_adjacency_graph <fem/assemble_vertices_adjacency_graph>`: Adjacency graph of the vertices of a mesh.
- :doc:`assemble_elements_adjacency_graph <fem/assemble_elements_adjacency_graph>`: Adjacency graph of the elements of a mesh.

Distances
^^^^^^^^^

- :doc:`bfs_distance <fem/bfs_distance>`: Shortest path distances from a starting node to all the nodes of a graph.
- :doc:`compute_distance_matrix <fem/compute_distance_matrix>`: Shortest path distances between all the pairs of nodes of a graph.
- :doc:`compute_vertices_distance_matrix <fem/compute_vertices_distance_matrix>`: Shortest path distances between all the pairs of vertices of a mesh.
- :doc:`compute_elements_distance_matrix <fem/compute_elements_distance_matrix>`: Shortest path distances between all the pairs of elements of a mesh.

Neighborhoods
^^^^^^^^^^^^^

- :doc:`bfs_neighborhood <fem/bfs_neighborhood>`: Nodes within a given distance from a starting node of a graph.
- :doc:`compute_neighborhood <fem/compute_neighborhood>`: Neighborhood of each node of a graph.
- :doc:`compute_vertices_neighborhood <fem/compute_vertices_neighborhood>`: Neighborhood of each vertex of a mesh.
- :doc:`compute_elements_neighborhood <fem/compute_elements_neighborhood>`: Neighborhood of each element of a mesh.
- :doc:`remap_integration_points_neighborhood <fem/remap_integration_points_neighborhood>`: Remap a neighborhood of the elements to the integration points located in these elements.
- :doc:`compute_integration_points_neighborhood <fem/compute_integration_points_neighborhood>`: Integration points located in the neighborhood of each element.
- :doc:`compute_neighborhood_statistics <fem/compute_neighborhood_statistics>`: Statistics of values over the neighborhood of each node.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./fem/assemble_vertices_adjacency_graph.rst
   ./fem/assemble_elements_adjacency_graph.rst
   ./fem/bfs_distance.rst
   ./fem/compute_distance_matrix.rst
   ./fem/compute_vertices_distance_matrix.rst
   ./fem/compute_elements_distance_matrix.rst
   ./fem/bfs_neighborhood.rst
   ./fem/compute_neighborhood.rst
   ./fem/compute_vertices_neighborhood.rst
   ./fem/compute_elements_neighborhood.rst
   ./fem/remap_integration_points_neighborhood.rst
   ./fem/compute_integration_points_neighborhood.rst
   ./fem/compute_neighborhood_statistics.rst
