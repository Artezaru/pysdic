.. currentmodule:: pysdic

Mesh structures
==================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top



Mesh class
-------------------------------------------

.. autoclass:: Mesh

Instantiate and export Mesh object
-------------------------------------------

By default, the meshes are created from a set of vertices and connectivity. 
The vertices are represented as a :class:`pysdic.PointCloud` object, and the connectivity is represented as a :class:`pysdic.Connectivity` object.
:class:`Mesh` can also use the following methods to instantiate a :class:`Mesh` object from different file formats.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.from_meshio
   Mesh.from_npz
   Mesh.from_vtk

The :class:`Mesh` class provides methods to export the mesh to various file formats.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.to_meshio
   Mesh.to_npz
   Mesh.to_vtk


Accessing Mesh attributes
-------------------------------------------

The public attributes of a :class:`Mesh` object can be accessed using the following properties:

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.connectivity
   Mesh.vertices

For easier access to the vertices and connectivity information, the following properties are also available and are equivalent to the corresponding properties of the :class:`pysdic.PointCloud` and :class:`pysdic.Connectivity` objects:

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.coordinates
   Mesh.elements
   Mesh.element_type
   Mesh.n_dimensions
   Mesh.n_elements
   Mesh.n_points
   Mesh.n_topological_dimensions
   Mesh.n_vertices
   Mesh.n_vertices_per_element
   Mesh.points

To ensure the integrity of the mesh, the following properties can be used:

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.copy
   Mesh.validate

Add, remove or modify vertices or connectivity of the Mesh objects
--------------------------------------------------------------------

The topology of the mesh can be modified using the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.add_connectivity
   Mesh.add_vertices
   Mesh.get_used_vertices_mask
   Mesh.filter_connectivity
   Mesh.filter_vertices
   Mesh.keep_connectivity
   Mesh.keep_vertices
   Mesh.remove_connectivity
   Mesh.remove_unused_vertices
   Mesh.remove_vertices

Accessing and managing vertices and connectivity properties
--------------------------------------------------------------------

To access and manage the properties of vertices and connectivity, use the :meth:`Mesh.vertices` and :meth:`Mesh.connectivity` objects, which provide methods to compute and store properties related to the vertices and connectivity of the mesh.
The vertices and connectivity properties are common to all meshes build on the same set of vertices and connectivity, and they are stored in :class:`PointCloud` and :class:`Connectivity` objects, respectively.
The properties can be accessed and managed using the following methods:

.. code-block:: python

   # Example of accessing and managing vertices and connectivity properties
   from pysdic import Mesh

   # Create a simple mesh (this is just an example, replace with actual vertices and connectivity)
   vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
   connectivity = [[0, 1, 2]]
   mesh = Mesh(vertices=vertices, connectivity=connectivity)

   # Access vertices properties
   intensity = [1, 2, 3]  # Example property for vertices
   mesh.vertices.set_property('intensity', intensity)
   retrieved_intensity = mesh.vertices.get_property('intensity')

   # Access connectivity properties
   area = [0.5]  # Example property for connectivity
   mesh.connectivity.set_property('area', area)
   retrieved_area = mesh.connectivity.get_property('area')
   

Precomputed quantities of the mesh
--------------------------------------------------------------------

To perform geometric computations and property interpolations on the mesh, some quantities can be precomputed and stored in the mesh for later use.
The precomputed quantities of the mesh are stored in a dictionary, and they can be accessed and managed using the following methods:

.. warning::

   If any modification is made to the vertices or connectivity of the mesh, the precomputed properties may become invalid and should be cleared using :meth:`Mesh.clear_precomputed` to avoid using outdated properties in subsequent computations.
   It is the responsibility of the user to ensure that the precomputed properties are consistent with the current state of the mesh, especially after any modifications to the vertices or connectivity.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.get_precomputed
   Mesh.list_precomputed
   Mesh.clear_precomputed

The method with :obj:`precompute` argument allows to compute a quantity and store it in the precomputed properties of the mesh for later use (or into :class:`IntegrationPoints`).
Set to :obj:`True` to store the computed quantity in the precomputed properties of the mesh, and set to :obj:`False` to compute the quantity without storing it in the precomputed properties of the mesh.

The precomputed quantities available are :

- ``vertices_adjacency_matrix``
- ``elements_adjacency_matrix``
- ``vertices_neighborhood``
- ``elements_neighborhood``

The mesh can be used to precompute quantities for :class:`IntegrationPoints` as well such as:

- ``integration_points_neighborhood``
- ``shape_functions``
- ``shape_function_derivatives``


Compute connectivity and adjacency graphs
--------------------------------------------------------------------

The connectivity and adjacency graphs of a mesh can be computed using the following methods, which are based on Breadth-First Search (BFS) techniques.
The connectivity graph represents the direct connections between vertices or elements in the mesh, while the adjacency graph represents the distances between vertices or elements in the mesh based on the connectivity.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.compute_vertices_neighborhood
   Mesh.compute_elements_neighborhood
   Mesh.compute_vertices_adjacency_matrix
   Mesh.compute_elements_adjacency_matrix

Once computed and stored in the precomputed properties of the mesh, the neighborhood and adjacency graphs can be used to compute statistics of a property according to the adjacency (mean, median, std, ...), as well as to perform geometric computations and property interpolations on the mesh.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.compute_vertices_neighborhood_statistics
   Mesh.compute_elements_neighborhood_statistics


Mesh geometric computations and interpolations
---------------------------------------------------------------------

Some methods are provided to perform geometric computations and property interpolations on :class:`Mesh` objects:

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.compute_shape_functions
   Mesh.compute_property_interpolation
   Mesh.compute_property_projection
   Mesh.compute_property_derivative


Visualize surface meshes
-------------------------------------------

The package ``pysdic`` provides functions to visualize 3D surface meshes using the Pyvista library.
The ``element_type`` of the mesh must be set.
Furthermore, before visualizing a mesh with a texture, make sure to have the ``uvmap`` defined.

.. autosummary::
   :toctree: ../_autosummary/

   Mesh.uvmap
   Mesh.visualize


