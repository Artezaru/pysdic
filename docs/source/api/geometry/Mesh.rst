pysdic.Mesh
===========

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Mesh` combines :math:`N_v` vertices (a :class:`pysdic.PointCloud`) and :math:`N_e` elements (a :class:`pysdic.Connectivity`). It gives access to the finite element tools of :doc:`pysdic.fem <../fem>` directly on the mesh (interpolation, derivatives, normals, ray casting, neighborhoods, ...).


Mesh class
----------

.. autoclass:: Mesh
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Mesh` class can be instantiated in several ways:

- **From the coordinates of the vertices and the elements** with the constructor (``Mesh(vertices, elements, element_type=...)``).
- **From other libraries**: ``meshio`` (:meth:`from_meshio`), ``pyvista`` (:meth:`from_pyvista`) or ``vtk`` (:meth:`from_vtk`, :meth:`from_vtk_file`).
- **With the mesh creation functions** (see :doc:`Geometry <../geometry>`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.from_meshio
   Mesh.from_pyvista
   Mesh.from_vtk
   Mesh.from_vtk_file


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Mesh` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.connectivity
   Mesh.coordinates
   Mesh.element_type
   Mesh.elements
   Mesh.n_dimensions
   Mesh.n_elements
   Mesh.n_topological_dimensions
   Mesh.n_vertices
   Mesh.n_vertices_per_element
   Mesh.uvmap
   Mesh.vertices


Structure methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.copy
   Mesh.add_connectivity
   Mesh.add_vertices
   Mesh.filter_connectivity
   Mesh.keep_connectivity_by_index
   Mesh.remove_connectivity_by_index
   Mesh.filter_vertices
   Mesh.keep_vertices_by_index
   Mesh.remove_vertices_by_index
   Mesh.get_used_vertices_mask
   Mesh.remove_unused_vertices
   Mesh.get_degenerated_elements_mask
   Mesh.remove_degenerated_elements


Checks
^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.validate
   Mesh.validate_integration_points


Operations at integration points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interpolate, differentiate and project the properties of the mesh at integration points (see :class:`pysdic.IntegrationPoints`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.compute_shape_functions
   Mesh.compute_shape_function_matrix
   Mesh.compute_jacobian_matrix
   Mesh.compute_integration_points_coordinates
   Mesh.compute_property_interpolation
   Mesh.compute_property_derivative
   Mesh.compute_property_projection
   Mesh.compute_strains


Geometry of the elements
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.compute_elements_centers
   Mesh.compute_elements_areas
   Mesh.compute_elements_normals
   Mesh.compute_vertices_normals
   Mesh.compute_integration_points_normals
   Mesh.cast_rays
   Mesh.locate_points


Adjacency and neighborhoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.assemble_vertices_adjacency_graph
   Mesh.assemble_elements_adjacency_graph
   Mesh.compute_vertices_neighborhood
   Mesh.compute_elements_neighborhood
   Mesh.compute_integration_points_neighborhood


Conversion and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Export to other libraries (the ``meshio``, ``pyvista`` and ``vtk`` methods require the corresponding optional packages).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.to_meshio
   Mesh.to_pyvista
   Mesh.to_vtk
   Mesh.to_vtk_file
   Mesh.quick_visualize


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.Mesh` is converted to a flat dictionary with ``to_dict`` (and created back with
``from_dict``). It inherits the file methods of the serialization mixins
(see :doc:`Core Tools <../core>`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Mesh.to_dict
   Mesh.from_dict
   Mesh.to_json_file
   Mesh.from_json_file
   Mesh.to_hdf5_file
   Mesh.from_hdf5_file
   Mesh.to_npz_file
   Mesh.from_npz_file
