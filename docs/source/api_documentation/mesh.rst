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
The vertices are represented as a :class:`pysdic.PointCloud` object, and the connectivity is represented as a NumPy array of shape (:math:`N_e`, :math:`N_{vpe}`),
where each row contains the indices of the vertices that form an element and :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

:class:`Mesh` can also use the following methods to instantiate a :class:`Mesh` object from different file formats.

.. autosummary::
   :toctree: ../generated/

   Mesh.from_meshio
   Mesh.from_npz
   Mesh.from_vtk

The :class:`Mesh` class provides methods to export the mesh to various file formats.

.. autosummary::
   :toctree: ../generated/

   Mesh.to_meshio
   Mesh.to_npz
   Mesh.to_vtk


Accessing Mesh attributes
-------------------------------------------

The public attributes of a :class:`Mesh` object can be accessed using the following properties:

.. autosummary::
   :toctree: ../generated/

   Mesh.internal_bypass
   Mesh.connectivity
   Mesh.elements
   Mesh.elements_type
   Mesh.elements_properties
   Mesh.n_vertices
   Mesh.N_v
   Mesh.n_elements
   Mesh.N_e
   Mesh.n_vertices_per_element
   Mesh.N_vpe
   Mesh.n_dimensions
   Mesh.E
   Mesh.n_topological_dimensions
   Mesh.K
   Mesh.meshio_cell_type
   Mesh.set_elements_type
   Mesh.vertices
   Mesh.vtk_cell_type


Manage vertices and elements properties
-------------------------------------------

Several properties can be associated with the vertices and elements of the mesh.
These properties can be accessed and modified using the following methods:

.. autosummary::
   :toctree: ../generated/

   Mesh.clear_elements_properties
   Mesh.clear_vertices_properties
   Mesh.clear_properties
   Mesh.get_elements_property
   Mesh.get_vertices_property
   Mesh.remove_elements_property
   Mesh.remove_vertices_property
   Mesh.set_elements_property
   Mesh.set_vertices_property
   Mesh.list_elements_properties
   Mesh.list_vertices_properties


For texture visualization of surface meshes, UV mapping can be accessed and modified using the :attr:`elements_uvmap` property.

.. autosummary::
   :toctree: ../generated/

   Mesh.elements_uvmap


Add, remove or modify vertices or connectivity of the Mesh objects
--------------------------------------------------------------------

To manipulate only the geometry of the mesh, access the :obj:`vertices` attribute (:class:`pysdic.PointCloud`) and use its methods.

The topology of the mesh can be modified using the following methods:

.. autosummary::
   :toctree: ../generated/

   Mesh.add_elements
   Mesh.add_vertices
   Mesh.are_used_vertices
   Mesh.is_empty
   Mesh.keep_elements
   Mesh.remove_elements
   Mesh.remove_unused_vertices
   Mesh.remove_vertices



Mesh geometric computations and interpolations
-------------------------------------------------

Some methods are provided to perform geometric computations and property interpolations on :class:`Mesh` objects:

.. autosummary::
   :toctree: ../generated/

   Mesh.copy
   Mesh.shape_functions
   Mesh.validate


.. seealso::

    - :func:`pysdic.construct_jacobian` : Function to construct the Jacobian matrix for coordinate transformations using shape functions.
    - :func:`pysdic.interpolate_property` : Function to interpolate properties within elements using shape functions.
    - :func:`pysdic.project_property_to_vertices` : Function to project properties from integration points back to element nodes using shape functions.



Visualize surface meshes
-------------------------------------------

The package ``pysdic`` provides functions to visualize 3D surface meshes using the Pyvista library.

.. autosummary::
   :toctree: ../generated/

   Mesh.visualize
   Mesh.visualize_elements_property
   Mesh.visualize_integration_points
   Mesh.visualize_texture
   Mesh.visualize_vertices_property


