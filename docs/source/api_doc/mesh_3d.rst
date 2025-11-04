.. currentmodule:: pysdic.geometry

pysdic.geometry.Mesh3D [Abstract Base Class]
============================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Mesh3D class
-------------------------------------------

.. autoclass:: Mesh3D

Instantiate and export PointCloud3D object
-------------------------------------------

The Mesh3D class is an ABC (Abstract Base Class) and cannot be instantiated directly.

By default, the meshes are created from a set of vertices and connectivity (see example below). 
The vertices are represented as a :class:`pysdic.geometry.PointCloud3D` object, and the connectivity is represented as a NumPy array of shape (M, K), 
where each row contains the indices of the vertices that form an element.

The subclasses of Mesh3D can also use the following methods to instantiate a Mesh3D object from different file formats.

.. autosummary::
   :toctree: ../generated/

   Mesh3D.from_meshio
   Mesh3D.from_vtk
   Mesh3D.from_empty
   Mesh3D.from_mesh

The Mesh3D class provides methods to export the mesh to various file formats.

.. autosummary::
   :toctree: ../generated/

    Mesh3D.to_meshio
    Mesh3D.to_vtk


Accessing Mesh3D attributes
-------------------------------------------

The  public attributes of a Mesh3D object can be accessed using the following properties:

.. autosummary::
   :toctree: ../generated/

    Mesh3D.internal_bypass
    Mesh3D.connectivity
    Mesh3D.elements
    Mesh3D.n_vertices
    Mesh3D.n_elements
    Mesh3D.n_vertices_per_element
    Mesh3D.n_dimensions
    Mesh3D.meshio_cell_type
    Mesh3D.vertices


Manage vertices and elements properties
-------------------------------------------

Several properties can be associated with the vertices and elements of the mesh.
These properties can be accessed and modified using the following methods:

.. autosummary::
   :toctree: ../generated/

    Mesh3D.clear_elements_properties
    Mesh3D.clear_vertices_properties
    Mesh3D.clear_properties
    Mesh3D.get_elements_property
    Mesh3D.get_vertices_property
    Mesh3D.remove_elements_property
    Mesh3D.remove_vertices_property
    Mesh3D.set_elements_property
    Mesh3D.set_vertices_property
    Mesh3D.list_elements_properties
    Mesh3D.list_vertices_properties


Add, remove or modify vertices or connectivity of the Mesh3D objects
--------------------------------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud3D`) and use its methods.

The topology of the mesh can be modified using the following methods:

.. autosummary::
   :toctree: ../generated/

    Mesh3D.add_elements
    Mesh3D.add_vertices
    Mesh3D.are_used_vertices
    Mesh3D.is_empty
    Mesh3D.keep_elements
    Mesh3D.remove_elements
    Mesh3D.remove_unused_vertices
    Mesh3D.remove_vertices



Mesh3D geometric computations and interpolations
-------------------------------------------------

Some methods are provided to perform geometric computations and property interpolations on Mesh3D objects:

.. autosummary::
   :toctree: ../generated/

    Mesh3D.copy
    Mesh3D.get_vertices_coordinates
    Mesh3D.interpolate_property_at_natural_coordinates
    Mesh3D.interpolate_property_at_integration_points
    Mesh3D.integration_points_to_global_coordinates
    Mesh3D.natural_to_global_coordinates
    Mesh3D.project_integration_property_to_vertices
    Mesh3D.shape_functions
    Mesh3D.shape_functions_matrix
    Mesh3D.validate


