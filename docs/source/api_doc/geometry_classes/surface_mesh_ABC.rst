.. currentmodule:: pysdic.geometry
   

SurfaceMesh Structure [Abstract Base Class]
====================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


SurfaceMesh class
-------------------------------------------

.. autoclass:: SurfaceMesh

Instantiate and export SurfaceMesh object
-------------------------------------------

The SurfaceMesh class is an ABC (Abstract Base Class) and cannot be instantiated directly.

By default, the meshes are created from a set of vertices and connectivity (see example below). 
The vertices are represented as a :class:`pysdic.geometry.PointCloud` object, and the connectivity is represented as a NumPy array of shape (:math:`N_e`, :math:`N_{vpe}`),
where each row contains the indices of the vertices that form an element.

See the methods of the parent class :class:`pysdic.geometry.Mesh` for more details on how to instantiate and export SurfaceMesh objects.

Additional SurfaceMesh attributes
-------------------------------------------

For the common attributes inherited from :class:`pysdic.geometry.Mesh`, see the class documentation of :class:`pysdic.geometry.Mesh`.

An additional ``elements_property`` under the key ``"uvmap"`` can be used to store the UV mapping of the mesh. 
The UV mapping associates 2D coordinates at each vertex of each element in order to map a texture onto the surface.
The property can be accessed using the following attribute :

.. autosummary::
   :toctree: ../../generated/geometry_classes/

   SurfaceMesh.elements_uvmap

Manipulating SurfaceMesh objects
-------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud`) and use its methods.
For the common methods inherited from :class:`pysdic.geometry.Mesh`, see the class documentation of :class:`pysdic.geometry.Mesh` for other inherited methods.

Visualize SurfaceMesh objects
-------------------------------------------

The SurfaceMesh class provides methods to visualize the mesh and its properties using PyVista for meshes with embedded dimension :math:`E \leq 3`.
   
.. autosummary::
   :toctree: ../../generated/geometry_classes/
   
   SurfaceMesh.visualize
   SurfaceMesh.visualize_integration_points
   SurfaceMesh.visualize_texture
   SurfaceMesh.visualize_vertices_property