.. currentmodule:: pysdic.geometry

pysdic.geometry.SurfaceMesh3D [Abstract Base Class]
====================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

SurfaceMesh3D class
-------------------------------------------

.. autoclass:: SurfaceMesh3D

Instantiate and export SurfaceMesh3D object
-------------------------------------------

The SurfaceMesh3D class is an ABC (Abstract Base Class) and cannot be instantiated directly.

By default, the meshes are created from a set of vertices and connectivity (see example below). 
The vertices are represented as a :class:`pysdic.geometry.PointCloud3D` object, and the connectivity is represented as a NumPy array of shape (M, K), 
where each row contains the indices of the vertices that form an element.

See the methods of the parent class :class:`pysdic.geometry.Mesh3D` for more details on how to instantiate and export SurfaceMesh3D objects.

Additional SurfaceMesh3D attributes
-------------------------------------------

For the common attributes inherited from :class:`pysdic.geometry.Mesh3D`, see the class documentation of :class:`pysdic.geometry.Mesh3D`.

An additional ``elements_property`` under the key ``"uvmap"`` can be used to store the UV mapping of the mesh. 
The UV mapping associates 2D coordinates at each vertex of each element in order to map a texture onto the 3D surface.
The property can be accessed using the following attribute :

.. autosummary::
   :toctree: ../generated/

    SurfaceMesh3D.elements_uvmap

Manipulating SurfaceMesh3D objects
-------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud3D`) and use its methods.
For the common methods inherited from :class:`pysdic.geometry.Mesh3D`, see the class documentation of :class:`pysdic.geometry.Mesh3D` for other inherited methods.

Visualize SurfaceMesh3D objects
-------------------------------------------

The SurfaceMesh3D class provides methods to visualize the mesh and its properties using PyVista.

.. autosummary::
   :toctree: ../generated/
   
    SurfaceMesh3D.is_visualizable
    SurfaceMesh3D.visualize
    SurfaceMesh3D.visualize_integration_points
    SurfaceMesh3D.visualize_texture
    SurfaceMesh3D.visualize_vertices_property