.. currentmodule:: pysdic.geometry

pysdic.geometry.QuadraticTriangleMesh3D
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

QuadraticTriangleMesh3D class
-------------------------------------------

.. autoclass:: QuadraticTriangleMesh3D

Instantiate a QuadraticTriangleMesh3D object
---------------------------------------------

The QuadraticTriangleMesh3D is a subclass of Mesh3D and can be instantiated directly or using the methods inherited from :class:`pysdic.geometry.Mesh3D` and :class:`pysdic.geometry.SurfaceMesh3D`.

The QuadraticTriangleMesh3D class can also be instantiated from an :class:`pysdic.geometry.LinearTriangleMesh3D` object using the class method :meth:`from_lineartriangle`.

.. autosummary::
   :toctree: ../generated/

    QuadraticTriangleMesh3D.from_lineartriangle
    QuadraticTriangleMesh3D.to_lineartriangle

Manipulating QuadraticTriangleMesh3D objects
---------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud3D`) and use its methods.
For the common methods inherited from :class:`pysdic.geometry.Mesh3D`, see the class documentation of :class:`pysdic.geometry.Mesh3D` for other inherited methods.
For the common methods inherited from :class:`pysdic.geometry.SurfaceMesh3D`, see the class documentation of :class:`pysdic.geometry.SurfaceMesh3D` for other inherited methods.

The QuadraticTriangleMesh3D class also provides the following additional methods:

.. autosummary::
   :toctree: ../generated/

    QuadraticTriangleMesh3D.shape_functions

Visualize QuadraticTriangleMesh3D objects
-------------------------------------------

To visualize the mesh and its properties, use the methods inherited from :class:`pysdic.geometry.SurfaceMesh3D`.


Examples of a simple QuadraticTriangleMesh3D workflow
-------------------------------------------------------

Creating a QuadraticTriangleMesh3D from vertices and connectivity:

.. code-block:: python

   import numpy
   from pysdic.geometry import QuadraticTriangleMesh3D

   vertices = numpy.array([
      [0.0, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 0.5, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.5],
      [0.0, 0.0, 1.0],
      [0.5, 0.5, 0.0],
      [0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5],
   ])

   connectivity = numpy.array([
      [0, 2, 4, 1, 7, 3],
      [0, 2, 6, 1, 8, 5],
      [0, 4, 6, 3, 9, 5],
      [2, 4, 6, 7, 9, 8],
   ])

   mesh = QuadraticTriangleMesh3D(
      vertices=vertices,
      connectivity=connectivity,
   )

Visualizing the mesh:

.. code-block:: python

   mesh.visualize()

Set a displacement property on the vertices and visualize it:

.. code-block:: python

   displacement = numpy.array([
      [0.0, 0.0, 0.0],
      [0.1, 0.0, 0.0],
      [0.0, 0.1, 0.0],
      [0.05, 0.0, 0.0],
      [0.0, 0.05, 0.0],
      [0.0, 0.0, 0.1],
      [0.0, 0.0, 0.05],
      [0.05, 0.05, 0.0],
      [0.05, 0.0, 0.05],
      [0.0, 0.05, 0.05],
   ])

   mesh.set_vertices_property("displacement", displacement)

   mesh.visualize_vertices_property("displacement")