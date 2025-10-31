.. currentmodule:: pysdic.geometry

pysdic.geometry.LinearTriangleMesh3D
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

LinearTriangleMesh3D class
-------------------------------------------

.. autoclass:: LinearTriangleMesh3D

Instantiate a LinearTriangleMesh3D object
-------------------------------------------

The LinearTriangleMesh3D is a subclass of Mesh3D and can be instantiated directly or using the methods inherited from :class:`pysdic.geometry.Mesh3D`.

The LinearTriangleMesh3D class can also be instantiated from an Open3D TriangleMesh object using the class method :meth:`from_open3d`.

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.from_open3d
    LinearTriangleMesh3D.to_open3d

Additional LinearTriangleMesh3D attributes
-------------------------------------------

For the common attributes inherited from :class:`pysdic.geometry.Mesh3D`, see the class documentation of :class:`pysdic.geometry.Mesh3D`.

An additional ``elements_property`` under the key ``"uvmap"`` can be used to store the UV mapping of the mesh. The property can be accessed using the following attribute :

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.elements_uvmap


Manipulating LinearTriangleMesh3D objects
-------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud3D`) and use its methods.
For the common methods inherited from :class:`pysdic.geometry.Mesh3D`, see the class documentation of :class:`pysdic.geometry.Mesh3D` for other inherited methods.

The LinearTriangleMesh3D class also provides the following additional methods:

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.cast_rays
    LinearTriangleMesh3D.compute_elements_areas
    LinearTriangleMesh3D.compute_elements_normals
    LinearTriangleMesh3D.compute_vertices_normals
    LinearTriangleMesh3D.shape_functions

Visualize LinearTriangleMesh3D objects
-------------------------------------------

The LinearTriangleMesh3D class provides methods to visualize the mesh and its properties using PyVista.

.. autosummary::
   :toctree: ../generated/
   
    LinearTriangleMesh3D.visualize
    LinearTriangleMesh3D.visualize_integration_points
    LinearTriangleMesh3D.visualize_texture
    LinearTriangleMesh3D.visualize_vertices_property


Examples of a simple PointCloud3D workflow
-------------------------------------------

Creating a LinearTriangleMesh3D from vertices and connectivity:

.. code-block:: python

   import numpy
   from pysdic.geometry import LinearTriangleMesh3D

   vertices = numpy.array([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
   ])

   connectivity = numpy.array([
      [0, 1, 2],
      [0, 1, 3],
      [0, 2, 3],
      [1, 2, 3],
   ])

   mesh = LinearTriangleMesh3D.from_vertices_and_connectivity(
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
      [0.0, 0.0, 0.1],
   ])

   mesh.set_vertices_property("displacement", displacement)

   mesh.visualize_vertices_property("displacement")

Casting rays on the mesh:

.. code-block:: python

   # Define some rays
   origins = numpy.array([
      [2.0, 2.0, 2.0],
      [2.2, 2.0, 2.0],
      [2.0, 2.2, 2.0],
      [2.2, 2.2, 2.0],
   ])

   directions = numpy.array([
      [-1.0, -1.0, -1.0],
      [-1.0, -1.0, -0.8],
      [-1.0, -0.8, -1.0],
      [-1.0, -0.8, -0.8],
   ])

   # Cast rays on the mesh
   intersection_points = mesh.cast_rays(origins, directions)

   print("Intersection points:", intersection_points)

Interpolating a vertices property at the intersection_points:

.. code-block:: python

   displacement_interpolated = mesh.interpolate_property_at_integration_points(
      intersection_points,
      property_key="displacement",
   )

   print("Interpolated displacement at intersection points:", displacement_interpolated)

