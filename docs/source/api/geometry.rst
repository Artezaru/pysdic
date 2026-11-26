Geometry (Point Clouds, Meshes and Integration Points)
======================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

The package ``pysdic`` provides objects to store and process the geometry of the observed
surfaces: point clouds, meshes made of vertices and elements, and integration points located
inside the elements. These objects wrap the array-based functions of
:doc:`pysdic.fem <fem>` and can be converted to and from ``meshio``, ``pyvista`` and ``vtk``.


Definition of Notations
-----------------------

- :math:`E`: the dimension of the space (e.g. 2 or 3).
- :math:`N_v`, :math:`N_e`: the number of vertices and elements of a mesh.
- :math:`N_{vpe}`: the number of vertices per element.
- :math:`K`: the topological dimension of the elements (1 for segments, 2 for triangles and quadrangles).
- :math:`N_p`: the number of points (of a point cloud or a set of integration points).
- :math:`\Xi = (\xi, \eta, ...)`: the natural coordinates of a point in the reference element.

The supported element types (``segment_2``, ``segment_3``, ``triangle_3``, ``triangle_6``,
``quadrangle_4``, ``quadrangle_8``) are described in :doc:`pysdic.fem <fem>`.

.. note::

   All the objects store their arrays as copies and return read-only views. They can carry
   **properties** (one array of shape :math:`(N, \dots)` per name), accessible with
   ``obj.properties`` or directly with ``obj[name]`` (see :doc:`Core Tools <core>`), and can be
   saved to JSON, HDF5 and NPZ files.


Geometrical objects
-------------------

- :doc:`PointCloud <geometry/PointCloud>`: A cloud of :math:`N_p` points in a space of dimension :math:`E`, with properties for each point.
- :doc:`Connectivity <geometry/Connectivity>`: The :math:`N_e` elements of a mesh (vertex indices), with an element type and properties for each element.
- :doc:`Mesh <geometry/Mesh>`: A mesh made of vertices (:class:`pysdic.PointCloud`) and elements (:class:`pysdic.Connectivity`), with the finite element operations, the geometry of the elements, the ray casting and the neighborhoods.
- :doc:`IntegrationPoints <geometry/IntegrationPoints>`: :math:`N_p` points located in the elements of a mesh by their element index and natural coordinates, with weights and properties.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./geometry/PointCloud.rst
   ./geometry/Connectivity.rst
   ./geometry/Mesh.rst
   ./geometry/IntegrationPoints.rst


Creating 3D surface meshes
--------------------------

Create structured surface meshes on a grid of :math:`(n_1, n_2)` vertices, with a UV map for texturing.
The ``n_*`` arguments are numbers of **vertices**, and the surfaces can be placed in any local frame
with the ``frame`` argument (requires the optional package ``py3dframe``).

- :doc:`create_uniform_quadrangle_4_mesh <geometry/create_uniform_quadrangle_4_mesh>`: ``quadrangle_4`` mesh of a given grid of vertices of shape :math:`(n_x, n_y, E)`, optionally connected along each direction.
- :doc:`create_3d_quadrangle_4_heightmap <geometry/create_3d_quadrangle_4_heightmap>`: ``quadrangle_4`` mesh of a height surface :math:`z = f(x, y)`.
- :doc:`create_3d_triangle_3_heightmap <geometry/create_3d_triangle_3_heightmap>`: ``triangle_3`` mesh of a height surface :math:`z = f(x, y)`.
- :doc:`create_3d_quadrangle_4_cylindrical <geometry/create_3d_quadrangle_4_cylindrical>`: ``quadrangle_4`` mesh of a cylindrical surface :math:`r = f(h, \theta)`.
- :doc:`create_3d_triangle_3_cylindrical <geometry/create_3d_triangle_3_cylindrical>`: ``triangle_3`` mesh of a cylindrical surface :math:`r = f(h, \theta)`, optionally closed on the axis.
- :doc:`create_3d_quadrangle_4_spherical <geometry/create_3d_quadrangle_4_spherical>`: ``quadrangle_4`` mesh of a spherical surface :math:`r = f(\theta, \phi)`.
- :doc:`create_3d_triangle_3_spherical <geometry/create_3d_triangle_3_spherical>`: ``triangle_3`` mesh of a spherical surface :math:`r = f(\theta, \phi)`, optionally closed at the poles.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./geometry/create_uniform_quadrangle_4_mesh.rst
   ./geometry/create_3d_quadrangle_4_heightmap.rst
   ./geometry/create_3d_triangle_3_heightmap.rst
   ./geometry/create_3d_quadrangle_4_cylindrical.rst
   ./geometry/create_3d_triangle_3_cylindrical.rst
   ./geometry/create_3d_quadrangle_4_spherical.rst
   ./geometry/create_3d_triangle_3_spherical.rst


Creating 2D surface meshes
--------------------------

Create structured meshes of planar domains (:math:`E = 2`), e.g. for a planar analysis with a camera of the plane or
a camera observing the plane :math:`Z_w = 0` (see :doc:`Implemented Extrinsic <implemented_extrinsic>`).

- :doc:`create_2d_quadrangle_4_rectangle <geometry/create_2d_quadrangle_4_rectangle>`: ``quadrangle_4`` mesh of a rectangle.
- :doc:`create_2d_triangle_3_rectangle <geometry/create_2d_triangle_3_rectangle>`: ``triangle_3`` mesh of a rectangle.
- :doc:`create_2d_quadrangle_4_polar <geometry/create_2d_quadrangle_4_polar>`: ``quadrangle_4`` mesh of a polar domain (disk, annulus or sector).
- :doc:`create_2d_triangle_3_polar <geometry/create_2d_triangle_3_polar>`: ``triangle_3`` mesh of a polar domain (disk, annulus or sector).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./geometry/create_2d_quadrangle_4_rectangle.rst
   ./geometry/create_2d_triangle_3_rectangle.rst
   ./geometry/create_2d_quadrangle_4_polar.rst
   ./geometry/create_2d_triangle_3_polar.rst


Converting meshes
-----------------

- :doc:`quadrangle_4_to_triangle_3 <geometry/quadrangle_4_to_triangle_3>`: Split each quadrangle of a ``quadrangle_4`` mesh into two triangles (``"first"``, ``"second"``, ``"alternate"`` or ``"random"`` diagonal), with the properties and the UV map.
- :doc:`triangle_3_to_triangle_6 <geometry/triangle_3_to_triangle_6>`: Convert a ``triangle_3`` mesh into a quadratic ``triangle_6`` mesh (a vertex added at the middle of each edge).
- :doc:`quadrangle_4_to_quadrangle_8 <geometry/quadrangle_4_to_quadrangle_8>`: Convert a ``quadrangle_4`` mesh into a serendipity ``quadrangle_8`` mesh (a vertex added at the middle of each edge).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./geometry/quadrangle_4_to_triangle_3.rst
   ./geometry/triangle_3_to_triangle_6.rst
   ./geometry/quadrangle_4_to_quadrangle_8.rst


Visualization
-------------

- :doc:`Plotter3D <geometry/Plotter3D>`: Display point clouds, meshes (with properties, textures and vector fields) and integration points with ``pyvista``.

.. important::

   The visualization requires the optional package ``pyvista`` (``pip install pyvista``).
   The ``quick_visualize`` methods of :class:`pysdic.PointCloud` and :class:`pysdic.Mesh`
   give a fast preview without creating a plotter.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./geometry/Plotter3D.rst