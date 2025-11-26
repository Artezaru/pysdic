API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Geometry Submodule
------------------

The submodule ``pysdic.geometry`` contains objects and functions to manipulate geometrical entities as 3-dimensional points and meshes.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.geometry classes

   ./api_doc/geometry_classes/point_cloud.rst
   ./api_doc/geometry_classes/mesh_ABC.rst
   ./api_doc/geometry_classes/surface_mesh_ABC.rst
   ./api_doc/geometry_classes/implemented_meshes.rst

.. toctree::
   :maxdepth: 1
   :caption: pysdic.geometry functions

   ./api_doc/geometry_functions/shape_functions.rst
   ./api_doc/geometry_functions/integrated_points_operations.rst

Some utility functions to create specific meshes are also provided:

.. toctree::
   :maxdepth: 1
   :caption: pysdic.geometry utility functions

   ./api_doc/create_linear_triangle_axisymmetric.rst
   ./api_doc/create_linear_triangle_heightmap.rst


Operators Submodule
---------------------

The submodule ``pysdic.operators`` contains objects and functions to build operators used in SDIC computations, such as derivation, integration, and interpolation operators.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.operators classes

   ./api_doc/operators_functions/derivation_operator.rst