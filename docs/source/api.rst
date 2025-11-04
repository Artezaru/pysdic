API Reference
==============

The package ``pysdic`` is composed of the following functions, classes, and modules:

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Geometry Submodule
------------------

The submodule ``pysdic.geometry`` contains objects and functions to manipulate geometrical entities as 3-dimensional points and meshes.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.geometry class module

   ./api_doc/point_cloud_3d.rst
   ./api_doc/mesh_3d.rst
   ./api_doc/linear_triangle_mesh_3d.rst
   ./api_doc/integration_points.rst

Some utility functions to create specific meshes are also provided:

.. toctree::
   :maxdepth: 1
   :caption: pysdic.geometry utility functions

   ./api_doc/create_linear_triangle_axisymmetric.rst
   ./api_doc/create_linear_triangle_heightmap.rst

Imaging Submodule
------------------

The submodule ``pysdic.imaging`` contains objects and functions to project 3D geometries into 2D images and vice versa.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.imaging class module

   ./api_doc/camera.rst
   ./api_doc/image.rst
   ./api_doc/view.rst

The projection results are stored into specialized objects:

.. toctree::
   :maxdepth: 1
   :caption: pysdic.imaging results dataclasses

   ./api_doc/projection_result.rst
   ./api_doc/image_projection_result.rst


Assemblers Submodule
--------------------

The submodule ``pysdic.assemblers`` contains functions to assemble the residuals and Jacobians for different SDIC measurement types.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.assemblers functions

   ./api_doc/assembly_FESDIC_displacement.rst
   ./api_doc/assembly_SDIC_distortion.rst


Visualizer Submodule
--------------------

The submodule ``pysdic.visualizer`` contains objects and functions to visualize 3D geometries and 2D images into ``pyqt5`` windows.

.. toctree::
   :maxdepth: 1
   :caption: pysdic.visualizer utility functions

   ./api_doc/visualize_qt_pyvista_linear_triangle_mesh_3d.rst