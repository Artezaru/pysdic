API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Manipulate meshes and integration points
-----------------------------------------

The package ``pysdic`` provides classes to store and manipulate geometrical entities such as point clouds, meshes, and integration points.
The objective is to facilitate the handling of these entities in Stereo Digital Image Correlation (SDIC) analyses and building operators to solve the underlying optimization problems.

.. toctree::
   :maxdepth: 1
   :caption: Shape Functions and Integration Points

   ./api_documentation/shape_functions.rst
   ./api_documentation/gauss_points.rst
   ./api_documentation/integration_points_operations.rst

.. toctree::
   :maxdepth: 1
   :caption: Objects to store PointClouds, Meshes and Integration Points

   ./api_documentation/point_cloud.rst
   ./api_documentation/mesh.rst
   ./api_documentation/integration_points.rst
   ./api_documentation/create_3D_surface_meshes.rst

.. toctree::
   :maxdepth: 1
   :caption: Operate on Specific Meshes

   ./api_documentation/triangle_3_meshes_operations.rst


Image Processing and camera manipulations
-----------------------------------------

The package ``pysdic`` provides functions to handle images and camera models commonly used in SDIC analyses.
This allows users to project 3D points onto 2D image planes, undistort images, and perform other camera-related operations.

.. seealso::

    - Package ``pycvcam`` for advanced camera model manipulations and calibrations.

.. toctree::
   :maxdepth: 1
   :caption: Objects and Functions to Handle Images, Cameras and Views

   ./api_documentation/image.rst
   ./api_documentation/camera.rst
   ./api_documentation/view.rst
   ./api_documentation/projection_result.rst
   ./api_documentation/image_projection_result.rst


Regularize displacements
-----------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Temporal Derivation Operators

   ./api_documentation/temporal_derivation.rst


Implement Photometric considerations
-----------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Photometric Quantities and Lighting Models

   ./api_documentation/photometric_quantities.rst
   ./api_documentation/BRDF_models.rst


Submodule pysdic.build - Building Operators for SDIC
-------------------------------------------------------------------------

The submodule ``pysdic.sdic`` provides functions to build operators used in Stereo Digital Image Correlation (SDIC) analyses.
These operators are essential for formulating and solving the optimization problems that arise in SDIC.

.. toctree::
   :maxdepth: 1
   :caption: Building Operators for SDIC

   ./api_documentation/build_displacement_operator.rst


Submodule pysdic.blender - Integrating with Blender
-------------------------------------------------------------------------

The submodule ``pysdic.blender`` provides classes and functions to generate and manipulate 3D scenes in Blender.
This integration allows users to visualize and analyze SDIC results within the Blender environment.

.. toctree::
   :maxdepth: 1
   :caption: Blender Integration

   ./api_documentation/blender_experiment.rst
   ./api_documentation/blender_camera.rst
   ./api_documentation/blender_spotlight.rst
   ./api_documentation/blender_mesh.rst
   ./api_documentation/blender_material_bsdf.rst