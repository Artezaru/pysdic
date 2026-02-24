Package Objects API Reference
=============================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Geometrical and Physical Objects
--------------------------------

The ``pysdic`` package provides a collection of classes for storing point clouds,
meshes, and integration points.  These objects are the building blocks for many
SDIC analyses.

- :doc:`PointCloud <_docs/point_cloud>`: A class to store and manipulate point clouds, which are collections of points that can represent surfaces or volumes. 
- :doc:`Connectivity <_docs/connectivity>`: A class to store and manipulate connectivity information for meshes, such as adjacency matrices and element connectivity. 
- :doc:`Mesh <_docs/mesh>`: A class to store and manipulate meshes, which consist of a combination of :class:`PointCloud` and :class:`Connectivity` objects, along with additional attributes for storing mesh-specific information. 
- :doc:`IntegrationPoints <_docs/integration_points>`: A class to store and manipulate integration points, which are used for numerical integration in finite element analyses.

.. toctree::
   :maxdepth: 1
   :caption: Objects to Store Point Clouds, Meshes and Integration Points
   :hidden:

   _docs/point_cloud
   _docs/connectivity
   _docs/mesh
   _docs/integration_points


Some default mesh creation functions are also provided to create meshes for common geometries (See :doc:`Mesh Creation Functions <_docs/mesh_creation>`).

.. toctree::
   :maxdepth: 1
   :caption: Default Mesh Creation Functions
   :hidden:

   _docs/mesh_creation

Image Processing and Camera Manipulations
-----------------------------------------

To perform stereo digital image correlation analyses, the ``pysdic`` package provides classes to handle images, cameras, and views, as well as functions to perform projections and store projection results. 

- :doc:`Image <_docs/image>`: A class to store and manipulate images, which are 2D arrays of pixel values that can represent the visual information captured by cameras. 
- :doc:`Camera <_docs/camera>`: A class to store and manipulate camera parameters, such as intrinsic and extrinsic parameters, which define the camera's position, orientation, and optical properties. 
- :doc:`View <_docs/view>`: A class to store and manipulate views, which consist of a combination of :class:`Camera` and :class:`Image` objects, along with additional attributes for storing view-specific information.
- :doc:`ProjectionResult <_docs/projection_result>`: A class to store the results of projecting 3D points onto 2D image planes using camera parameters. 
- :doc:`ImageProjectionResult <_docs/image_projection_result>`: A class to store the results of projecting 3D points onto 2D image planes along with the corresponding pixel values from the images.

.. seealso::

   - The ``pycvcam`` package for camera‑model manipulations and calibrations. The quantities stored in the :class:`Camera` class are objects from the ``pycvcam`` package.

.. toctree::
   :maxdepth: 1
   :caption: Objects and Functions to Handle Images, Cameras and Views
   :hidden:

   _docs/image
   _docs/camera
   _docs/view
   _docs/projection_result
   _docs/image_projection_result


