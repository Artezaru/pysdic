.. currentmodule:: pysdic.geometry

pysdic.geometry.PointCloud3D
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

PointCloud3D class
-------------------------------------------

.. autoclass:: PointCloud3D

Instantiate a PointCloud3D object.
-------------------------------------------

To Instantiate a PointCloud3D object, use one of the following class methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.from_array
   PointCloud3D.from_cls
   PointCloud3D.from_empty

Accessing PointCloud3D attributes
-------------------------------------------

The attributes of a PointCloud3D object can be accessed as follows:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.points
   PointCloud3D.n_points
   PointCloud3D.shape

Manipulating PointCloud3D points
-------------------------------------------

The points of a PointCloud3D object can be manipulated using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.allclose
   PointCloud3D.as_array
   PointCloud3D.bounding_box
   PointCloud3D.concatenate
   PointCloud3D.concatenate_inplace
   PointCloud3D.copy
   PointCloud3D.frame_transform
   PointCloud3D.frame_transform_inplace
   PointCloud3D.keep_points
   PointCloud3D.keep_points_inplace
   PointCloud3D.keep_points_at
   PointCloud3D.keep_points_at_inplace
   PointCloud3D.merge
   PointCloud3D.merge_inplace
   PointCloud3D.remove_points
   PointCloud3D.remove_points_inplace
   PointCloud3D.remove_points_at
   PointCloud3D.remove_points_at_inplace
   PointCloud3D.unique
   PointCloud3D.unique_inplace

Operating on PointCloud3D objects
-------------------------------------------

The following methods can be used to operate on PointCloud3D objects:

- ``+`` operator: Concatenate two PointCloud3D objects.
- ``+=`` operator: In-place concatenation of two PointCloud3D objects.
- ``==`` operator: Check if two PointCloud3D objects are equal (based on their points).
- ``!=`` operator: Check if two PointCloud3D objects are not equal (based on their points).
- ``len()`` function: Get the number of points in a PointCloud3D object.

Visualizing PointCloud3D objects
-------------------------------------------

Visualizing a PointCloud3D object can be done using the following method:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.visualize

Examples of a simple PointCloud3D workflow
-------------------------------------------

Here is an example of a simple workflow using the PointCloud3D class:

First create a PointCloud3D object from a NumPy array:

.. code-block:: python

   import numpy
   from pysdic.geometry import PointCloud3D

   # Create a random NumPy array of shape (100, 3)
   points_array = numpy.random.rand(100, 3)

   # Instantiate a PointCloud3D object from the NumPy array
   point_cloud = PointCloud3D.from_array(points_array)

Now lets change the frame of reference of the point cloud by applying a translation:

.. code-block:: python

   from py3dframe import Frame

   # Define the actual frame of reference of the point cloud
   actual_frame = Frame.canonical()

   # Define a new frame of reference by translating the actual frame
   new_frame = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Translation by (1, 2, 3)

   # Transform the point cloud to the new frame of reference
   point_cloud = point_cloud.frame_transform(actual_frame, new_frame)

Now visualize the point cloud:

.. code-block:: python

   point_cloud.visualize()
