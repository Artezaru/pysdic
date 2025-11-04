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

Instantiate and export PointCloud3D object
-------------------------------------------

To Instantiate a PointCloud3D object, use one of the following class methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.from_array
   PointCloud3D.from_cls
   PointCloud3D.from_empty
   PointCloud3D.from_obj
   PointCloud3D.from_ply
   PointCloud3D.from_vtk
   PointCloud3D.from_xyz
   

The PointCloud3D can then be exported to different formats using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.as_array
   PointCloud3D.to_array
   PointCloud3D.to_obj
   PointCloud3D.to_ply
   PointCloud3D.to_vtk
   PointCloud3D.to_xyz

Accessing PointCloud3D attributes
-------------------------------------------

The public attributes of a PointCloud3D object can be accessed using the following properties:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.coordinates
   PointCloud3D.points
   PointCloud3D.n_points
   PointCloud3D.shape

Add, remove or modify points in PointCloud3D objects
-----------------------------------------------------

The points of a PointCloud3D object can be manipulated using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.allclose
   PointCloud3D.concatenate
   PointCloud3D.copy
   PointCloud3D.frame_transform
   PointCloud3D.keep_points
   PointCloud3D.keep_points_at
   PointCloud3D.merge
   PointCloud3D.remove_points
   PointCloud3D.remove_points_at
   PointCloud3D.unique

Operations on PointCloud3D objects
-------------------------------------------

The following methods can be used to operate on PointCloud3D objects:

- ``+`` operator: Concatenate two PointCloud3D objects.
- ``+=`` operator: In-place concatenation of two PointCloud3D objects.
- ``len()`` function: Get the number of points in a PointCloud3D object.

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.__add__
   PointCloud3D.__iadd__
   PointCloud3D.__len__


PointCloud3D object geometric computations
-------------------------------------------

The following methods can be used to perform geometric computations on PointCloud3D objects:

.. autosummary::
   :toctree: ../generated/

   PointCloud3D.bounding_box
   PointCloud3D.bounding_sphere


Visualize PointCloud3D objects
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
