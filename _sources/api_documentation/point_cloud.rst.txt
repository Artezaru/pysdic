.. currentmodule:: pysdic

PointCloud structures
==================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

PointCloud class
-------------------------------------------

.. autoclass:: PointCloud

Instantiate and export PointCloud object
--------------------------------------------------

To Instantiate a :class:`PointCloud` object, use one of the following class methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud.from_array
   PointCloud.from_meshio
   PointCloud.from_npz
   PointCloud.from_obj
   PointCloud.from_ply
   PointCloud.from_vtk
   PointCloud.from_xyz

   
The :class:`PointCloud` can then be exported to different formats using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud.to_array
   PointCloud.to_meshio
   PointCloud.to_npz
   PointCloud.to_obj
   PointCloud.to_ply
   PointCloud.to_vtk
   PointCloud.to_xyz


Accessing PointCloud attributes
-------------------------------------------

The public attributes of a :class:`PointCloud` object can be accessed using the following properties:

.. autosummary::
   :toctree: ../generated/

   PointCloud.coordinates
   PointCloud.points
   PointCloud.n_dimensions
   PointCloud.n_points
   PointCloud.shape


Add, remove or modify points in PointCloud objects
-----------------------------------------------------

The points of a :class:`PointCloud` object can be manipulated using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud.all_close
   PointCloud.all_finite
   PointCloud.concatenate
   PointCloud.copy
   PointCloud.frame_transform
   PointCloud.is_finite
   PointCloud.is_nan
   PointCloud.keep_points
   PointCloud.keep_points_at
   PointCloud.merge
   PointCloud.remove_not_finite
   PointCloud.remove_points
   PointCloud.remove_points_at
   PointCloud.unique

Change the dimension of PointCloud objects
-----------------------------------------------------

The dimension of a :class:`PointCloud` object can be changed using the following methods:

.. autosummary::
   :toctree: ../generated/

   PointCloud.extend_as_hyperplane
   PointCloud.extend_n_dimensions
   PointCloud.reduce_to_hyperplane


Operations on PointCloud objects
-------------------------------------------

The following methods can be used to operate on :class:`PointCloud` objects:

- ``+`` operator: Concatenate two :class:`PointCloud` objects.
- ``+=`` operator: In-place concatenation of two :class:`PointCloud` objects.
- ``len()`` function: Get the number of points in a :class:`PointCloud` object.

.. autosummary::
   :toctree: ../generated/

   PointCloud.__add__
   PointCloud.__iadd__
   PointCloud.__len__


PointCloud object geometric computations
-------------------------------------------

The following methods can be used to perform geometric computations on :class:`PointCloud` objects:

.. autosummary::
   :toctree: ../generated/

   PointCloud.bounding_box
   PointCloud.bounding_sphere


Visualize PointCloud object (1D, 2D, 3D only)
----------------------------------------------

The :class:`PointCloud` class provides a method to visualize the point cloud in 1D, 2D, or 3D space using ``pyvista``:

.. autosummary::
   :toctree: ../generated/

   PointCloud.visualize


Example of a simple PointCloud workflow
-------------------------------------------

Here is an example of a simple workflow using the :class:`PointCloud` class:

First create a :class:`PointCloud` object from a NumPy array:

.. code-block:: python

   import numpy
   from pysdic import PointCloud

   # Create a random NumPy array of shape (100, 3)
   points_array = numpy.random.rand(100, 3)

   # Instantiate a :class:`PointCloud` object from the NumPy array
   point_cloud = PointCloud.from_array(points_array)

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


















