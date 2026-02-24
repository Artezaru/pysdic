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
   :toctree: ../_autosummary/

   PointCloud.from_array
   PointCloud.from_meshio
   PointCloud.from_vtk
   PointCloud.from_npz

The :class:`PointCloud` can then be exported to different formats using the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.to_array
   PointCloud.to_meshio
   PointCloud.to_vtk
   PointCloud.to_npz


Accessing PointCloud attributes
-------------------------------------------

The public attributes of a :class:`PointCloud` object can be accessed using the following properties:

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.coordinates
   PointCloud.points
   PointCloud.n_dimensions
   PointCloud.n_points
   PointCloud.shape



Add and manage PointCloud points properties
-------------------------------------------

The properties of the points in a :class:`PointCloud` object can be managed using the following methods:

.. note::

   Point properties are stored as named NumPy arrays of shape :math:`(N_p, A)` where :math:`N_p` is the number of points and :math:`A` is the number of property components (e.g., 3 for RGB color).
   All the point properties must have the same number of points as the :class:`PointCloud` object and are stored as :obj:`numpy.float64` arrays.

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.set_property
   PointCloud.get_property
   PointCloud.has_property
   PointCloud.delete_property
   PointCloud.list_properties
   PointCloud.copy_properties
   PointCloud.clear_properties


Add, remove or modify points in PointCloud objects
-----------------------------------------------------

The points of a :class:`PointCloud` object can be manipulated using the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.all_close
   PointCloud.all_finite
   PointCloud.concatenate
   PointCloud.copy
   PointCloud.filter_points
   PointCloud.frame_transform
   PointCloud.is_finite
   PointCloud.is_nan
   PointCloud.keep_points
   PointCloud.remove_not_finite
   PointCloud.remove_points


PointCloud object geometric computations
-------------------------------------------

The following methods can be used to perform geometric computations on :class:`PointCloud` objects:

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.bounding_box
   PointCloud.bounding_sphere


Visualize PointCloud object (1D, 2D, 3D only)
----------------------------------------------

The :class:`PointCloud` class provides a method to visualize the point cloud in 1D, 2D, or 3D space using ``pyvista``:

.. autosummary::
   :toctree: ../_autosummary/

   PointCloud.visualize

















