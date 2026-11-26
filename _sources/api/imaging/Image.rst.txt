pysdic.Image
============

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Image` stores a grayscale or color image and builds interpolation functions (one :class:`scipy.interpolate.RectBivariateSpline` per channel) to evaluate the image and its derivatives at any point of a region of interest (ROI).

.. important::

   Two coordinate systems are used: the **pixel points** :math:`(u, v)` (row, column) and the **image points** :math:`(x, y)` (column, row) of the camera. The spline degrees ``ku`` and ``kv`` refer to the :math:`u` and :math:`v` axes of the pixel points.


Image class
-----------

.. autoclass:: Image
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Image` class can be instantiated in several ways:

- **From an array** of shape :math:`(H, W)` or :math:`(H, W, C)` with the constructor or the classmethod :meth:`from_array`.
- **From a file** read with OpenCV using the classmethod :meth:`from_file` (color images in BGR order).

.. important::

   Any modification of ``image``, ``roi``, ``ku`` or ``kv`` resets the interpolation functions: call :meth:`construct_interpolation_functions` afterwards.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Image.from_array
   Image.from_file


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Image` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Image.default_value
   Image.dtype
   Image.height
   Image.image
   Image.is_color
   Image.is_grayscale
   Image.ku
   Image.kv
   Image.n_channels
   Image.ndim
   Image.roi
   Image.shape
   Image.width


Interpolation
^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Image.construct_interpolation_functions
   Image.deconstruct_interpolation_functions
   Image.evaluate_at_image_points
   Image.evaluate_at_pixel_points
   Image.get_interpolation_function


Pixel and image points
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Image.get_image_pixel_points
   Image.get_image_image_points
   Image.pixel_points_to_image_points
   Image.image_points_to_pixel_points


Conversion and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Image.copy
   Image.to_array
   Image.to_file
   Image.quick_visualize
