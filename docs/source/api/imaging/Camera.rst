pysdic.Camera
=============

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Camera` is a :class:`pysdic.ComposedTransformation` of an extrinsic, a distortion and an intrinsic transformation, :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}`, associated with the size of its sensor in pixels.


Camera class
------------

.. autoclass:: Camera
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Camera` class can be instantiated in two ways:

- **From the sensor size and the transformations** with the constructor: ``Camera(sensor_height, sensor_width, intrinsic=..., distortion=..., extrinsic=...)``. The missing transformations are replaced by :class:`pysdic.NoIntrinsic`, :class:`pysdic.NoDistortion` and :class:`pysdic.NoExtrinsic`.
- **From a composition** ``[extrinsic, distortion, intrinsic]`` (e.g. built with :func:`pysdic.compose_projection`) using the classmethod :meth:`from_composition`.

.. important::

   The sensor size and the three transformations cannot be changed after the creation of the camera. Only their parameters can be modified, with ``camera.parameters`` or the parameters of each transformation (they are shared, not copied).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Camera.from_composition


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Camera` class.


Attributes
^^^^^^^^^^

The parameters of the camera are inherited from :class:`pysdic.ComposedTransformation`:

- ``parameters``: the concatenation ``[extrinsic, distortion, intrinsic]`` of the parameters of the three transformations.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Camera.distortion
   Camera.extrinsic
   Camera.intrinsic
   Camera.sensor_height
   Camera.sensor_width


Projection and rays
^^^^^^^^^^^^^^^^^^^

The ``apply`` method of :class:`pysdic.Transformation` maps the **world points** to the **image points**. The following methods give a more convenient access to the projection, the Jacobians of each transformation, and the normalized points and rays of the pixels (cached).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Camera.project
   Camera.get_camera_pixel_points
   Camera.get_camera_normalized_points
   Camera.get_camera_rays


Visualization
^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Camera.quick_visualize
