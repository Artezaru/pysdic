pysdic.ImagePlotter
===================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.ImagePlotter` draws images, the sensor of a camera and the geometrical objects of ``pysdic`` (:class:`pysdic.PointCloud`, :class:`pysdic.Mesh`, :class:`pysdic.IntegrationPoints`) in the **image coordinate system** with ``matplotlib``. Its API mirrors the one of :class:`pysdic.Plotter3D`.

The objects are projected with the camera when their dimension is the input dimension of the camera (``projected=None``), or drawn directly as image points :math:`(x, y)` when they are 2D (see the ``projected`` argument).


ImagePlotter class
------------------

.. autoclass:: ImagePlotter
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`ImagePlotter` class is instantiated with a source and optional figure settings:

- **From a camera**: ``ImagePlotter(camera)``, to project the objects.
- **From a view**: ``ImagePlotter(view)``, to project the objects and draw the image of the view by default.
- **From a sensor size**: ``ImagePlotter((height, width))``, to draw only image points.

The ``ax`` argument draws in existing matplotlib axes (e.g. a subplot of a larger figure).


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.ImagePlotter` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ImagePlotter.ax
   ImagePlotter.camera
   ImagePlotter.figure
   ImagePlotter.sensor_height
   ImagePlotter.sensor_width


Sensor and image
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ImagePlotter.add_sensor
   ImagePlotter.add_image


Geometrical objects
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ImagePlotter.add_point_cloud
   ImagePlotter.add_mesh
   ImagePlotter.add_integration_points
   ImagePlotter.add_rectangle


Figure methods
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ImagePlotter.set_title
   ImagePlotter.clear
   ImagePlotter.show
   ImagePlotter.close
