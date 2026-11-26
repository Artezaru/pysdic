pysdic.View
===========

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.View` associates a :class:`pysdic.Camera` and a :class:`pysdic.Image` acquired by this camera, to project world points to the gray levels (or colors) of the image with their Jacobians.

.. note::

   Several views can share the same camera or the same image: sharing an object shares its state (parameters of the camera, data and interpolation functions of the image).


View class
----------

.. autoclass:: View
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`View` class is instantiated with its constructor ``View(camera, image)``. The height and width of the image must be equal to the sensor size of the camera. The camera and the image cannot be changed after the creation of the view.


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.View` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   View.camera
   View.image


Projection to the image
^^^^^^^^^^^^^^^^^^^^^^^

The world points are projected with :meth:`pysdic.Camera.project`, then the image and its gradient are evaluated at the image points. The result is a :class:`pysdic.ImageProjectionResult`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   View.image_project


Visualization
^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   View.quick_visualize
