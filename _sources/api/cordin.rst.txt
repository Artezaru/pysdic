CORDIN Captures (``pysdic.cordin``)
===================================

.. currentmodule:: pysdic.cordin

.. contents:: Table of Contents
   :local:
   :depth: 1

The submodule ``pysdic.cordin`` reads the description of the captures of the CORDIN high-speed cameras (rotating
mirror cameras): the parameters of the capture, the details of the camera and the frames acquired by each module
(sensor) of the camera. The functions are called from their namespace:

.. code-block:: python

   from pysdic.cordin import CORDIN

   capture = CORDIN("capture.xml")
   frame = capture.get_frame(module=3)     # The frame acquired by the module 3
   module = capture.get_module(frame=12)   # The module that acquired the frame 12

.. seealso::

   :func:`pysdic.create_cordin_analysis` to build the analysis of a CORDIN capture: one sensor seen through several
   points of view, with a distortion changing with the frame (rotating mirror).


Capture
-------

- :doc:`CORDIN <cordin/CORDIN>`: A CORDIN capture read from its XML file, with the correspondence between the frames and the modules of the camera.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./cordin/CORDIN.rst


Reading functions
-----------------

The low-level functions used by :class:`CORDIN`, to read the XML file of a capture directly.

- :doc:`load_capture_info <cordin/load_capture_info>`: Load the XML description of a capture.
- :doc:`get_capture_parameters <cordin/get_capture_parameters>`: The parameters of the capture (rate, number of frames, ...).
- :doc:`get_camera_details <cordin/get_camera_details>`: The details of the camera (model, modules, ...).
- :doc:`get_frames <cordin/get_frames>`: The frames of the capture with their module.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./cordin/load_capture_info.rst
   ./cordin/get_capture_parameters.rst
   ./cordin/get_camera_details.rst
   ./cordin/get_frames.rst