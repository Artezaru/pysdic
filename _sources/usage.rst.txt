Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

The package ``pysdic`` provides a collection of examples demonstrating how to use the package, from the geometry and
the cameras to a complete SDIC measurement. The examples are grouped in galleries, one per part of the package. Each
example can be downloaded as a Python script or a Jupyter notebook.

.. note::

   Some examples need optional packages (``open3d`` for the ray casting, ``pyvista`` for the 3D displays, ...): they
   are given at the top of each example (see :doc:`Installation <installation>`).


Galleries
---------

.. grid:: 2

   .. grid-item-card::
      :img-top: /_static/_icons/geometry.png
      :text-align: center

      Geometry
      ^^^

      Point clouds, meshes and integration points: the creation of 2D and 3D surface meshes, the geometry of the
      surfaces, the adjacency of the elements and the ray casting.

      +++

      .. button-ref:: /_gallery/geometry/index
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the geometry examples

   .. grid-item-card::
      :img-top: /_static/_icons/projection.png
      :text-align: center

      Computer vision
      ^^^

      The camera models: the projection of points, the distortion of images, the optical flow, the calibration
      of the parameters and the triangulation.

      +++

      .. button-ref:: /_gallery/computer_vision/index
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the computer vision examples

   .. grid-item-card::
      :img-top: /_static/_icons/camera.png
      :text-align: center

      Imaging
      ^^^

      Cameras, images and views: the projection of points and meshes on a camera, the projection of the gray
      levels of an image, the drawings of the image plotter and the synthetic images of a cylinder.

      +++

      .. button-ref:: /_gallery/imaging/index
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the imaging examples

   .. grid-item-card::
      :img-top: /_static/_icons/pysdic_icon.png
      :text-align: center

      SDIC
      ^^^

      The SDIC analysis: the measurement of a displacement, the optimization of the cameras and the finite
      difference operators of the temporal regularization.

      +++

      .. button-ref:: /_gallery/sdic/index
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the SDIC examples

   .. grid-item-card::
      :img-top: /_static/_icons/examples.png
      :text-align: center

      Workflows
      ^^^

      The procedures of :mod:`pysdic.workflows`: the integration points, the initialization by optical flow, the
      shape at the reference frame, the tools on the displacements and the quality control.

      +++

      .. button-ref:: /_gallery/workflows/index
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the workflows examples

   .. grid-item-card::
      :img-top: /_static/_icons/blender.png
      :text-align: center

      Blender rendering
      ^^^

      The rendering of synthetic images with lighting and materials, with the separate package
      ``pysdic-blender`` (see its reference page for an example).

      +++

      .. button-ref:: api/blender
         :ref-type: doc
         :expand:
         :color: secondary
         :click-parent:

         To the Blender rendering guide

The galleries follow the order of a first SDIC analysis: :doc:`Geometry </_gallery/geometry/index>` (the surface),
:doc:`Computer vision </_gallery/computer_vision/index>` (the camera models), :doc:`Imaging </_gallery/imaging/index>`
(the cameras and the images), :doc:`SDIC </_gallery/sdic/index>` (the measurement), then
:doc:`Workflows </_gallery/workflows/index>` (the recurring procedures of an analysis).


All the examples
----------------

.. include:: /_gallery/index.rst
   :start-after: _gallery: