Computer Vision Cameras Models (Extrinsic, Distortion, Intrinsic)
===================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

A camera projection model is a mathematical representation of how a camera captures and projects a 3D scene onto a 2D image plane.
The following sections describe the camera projection models implemented in the package,
their nomenclature, parameters, and the processes available to project points, distort images or compute rays.

.. important::

   All the camera operations in ``pysdic`` to process the transformation from the **world points** to the **image points**
   are based on the :class:`pysdic.Transformation` interface (see :doc:`Morphisms <morphisms>`).
   Any custom transformation implementing this interface can be used as an extrinsic, a distortion or an intrinsic.


Definition of Notations
-----------------------

A computer vision camera is modeled by three main components:

1. **Extrinsic**: The transformation from the **world coordinate system** to the **normalized camera coordinate system** (**world points** to **normalized points**)
2. **Distortion**: The transformation from the **normalized camera coordinate system** to the **distorted camera coordinate system** (**normalized points** to **distorted points**)
3. **Intrinsic**: The transformation from the **distorted camera coordinate system** to the **image coordinate system** (**distorted points** to **image points**)

.. code-block:: text

   world_points → normalized_points → distorted_points → image_points

The complete projection is the composition :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}`
(see :func:`pysdic.compose_projection`).

As described in the figure below, the package ``pysdic`` uses the following notation:

- ``world_points``: The 3-D points :math:`\vec{X}_w` with shape (...,3) expressed in the **world coordinate system** :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)`, or the 2-D points :math:`\vec{x}_w` with shape (...,2) for planar problems.
- ``normalized_points``: The 2-D points :math:`\vec{x}_n` with shape (...,2) expressed in the **normalized camera coordinate system** :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
- ``distorted_points``: The distorted 2-D points :math:`\vec{x}_d` with shape (...,2) expressed in the **normalized camera coordinate system** :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
- ``image_points``: The 2-D points :math:`\vec{x}_i` with shape (...,2) expressed in the **image coordinate system** :math:`(\vec{e}_x, \vec{e}_y)` in the sensor plane.
- ``pixel_points``: The 2-D points :math:`\vec{x}_p` with shape (...,2) expressed in the **pixel coordinate system** :math:`(u, v)` in the matrix of pixels.

.. figure:: /_static/cv_definition.png
   :align: center
   :width: 60%
   :alt: Definition of quantities in ``pysdic``.

The coordinate systems are defined as follows:

- **world coordinate system**: The 3-D coordinate system :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)` in which the 3-D points are expressed.
- **normalized camera coordinate system**: The 2-D coordinate system :math:`(\vec{I}, \vec{J})` extracted from the 3-D camera system :math:`(\vec{I}, \vec{J}, \vec{K})` by applying the extrinsic transformation. This 2-D plane is orthogonal to the optical axis :math:`\vec{K}`.
- **image coordinate system**: The 2-D coordinate system :math:`(\vec{e}_x, \vec{e}_y)` in the image plane with the origin at the top-left corner of the image and the axes defined by the sensor plane (width and height).
- **pixel coordinate system**: The 2-D coordinate system :math:`(u, v)` in the matrix of pixels with the origin at the top-left corner of the image and the axes defined by the pixel grid (width and height).

.. note::

   To convert the **pixel points** to the **image points** (swapping from pixel coordinates to image coordinates), a simple switch of coordinate system can be performed:

   .. code-block:: python

      pixel_points = numpy.indices((img_height, img_width)).reshape(2, -1).T  # Axis [H, W -> Y, X]
      image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format


Morphisms (Transformations and Parametrizations)
------------------------------------------------

The extrinsic, distortion and intrinsic models are subclasses of :class:`pysdic.Transformation`.
The parameters of a transformation can be re-expressed with a :class:`pysdic.Parametrization`
(for example to optimize only some of them), and several transformations can be composed together.

.. grid:: 1

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Morphisms
      ^^^

      This section describes the common interface of the transformations and parametrizations
      (``apply``, ``invert``, ``encode``, ``decode``, Jacobians, compositions and reparametrizations).

      +++

      .. button-ref:: morphisms
         :expand:
         :color: secondary
         :click-parent:

         To the morphisms reference guide

.. toctree::
   :maxdepth: 1
   :hidden:

   ./morphisms.rst


Implemented transformations and parametrizations
------------------------------------------------

The package ``pysdic`` provides a set of implemented transformations that can be used to process the
transformation from the **world points** to the **image points**, and a set of implemented parametrizations.

.. grid:: 2

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Extrinsic
      ^^^

      The implemented ``Extrinsic`` models from the **world points** (3D or 2D)
      to the **normalized points**, and the camera rays.

      +++

      .. button-ref:: implemented_extrinsic
         :expand:
         :color: secondary
         :click-parent:

         To the extrinsic models reference guide

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Distortion
      ^^^

      The implemented ``Distortion`` models from the **normalized points**
      to the **distorted points**.

      +++

      .. button-ref:: implemented_distortion
         :expand:
         :color: secondary
         :click-parent:

         To the distortion models reference guide

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Intrinsic
      ^^^

      The implemented ``Intrinsic`` models from the **distorted points**
      to the **image points**.

      +++

      .. button-ref:: implemented_intrinsic
         :expand:
         :color: secondary
         :click-parent:

         To the intrinsic models reference guide

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Parametrizations
      ^^^

      The implemented ``Parametrization`` models to change the parametric
      representation of a transformation.

      +++

      .. button-ref:: implemented_parametrization
         :expand:
         :color: secondary
         :click-parent:

         To the parametrizations reference guide

.. toctree::
   :maxdepth: 1
   :hidden:

   ./implemented_extrinsic.rst
   ./implemented_distortion.rst
   ./implemented_intrinsic.rst
   ./implemented_parametrization.rst


Projection of points
--------------------

Project the **world points** to the **image points** through the extrinsic, distortion and intrinsic
transformations, with optional Jacobians with respect to the points and to the parameters of each step.

- :doc:`project_points <computer_vision/project_points>`: Project the **world points** to the **image points**.
- :doc:`compose_projection <computer_vision/compose_projection>`: Build the projection :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}` as a :class:`pysdic.ComposedTransformation`.
- :doc:`ProjectionResult <computer_vision/ProjectionResult>`: Result of :func:`pysdic.project_points`, with the blocks of the Jacobian for the extrinsic, distortion and intrinsic parameters.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./computer_vision/project_points.rst
   ./computer_vision/compose_projection.rst
   ./computer_vision/ProjectionResult.rst


Distortion of points and images
-------------------------------

Distort or undistort the **image points** or whole images with the intrinsic and distortion transformations
(and an optional rectification :math:`R` and projection :math:`P` for the points).

- :doc:`distort_points <computer_vision/distort_points>`: Distort the **image points** (:math:`\text{Distortion} \circ \text{Intrinsic}^{-1}`).
- :doc:`undistort_points <computer_vision/undistort_points>`: Undistort the **image points** to the **normalized points** (:math:`\text{Distortion}^{-1} \circ \text{Intrinsic}^{-1}`).
- :doc:`DistortionResult <computer_vision/DistortionResult>`: Result of :func:`pysdic.distort_points` and :func:`pysdic.undistort_points`, with the blocks of the Jacobian for each step.
- :doc:`distort_image <computer_vision/distort_image>`: Distort an image (the source image is the perfect image).
- :doc:`undistort_image <computer_vision/undistort_image>`: Undistort an image, as ``cv2.undistort``, with any intrinsic and distortion.
- :doc:`display_distortion_field <computer_vision/display_distortion_field>`: Display the displacement field generated by a distortion on the image.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./computer_vision/distort_points.rst
   ./computer_vision/undistort_points.rst
   ./computer_vision/DistortionResult.rst
   ./computer_vision/distort_image.rst
   ./computer_vision/undistort_image.rst
   ./computer_vision/display_distortion_field.rst


Rays and triangulation
----------------------

Compute the rays of a camera through the **image points** and triangulate the points seen by several cameras
(see also the class :class:`pysdic.Rays` in :doc:`Implemented Extrinsic <implemented_extrinsic>`).

- :doc:`compute_rays <computer_vision/compute_rays>`: Compute the rays in the world coordinate system through the **image points**.
- :doc:`intersect_rays <computer_vision/intersect_rays>`: Compute the point closest to several rays in the least squares sense.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./computer_vision/compute_rays.rst
   ./computer_vision/intersect_rays.rst


Optical flow operations
------------------------------

The package includes functions to compute the optical flow between two images
based on the ``OpenCV`` ``DIS`` (Dense Inverse Search) algorithm.

- :doc:`compute_optical_flow <computer_vision/compute_optical_flow>`: Compute the optical flow between two images using the DIS method of OpenCV.
- :doc:`DisflowConfig <core/DisflowConfig>`: Parameters of the DIS optical flow algorithm.
- :doc:`display_optical_flow <computer_vision/display_optical_flow>`: Display the magnitude and the components of the optical flow overlaid on an image.
- :doc:`display_optical_flow_quiver <computer_vision/display_optical_flow_quiver>`: Display the optical flow as a quiver plot overlaid on the two images.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./computer_vision/compute_optical_flow.rst
   ./computer_vision/display_optical_flow.rst
   ./computer_vision/display_optical_flow_quiver.rst

See the section :doc:`Usage <../usage>` for more details and examples on how to use these processes.


Optimisation processes
----------------------

The package provides optimisation processes based on the Gauss-Newton solver ``pysolvegn`` to estimate
the parameters or the input points of the transformations. The transformations are combined in **chains**:
a chain is a sequence of indices of transformations applied in this order (for example ``(0, 2, 3)`` is
:math:`T_3 \circ T_2 \circ T_0`), and a transformation can be shared by several chains (for example the
intrinsic and distortion of a camera observing a target in several positions).

Each process comes in two versions:

- ``build_*_term``: build a ``pysolvegn`` term and the initial vector ``p0``, to combine several terms
  (different losses or weights) or to use the advanced options of ``pysolvegn``.
- ``optimize_*``: build the term and directly solve the problem.

The terms can be built in mode ``"rJ"`` (residuals and Jacobian, with robust losses) or ``"gH"``
(gradient and Gauss-Newton Hessian, lower memory).

Optimization of the parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate the parameters :math:`\vec{p}` of the transformations such that :math:`C_c(X_c; \vec{p}) \approx Y_c`
for each chain (for example the calibration of a camera, with ``pysolvegn.solve``).

- :doc:`optimize_transformation_parameters <computer_vision/optimize_transformation_parameters>`: Optimize the parameters of a single transformation.
- :doc:`optimize_chains_transformation_parameters <computer_vision/optimize_chains_transformation_parameters>`: Optimize the parameters of several transformations combined in chains.
- :doc:`build_transformation_term <computer_vision/build_transformation_term>`: Build the ``pysolvegn.Term`` for a single transformation.
- :doc:`build_chains_transformation_term <computer_vision/build_chains_transformation_term>`: Build the ``pysolvegn.Term`` for several transformations combined in chains.

Optimization of the input points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate the input points :math:`X` such that :math:`C_c(X) \approx Y_c` for each chain, the parameters
being fixed (for example a numerical inverse of a transformation, or the triangulation of points seen
by several cameras). Each point is an independent problem solved with ``pysolvegn.solve_batch``.

- :doc:`optimize_transformation_input_points <computer_vision/optimize_transformation_input_points>`: Optimize the input points of a single transformation.
- :doc:`optimize_chains_transformation_input_points <computer_vision/optimize_chains_transformation_input_points>`: Optimize the input points shared by several chains of transformations.
- :doc:`build_transformation_input_points_term <computer_vision/build_transformation_input_points_term>`: Build the ``pysolvegn.BatchTerm`` for a single transformation.
- :doc:`build_chains_transformation_input_points_term <computer_vision/build_chains_transformation_input_points_term>`: Build the ``pysolvegn.BatchTerm`` for several chains of transformations.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./computer_vision/optimize_transformation_parameters.rst
   ./computer_vision/optimize_chains_transformation_parameters.rst
   ./computer_vision/build_transformation_term.rst
   ./computer_vision/build_chains_transformation_term.rst
   ./computer_vision/optimize_transformation_input_points.rst
   ./computer_vision/optimize_chains_transformation_input_points.rst
   ./computer_vision/build_transformation_input_points_term.rst
   ./computer_vision/build_chains_transformation_input_points_term.rst


Create a Camera Object by Composing Transformations
-----------------------------------------------------

The package provides a :class:`pysdic.Camera` class that is used for SDIC applications.
This class is a wrapper around the transformations and processes described above, which allows to create a camera
model by composing the extrinsic, distortion, and intrinsic transformations together, and to apply the transformations and processes to points or images.

.. seealso::

   See the section :doc:`Imaging Model <imaging>` for more details on how to create a :class:`pysdic.Camera` and how to use it for SDIC applications.
