API Reference
=============

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

This section contains a detailed description of the functions, modules and objects included in ``pysdic``.
The reference describes how the methods work and which parameters can be used.
This page summarizes the logic of a SDIC analysis and the objects to create, then gives links to the
reference of each part of the package.


What is a SDIC analysis?
------------------------

A stereo digital image correlation (SDIC) analysis measures the displacement of a surface from the images
acquired by one or several cameras at several instants (the **frames**).

The surface is described by a reference **mesh**. Its displacement at a frame :math:`f` is the displacement
:math:`\vec{u}_f` of its vertices, interpolated inside the elements with the shape functions. A material point
:math:`\vec{X}` of the surface is projected in each image by the **camera** that acquired it, and the gray level
of the image at this projection is read. A **pairing** compares the gray levels of the same material points in two
views (two cameras at the same frame, or the same camera at two frames):

.. math::

   r(\vec{X}) = [I_2 \circ P_2](\vec{X} + \vec{u}_{f_2}(\vec{X})) - [I_1 \circ P_1](\vec{X} + \vec{u}_{f_1}(\vec{X}))

where :math:`P_1`, :math:`P_2` are the projections of the cameras and :math:`I_1`, :math:`I_2` the images.
The displacements (and optionally some parameters of the cameras) are the unknowns that minimize the sum of the
squared residuals over the integration points of the mesh, solved with the Gauss-Newton solver ``pysolvegn``.

.. code-block:: text

   Geometry                    Imaging                                SDIC
   --------                    -------                                ----
   Mesh ─────────────────────────────────────────────────────────►  SDICAnalysis(mesh, views)
   IntegrationPoints ──────────────────────────────────────┐          ├── displacements (per frame)
                               Extrinsic ┐                 │          └── views (per frame)
                               Distortion├─► Camera ┐      │                 ▲         │
                               Intrinsic ┘          ├─► View ────────────────┘         │
                               Image ───────────────┘      │                           │
                                                           ▼                           ▼
                                        Pairing ──► SDICCorrelationTerm(analysis, unknowns, ...)
                                                           ▲               │ .term
                            SDICUnknowns(frames, transformations)          ▼
                                                                    pysolvegn.solve


Objects to create
-----------------

A SDIC analysis is built in five steps:

1. **The geometry** (see :doc:`Geometry <api/geometry>`): the reference :class:`pysdic.Mesh` of the observed surface
   (vertices and elements of a supported element type, created from a file or with the ``create_*`` functions),
   and the :class:`pysdic.IntegrationPoints` located in its elements: the material points whose gray levels are compared.

2. **The cameras** (see :doc:`Computer Vision Cameras Models <api/camera_models>`): each camera is the composition
   :math:`\text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}` of three :class:`pysdic.Transformation`,
   usually calibrated beforehand (for example with :func:`pysdic.optimize_chains_transformation_parameters`).

3. **The images and views** (see :doc:`Imaging Model <api/imaging>`): a :class:`pysdic.Camera` gathers the three
   transformations with the size of its sensor, an :class:`pysdic.Image` stores an acquired image with its interpolation
   functions and region of interest, and a :class:`pysdic.View` associates an image with the camera that acquired it.

4. **The analysis** (see :doc:`SDIC Analysis <api/sdic>`): a :class:`pysdic.SDICAnalysis` stores the mesh, the views
   of each frame and the displacements. The usual setups (two cameras, split-screen, Cordin, ...) are created directly
   with the ``create_*_analysis`` functions.

5. **The problem** (see :doc:`SDIC Analysis <api/sdic>`): a :class:`pysdic.SDICUnknowns` describes the unknowns (the
   displacements of some frames and/or some parameters of the cameras) without knowing the analysis, the views to
   compare are given as :class:`pysdic.Pairing` objects, and a :class:`pysdic.SDICCorrelationTerm` builds the least
   squares term on the analysis and the unknowns. A :class:`pysdic.SDICTemporalRegularizationTerm` and a
   :class:`pysdic.SDICFixedParametrization` can be built on the same analysis and unknowns
   (:func:`pysdic.check_same_problem`).

.. code-block:: python

   import pysolvegn
   from pysdic import Camera, Image, Pairing, SDICCorrelationTerm, SDICUnknowns, create_bicamera_analysis

   # 1. Geometry: reference mesh and integration points (see Geometry)
   mesh = ...
   integration_points = ...

   # 2. Cameras: calibrated transformations (see Computer Vision Cameras Models)
   left = Camera(height, width, extrinsic=left_extrinsic, distortion=left_distortion, intrinsic=left_intrinsic)
   right = Camera(height, width, extrinsic=right_extrinsic, distortion=right_distortion, intrinsic=right_intrinsic)

   # 3. Images (see Imaging Model)
   left_images = [Image.from_file(f"left_{f:03d}.tif") for f in range(n_frames)]
   right_images = [Image.from_file(f"right_{f:03d}.tif") for f in range(n_frames)]

   # 4. Analysis: views of each frame (see SDIC Analysis)
   analysis = create_bicamera_analysis(mesh, left, right, left_images, right_images)
   analysis.set_zero_displacement(0)  # Reference frame
   analysis.set_zero_displacement(1)  # Initial guess of the frame 1

   # 5. Problem: unknowns, pairings and correlation term (see SDIC Analysis)
   unknowns = SDICUnknowns([1], [])  # Displacement of the frame 1, cameras fixed
   pairings = [Pairing.temporal(0, 0, 1), Pairing.temporal(1, 0, 1), Pairing.stereo(1, 0, 1)]
   correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings, gl_correction={"mode": "ZNSSD", "scale": 255})

   result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
   analysis.apply_unknowns(unknowns, result.parameters)  # Total displacement of the frame 1 written in the analysis


SDIC analysis
-------------

.. grid:: 1

   .. grid-item-card::
      :img-top: /_static/_icons/pysdic_icon.png
      :text-align: center

      SDIC Analysis (Analysis, Pairings, Setups and Optimization)
      ^^^

      The :class:`pysdic.SDICAnalysis` (mesh, displacements and views of each frame), the constructors of the usual
      acquisition setups, the :class:`pysdic.Pairing` between views, the gray level corrections and the optimization
      of the displacements and of the cameras: :class:`pysdic.SDICUnknowns`, :class:`pysdic.SDICCorrelationTerm`,
      temporal regularization and fixed parametrization.

      +++

      .. button-ref:: api/sdic
         :expand:
         :color: secondary
         :click-parent:

         To the SDIC analysis reference guide.


Components of the analysis
--------------------------

.. grid:: 2

   .. grid-item-card::
      :img-top: /_static/_icons/geometry.png
      :text-align: center

      Geometry (Point Clouds, Meshes and Integration Points)
      ^^^

      The geometrical objects of the observed surfaces: :class:`pysdic.PointCloud`, :class:`pysdic.Mesh`
      and :class:`pysdic.IntegrationPoints`, the creation of structured surface meshes and the 3D visualization.

      +++

      .. button-ref:: api/geometry
         :expand:
         :color: secondary
         :click-parent:

         To the geometry reference guide.

   .. grid-item-card::
      :img-top: /_static/_icons/camera.png
      :text-align: center

      Imaging Model (Cameras, Images and Views)
      ^^^

      The :class:`pysdic.Camera`, :class:`pysdic.Image` and :class:`pysdic.View` objects to project the
      material points to the gray levels of the images, the visibility of the meshes and the 2D visualization.

      +++

      .. button-ref:: api/imaging
         :expand:
         :color: secondary
         :click-parent:

         To the imaging model reference guide.

   .. grid-item-card::
      :img-top: /_static/_icons/projection.png
      :text-align: center

      Computer Vision Cameras Models (Extrinsic, Distortion, Intrinsic)
      ^^^

      The transformations and parametrizations of the camera models, the projection, distortion and rays
      of points, the optical flow and the optimization processes (calibration, triangulation).

      +++

      .. button-ref:: api/camera_models
         :expand:
         :color: secondary
         :click-parent:

         To the computer vision camera models reference guide.

   .. grid-item-card::
      :img-top: /_static/_icons/topology.png
      :text-align: center

      Finite Element Tools (``pysdic.fem``)
      ^^^

      The low-level, array-based functions on meshes used by the geometrical objects: shape functions, Gauss
      quadrature, interpolation, assembly of operators, ray casting and neighborhoods.

      +++

      .. button-ref:: api/fem
         :expand:
         :color: secondary
         :click-parent:

         To the finite element tools reference guide.


Workflows
---------

.. grid:: 1

   .. grid-item-card::
      :img-top: /_static/_icons/workflows.png
      :text-align: center

      Workflows (``pysdic.workflows``)
      ^^^

      A bank of procedures built on the API for the recurring tasks of an analysis: preparation of the integration
      points, initialization of the displacements, shape at the reference frame, interpolation and filtering of the
      displacements, quality control (residuals per element, uncertainty).

      +++

      .. button-ref:: api/workflows
         :expand:
         :color: secondary
         :click-parent:

         To the workflows reference guide.


Synthetic images
----------------

.. grid:: 2

   .. grid-item-card::
      :img-top: /_static/_icons/rendering.png
      :text-align: center

      Synthetic Images (Textures, Generation and Noise)
      ^^^

      Generate the synthetic images of textured surface meshes seen by cameras, with a known displacement,
      to test and validate a SDIC analysis: speckle textures, rendering through the UV maps and acquisition defects.

      +++

      .. button-ref:: api/synthetic
         :expand:
         :color: secondary
         :click-parent:

         To the synthetic images reference guide.

   .. grid-item-card::
      :img-top: /_static/_icons/blender.png
      :text-align: center

      Blender Rendering (``pysdic-blender``)
      ^^^

      Render the same scenes with lighting, shading and materials with Blender, with the separate package
      ``pysdic-blender`` (Python 3.13).

      +++

      .. button-ref:: api/blender
         :expand:
         :color: secondary
         :click-parent:

         To the Blender rendering guide.


Internal tools (for developers)
-------------------------------

.. grid:: 2

   .. grid-item-card::
      :img-top: /_static/_icons/core.png
      :text-align: center

      Core Tools (internal classes)
      ^^^

      The internal building blocks shared by the objects of the package: the property container of the
      geometrical objects and the serialization mixins (JSON, HDF5, NPZ). They are not exported in the
      ``pysdic`` namespace, but their methods are available on the public objects.

      +++

      .. button-ref:: api/core
         :expand:
         :color: secondary
         :click-parent:

         To the core tools reference guide.

   .. grid-item-card::
      :img-top: /_static/_icons/core.png
      :text-align: center

      Numerical Tools (``pysdic.numeric``)
      ^^^

      Finite differences on non-uniform abscissas (coefficients, matrices, application to a signal) and
      polynomial fitting, used by the temporal operators.

      +++

      .. button-ref:: api/numeric
         :expand:
         :color: secondary
         :click-parent:

         To the numerical tools reference guide.


Acquisition devices
-------------------

.. grid:: 1

   .. grid-item-card::
      :img-top: /_static/_icons/camera.png
      :text-align: center

      CORDIN Captures (``pysdic.cordin``)
      ^^^

      Read the captures of the CORDIN high-speed cameras: parameters of the capture, details of the camera and
      correspondence between the frames and the modules.

      +++

      .. button-ref:: api/cordin
         :expand:
         :color: secondary
         :click-parent:

         To the CORDIN captures reference guide.


.. toctree::
   :caption: Contents:
   :hidden:

   api/sdic
   api/geometry
   api/imaging
   api/camera_models
   api/fem
   api/workflows
   api/synthetic
   api/blender
   api/core
   api/numeric
   api/cordin