Implemented Extrinsic Transformations
=====================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The package ``pysdic`` provides a set of implemented extrinsic transformation models
that can be used to transform **world points** to 2D **normalized points**:

.. code-block:: text

    world_points → normalized_points → distorted_points → image_points

The world points can be 3D points :math:`(X_w, Y_w, Z_w)` or 2D points :math:`(x_w, y_w)`,
depending on the extrinsic model.


Extrinsics with 3D world points
-------------------------------

These models map 3D **world points** to 2D **normalized points**. Since the depth is lost
by the projection, the inverse transformation needs the depth :math:`Z_c` of the points in the
camera coordinate system, given with the ``depth`` argument of ``invert``.
The rays of the camera passing through the normalized points can be computed with the method ``compute_rays``.

- :doc:`NoExtrinsic <morphisms/NoExtrinsic>`: Identity pose without rotation nor translation (default ``input_dim=3``): the **normalized points** are the "x,y" coordinates of the **world points** (orthographic projection, the coordinate :math:`z` is dropped).
- :doc:`PerspectiveExtrinsic <morphisms/PerspectiveExtrinsic>`: Like OpenCV's, apply a rigid transformation :math:`[R|T]` to the **world points** and normalize by the depth :math:`Z_c` (perspective division).
- :doc:`OrthographicExtrinsic <morphisms/OrthographicExtrinsic>`: Apply a rigid transformation :math:`[R|T]` to the **world points** without normalizing by the depth (orthographic projection).
- :doc:`RigidExtrinsic <morphisms/RigidExtrinsic>`: Common base class of :class:`pysdic.PerspectiveExtrinsic` and :class:`pysdic.OrthographicExtrinsic` (rigid transformation :math:`[R|T]` followed by a projection).

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/NoExtrinsic.rst
   morphisms/PerspectiveExtrinsic.rst
   morphisms/OrthographicExtrinsic.rst
   morphisms/RigidExtrinsic.rst


Extrinsics with 2D world points
-------------------------------

These models map 2D **world points** to 2D **normalized points**. No depth is lost:
the transformations are left and right invertible and the ``invert`` method does not need any depth.

Planar world observed by a 3D camera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The world points are the points :math:`(x_w, y_w)` of the plane :math:`Z_w = 0`, observed by a camera
with any 3D pose :math:`[R|T]` (6 parameters). This allows to work with planar displacement fields
while the camera is free to observe the plane. The rays computed with ``compute_rays`` are expressed in
the 3D world coordinate system.

- :doc:`WorldPlanePerspectiveExtrinsic <morphisms/WorldPlanePerspectiveExtrinsic>`: Perspective camera observing the plane :math:`Z_w = 0` (plane-induced homography :math:`H = [\vec{r}_1, \vec{r}_2, T]`).
- :doc:`WorldPlaneOrthographicExtrinsic <morphisms/WorldPlaneOrthographicExtrinsic>`: Orthographic camera observing the plane :math:`Z_w = 0` (affine map of the plane).

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/WorldPlanePerspectiveExtrinsic.rst
   morphisms/WorldPlaneOrthographicExtrinsic.rst

Transformations of the plane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The world points and the normalized points are linked by a 2D transformation of the plane,
used to run the camera algorithms on purely planar problems.

- :doc:`NoExtrinsic <morphisms/NoExtrinsic>` with ``input_dim=2``: Identity of the plane (the **normalized points** are equal to the **world points**).
- :doc:`PlanarRigidExtrinsic <morphisms/PlanarRigidExtrinsic>`: Apply a 2D rigid transformation composed of a rotation :math:`\theta` and a translation :math:`\vec{t}` to the **world points** (3 parameters).
- :doc:`PlanarHomographicExtrinsic <morphisms/PlanarHomographicExtrinsic>`: Apply a 2D projective homography :math:`H` to the **world points** (8 parameters, :math:`H_{22} = 1`).

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/PlanarRigidExtrinsic.rst
   morphisms/PlanarHomographicExtrinsic.rst


Rays
----

The method ``compute_rays`` of the extrinsics observing a 3D world (all the models above except
:class:`pysdic.PlanarRigidExtrinsic`, :class:`pysdic.PlanarHomographicExtrinsic` and :class:`pysdic.NoExtrinsic`
with ``input_dim=2``) returns the rays of the camera passing through the given **normalized points**:

- :doc:`Rays <morphisms/Rays>`: Collection of rays in 3D world coordinates, stored with their origin and their unit direction in an array with shape :math:`(\dots, 6)`.

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/Rays.rst