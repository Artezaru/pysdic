pysdic.RigidExtrinsic
=====================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.RigidExtrinsic` is the common base of the extrinsic transformations made of a rigid motion of the
world points into the frame of the camera, :math:`\vec{X}_c = R \, \vec{X} + \vec{t}`, followed by a projection onto
the normalized plane:

- :class:`pysdic.PerspectiveExtrinsic`: perspective projection :math:`(X_c / Z_c, \ Y_c / Z_c)` (pinhole camera);
- :class:`pysdic.OrthographicExtrinsic`: orthographic projection :math:`(X_c, \ Y_c)` (telecentric lens).

The rotation :math:`R` and the translation :math:`\vec{t}` (world to camera, OpenCV convention) are the parameters of
the transformation.


RigidExtrinsic class
--------------------

.. autoclass:: RigidExtrinsic
   :members:
   :show-inheritance:

.. seealso::

   - :doc:`PerspectiveExtrinsic <PerspectiveExtrinsic>` and :doc:`OrthographicExtrinsic <OrthographicExtrinsic>` for
     the two implementations.
   - :doc:`PlanarRigidExtrinsic <PlanarRigidExtrinsic>` and the ``WorldPlane*Extrinsic`` for the extrinsics of a plane.
