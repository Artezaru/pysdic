pysdic.ImageProjectionResult
============================

.. currentmodule:: pysdic

The :class:`pysdic.ImageProjectionResult` stores the result of :meth:`pysdic.View.image_project`: the image values
``gray_levels`` (or ``x``) with a channel axis, the projected ``image_points``, and the Jacobians with views on the
blocks of the extrinsic, distortion and intrinsic parameters.

.. seealso::

   - :class:`pysdic.ProjectionResult`: the result of the geometric projection of :meth:`pysdic.Camera.project`.

.. autoclass:: ImageProjectionResult
   :members:
   :show-inheritance:
