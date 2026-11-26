pysdic.WorldPlanePerspectiveExtrinsic
==========================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


WorldPlanePerspectiveExtrinsic class
------------------------------------------

.. autoclass:: WorldPlanePerspectiveExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`WorldPlanePerspectiveExtrinsic` class can be instantiated in three ways:

- **From a rotation and translation vector** using the classmethod :meth:`from_rt`.
- **From a 4×4 transformation matrix** using the classmethod :meth:`from_tmatrix`.
- **From a frame of reference** using the classmethod :meth:`from_frame`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlanePerspectiveExtrinsic.from_frame
   WorldPlanePerspectiveExtrinsic.from_rt
   WorldPlanePerspectiveExtrinsic.from_tmatrix


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.WorldPlanePerspectiveExtrinsic` class.

Attributes
^^^^^^^^^^

The extrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of extrinsic parameters ``[rvec, tvec]``.

The :class:`pysdic.WorldPlanePerspectiveExtrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlanePerspectiveExtrinsic.frame
   WorldPlanePerspectiveExtrinsic.homography_matrix
   WorldPlanePerspectiveExtrinsic.rotation_matrix
   WorldPlanePerspectiveExtrinsic.rotation_vector
   WorldPlanePerspectiveExtrinsic.transformation_matrix
   WorldPlanePerspectiveExtrinsic.translation_vector


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform extrinsic projections:

- ``apply``: maps ``world_points`` (2D points of the plane :math:`Z_w = 0`) → ``normalized_points``.
- ``invert``: maps ``normalized_points`` → ``world_points`` (intersection of the rays with the plane :math:`Z_w = 0`, no depth needed).
- ``compute_rays``: maps ``normalized_points`` → ``rays`` (origins and unit directions in 3D world coordinates).

Jacobians with respect to the input points and extrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂normalized_points / ∂world_points
- ``dp=True``: compute ∂normalized_points / ∂extrinsic_parameters

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlanePerspectiveExtrinsic._apply
   WorldPlanePerspectiveExtrinsic._invert

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlanePerspectiveExtrinsic.compute_rays
