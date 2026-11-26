pysdic.WorldPlaneOrthographicExtrinsic
==========================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


WorldPlaneOrthographicExtrinsic class
------------------------------------------

.. autoclass:: WorldPlaneOrthographicExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`WorldPlaneOrthographicExtrinsic` class can be instantiated in three ways:

- **From a rotation and translation vector** using the classmethod :meth:`from_rt`.
- **From a 4×4 transformation matrix** using the classmethod :meth:`from_tmatrix`.
- **From a frame of reference** using the classmethod :meth:`from_frame`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlaneOrthographicExtrinsic.from_frame
   WorldPlaneOrthographicExtrinsic.from_rt
   WorldPlaneOrthographicExtrinsic.from_tmatrix


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.WorldPlaneOrthographicExtrinsic` class.

Attributes
^^^^^^^^^^

The extrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of extrinsic parameters ``[rvec, tvec]``
  (the parameter :math:`t_z` has no effect on the normalized points).

The :class:`pysdic.WorldPlaneOrthographicExtrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlaneOrthographicExtrinsic.affine_matrix
   WorldPlaneOrthographicExtrinsic.frame
   WorldPlaneOrthographicExtrinsic.rotation_matrix
   WorldPlaneOrthographicExtrinsic.rotation_vector
   WorldPlaneOrthographicExtrinsic.transformation_matrix
   WorldPlaneOrthographicExtrinsic.translation_vector


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

   WorldPlaneOrthographicExtrinsic._apply
   WorldPlaneOrthographicExtrinsic._invert

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   WorldPlaneOrthographicExtrinsic.compute_rays
