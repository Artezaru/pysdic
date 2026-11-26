pysdic.PlanarRigidExtrinsic
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


PlanarRigidExtrinsic class
------------------------------------

.. autoclass:: PlanarRigidExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`PlanarRigidExtrinsic` class can be instantiated in two ways:

- **From a rotation angle and a translation vector** using the classmethod :meth:`from_rt`.
- **From a 3×3 homogeneous transformation matrix** using the classmethod :meth:`from_tmatrix`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PlanarRigidExtrinsic.from_rt
   PlanarRigidExtrinsic.from_tmatrix


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.PlanarRigidExtrinsic` class.

Attributes
^^^^^^^^^^

The extrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of extrinsic parameters ``[theta, t_x, t_y]``.

The :class:`pysdic.PlanarRigidExtrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PlanarRigidExtrinsic.rotation_angle
   PlanarRigidExtrinsic.rotation_matrix
   PlanarRigidExtrinsic.transformation_matrix
   PlanarRigidExtrinsic.translation_vector


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform extrinsic projections:

- ``apply``: maps ``world_points`` (2D) → ``normalized_points``.
- ``invert``: maps ``normalized_points`` → ``world_points`` (2D).

Jacobians with respect to the input points and extrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂normalized_points / ∂world_points
- ``dp=True``: compute ∂normalized_points / ∂extrinsic_parameters

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PlanarRigidExtrinsic._apply
   PlanarRigidExtrinsic._invert
