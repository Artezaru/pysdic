pysdic.PlanarHomographicExtrinsic
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


PlanarHomographicExtrinsic class
------------------------------------

.. autoclass:: PlanarHomographicExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`PlanarHomographicExtrinsic` class can be instantiated in three ways:

- **From a parameter vector** of shape ``(8,)`` passed to the constructor.
- **From a 3×3 homography matrix** using the classmethod :meth:`from_hmat`.
- **As the identity homography** using the classmethod :meth:`from_eye`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PlanarHomographicExtrinsic.from_eye
   PlanarHomographicExtrinsic.from_hmat


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.PlanarHomographicExtrinsic` class.

Attributes
^^^^^^^^^^

The extrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of extrinsic parameters ``[h00, h01, h02, h10, h11, h12, h20, h21]``.

The :class:`pysdic.PlanarHomographicExtrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PlanarHomographicExtrinsic.h00
   PlanarHomographicExtrinsic.h01
   PlanarHomographicExtrinsic.h02
   PlanarHomographicExtrinsic.h10
   PlanarHomographicExtrinsic.h11
   PlanarHomographicExtrinsic.h12
   PlanarHomographicExtrinsic.h20
   PlanarHomographicExtrinsic.h21
   PlanarHomographicExtrinsic.homography_matrix


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

   PlanarHomographicExtrinsic._apply
   PlanarHomographicExtrinsic._invert
