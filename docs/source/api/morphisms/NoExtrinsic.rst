pysdic.NoExtrinsic
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


NoExtrinsic class
------------------------------------

.. autoclass:: NoExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`NoExtrinsic` class is instantiated directly with its constructor:

- **With 3D world points** (default): ``NoExtrinsic()`` or ``NoExtrinsic(input_dim=3)``.
- **With planar world points**: ``NoExtrinsic(input_dim=2)``.


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.NoExtrinsic` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.NoExtrinsic` has no parameters: ``parameters`` is an empty vector.

The dimension of the world points is given by the ``input_dim`` attribute of the base class
:class:`pysdic.Transformation` (2 or 3).


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform extrinsic projections:

- ``apply``: maps ``world_points`` → ``normalized_points``.
- ``invert``: maps ``normalized_points`` → ``world_points`` (at the depth given by the ``depth`` argument for 3D world points).
- ``compute_rays``: maps ``normalized_points`` → ``rays`` (origins and unit directions in world coordinates, only for 3D world points).

Jacobians with respect to the input points and extrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂normalized_points / ∂world_points
- ``dp=True``: compute ∂normalized_points / ∂extrinsic_parameters (empty)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   NoExtrinsic._apply
   NoExtrinsic._invert

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   NoExtrinsic.compute_rays
