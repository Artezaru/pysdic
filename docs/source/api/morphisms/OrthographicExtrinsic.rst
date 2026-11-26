pysdic.OrthographicExtrinsic
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


OrthographicExtrinsic class
------------------------------------

.. autoclass:: OrthographicExtrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`OrthographicExtrinsic` class can be instantiated in three ways:

- **From a rotation and translation vector** using the classmethod :meth:`from_rt`.
- **From a 4×4 transformation matrix** using the classmethod :meth:`from_tmatrix`.
- **From a frame of reference** using the classmethod :meth:`from_frame`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   OrthographicExtrinsic.from_frame
   OrthographicExtrinsic.from_rt
   OrthographicExtrinsic.from_tmatrix


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.OrthographicExtrinsic` class.

Attributes
^^^^^^^^^^

The extrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of extrinsic parameters ``[rvec, tvec]``.

The :class:`pysdic.OrthographicExtrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   OrthographicExtrinsic.frame
   OrthographicExtrinsic.rotation_matrix
   OrthographicExtrinsic.rotation_vector
   OrthographicExtrinsic.transformation_matrix
   OrthographicExtrinsic.translation_vector


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform extrinsic projections:

- ``apply``: maps ``world_points`` → ``normalized_points``.
- ``invert``: maps ``normalized_points`` → ``world_points`` (at the depth given by the ``depth`` argument).
- ``compute_rays``: maps ``normalized_points`` → ``rays`` (origins and unit directions in world coordinates).

Jacobians with respect to the input points and extrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂normalized_points / ∂world_points
- ``dp=True``: compute ∂normalized_points / ∂extrinsic_parameters

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   OrthographicExtrinsic._apply
   OrthographicExtrinsic._invert

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   OrthographicExtrinsic.compute_rays
