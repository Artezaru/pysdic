pysdic.PinholeIntrinsic
========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


PinholeIntrinsic class
----------------------

.. autoclass:: PinholeIntrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`PinholeIntrinsic` class can be instantiated in three ways:

- **From a 3×3 intrinsic matrix** using the classmethod :meth:`from_matrix`.
- **From a parameter vector** using the classmethod :meth:`from_vector`.
- **From individual parameters** using the classmethod :meth:`from_components`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PinholeIntrinsic.from_matrix
   PinholeIntrinsic.from_vector
   PinholeIntrinsic.from_components

Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.PinholeIntrinsic` class.

Attributes
^^^^^^^^^^

The intrinsic parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of intrinsic parameters ``[f_x, f_y, c_x, c_y]`` (or ``[f_x, f_y, c_x, c_y, s]`` if ``use_skew=True``).

The :class:`pysdic.PinholeIntrinsic` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PinholeIntrinsic.allow_negative_focal
   PinholeIntrinsic.focal_length_x
   PinholeIntrinsic.focal_length_y
   PinholeIntrinsic.intrinsic_matrix
   PinholeIntrinsic.intrinsic_vector
   PinholeIntrinsic.principal_point_x
   PinholeIntrinsic.principal_point_y
   PinholeIntrinsic.skew
   PinholeIntrinsic.use_skew


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform intrinsic projections:

- ``apply``: maps ``distorted_points`` → ``image_points``.
- ``invert``: maps ``image_points`` → ``distorted_points``.

Jacobians with respect to the input points and intrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂image_points / ∂distorted_points
- ``dp=True``: compute ∂image_points / ∂intrinsic_parameters

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PinholeIntrinsic._apply
   PinholeIntrinsic._invert
