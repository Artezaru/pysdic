pysdic.NoIntrinsic
========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


NoIntrinsic class
----------------------

.. autoclass:: NoIntrinsic
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`NoIntrinsic` class takes no argument and is instantiated directly with ``NoIntrinsic()``.


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.NoIntrinsic` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.NoIntrinsic` has no parameters: ``parameters`` is an empty vector.


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform intrinsic projections:

- ``apply``: maps ``distorted_points`` → ``image_points`` (identity).
- ``invert``: maps ``image_points`` → ``distorted_points`` (identity).

Jacobians with respect to the input points and intrinsic parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂image_points / ∂distorted_points (identity matrix)
- ``dp=True``: compute ∂image_points / ∂intrinsic_parameters (empty)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   NoIntrinsic._apply
   NoIntrinsic._invert
