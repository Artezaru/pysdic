pysdic.NoDistortion
========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


NoDistortion class
----------------------

.. autoclass:: NoDistortion
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`NoDistortion` class takes no argument and is instantiated directly with ``NoDistortion()``.


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.NoDistortion` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.NoDistortion` has no parameters: ``parameters`` is an empty vector.


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform distortion projections:

- ``apply``: maps ``normalized_points`` → ``distorted_points`` (identity).
- ``invert``: maps ``distorted_points`` → ``normalized_points`` (identity).

Jacobians with respect to the input points and distortion parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂distorted_points / ∂normalized_points (identity matrix)
- ``dp=True``: compute ∂distorted_points / ∂distortion_parameters (empty)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   NoDistortion._apply
   NoDistortion._invert
