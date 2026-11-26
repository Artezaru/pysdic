pysdic.IdentityTransformation
==============================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


IdentityTransformation class
------------------------------

.. autoclass:: IdentityTransformation
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`IdentityTransformation` class is instantiated directly with its constructor:

- **Identity of** :math:`\mathbb{R}^{n_i}`: ``IdentityTransformation(input_dim=n_i)``.
- **Projection or embedding** from :math:`\mathbb{R}^{n_i}` to :math:`\mathbb{R}^{n_o}`: ``IdentityTransformation(input_dim=n_i, output_dim=n_o)``.


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.IdentityTransformation` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.IdentityTransformation` has no parameters: ``parameters`` is an empty vector.

The :class:`pysdic.IdentityTransformation` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IdentityTransformation.input_dim
   IdentityTransformation.output_dim


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform the transformation:

- ``apply``: maps ``input_points`` (:math:`n_i`) → ``output_points`` (:math:`n_o`).
- ``invert``: maps ``output_points`` (:math:`n_o`) → ``input_points`` (:math:`n_i`).

Jacobians with respect to the input points and parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂output_points / ∂input_points (truncated or zero-padded identity matrix)
- ``dp=True``: compute ∂output_points / ∂parameters (empty)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IdentityTransformation._apply
   IdentityTransformation._invert
