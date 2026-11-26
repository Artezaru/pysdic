pysdic.SubsetParametrization
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


SubsetParametrization class
------------------------------------

.. autoclass:: SubsetParametrization
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`SubsetParametrization` class is instantiated directly with its constructor:

- **With a boolean mask** of shape :math:`(n_o,)`: ``SubsetParametrization(mask)`` (the non-selected parameters are fixed to zero).
- **With a boolean mask and fixed values** :math:`\vec{v}` of shape :math:`(n_o,)`: ``SubsetParametrization(mask, fixed_values)``
  (the fixed values must be ``0.0`` at the selected indices, e.g. ``numpy.where(mask, 0.0, transformation.parameters)``).


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.SubsetParametrization` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.SubsetParametrization` defines the following attributes
(the constants ``mask``, ``indices`` and ``fixed_values`` are read-only):

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SubsetParametrization.fixed_values
   SubsetParametrization.indices
   SubsetParametrization.input_dim
   SubsetParametrization.mask
   SubsetParametrization.output_dim


Encoding and decoding
---------------------

Use the ``encode`` and ``decode`` methods from
:class:`pysdic.Parametrization` to perform the parametrization:

- ``encode``: maps ``p_i`` → ``p_o`` (``p_i`` inserted at the selected indices, the fixed values elsewhere).
- ``decode``: maps ``p_o`` → ``p_i`` (extraction of the selected parameters, left inverse of ``encode``).

The Jacobian with respect to the parameters can be computed by enabling the ``dp`` flag:

- ``dp=True``: compute ∂p_o / ∂p_i (constant selection matrix :math:`\mathbf{E} = \mathbf{I}_{n_o}[:, \mathcal{S}]`)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SubsetParametrization._encode
   SubsetParametrization._decode
