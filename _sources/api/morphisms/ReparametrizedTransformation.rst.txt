pysdic.ReparametrizedTransformation
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.ReparametrizedTransformation` provides a standardized way to manage reparametrized transformations,
allowing for the application of a parametrization :math:`P` to the parameters of a transformation :math:`T`:

.. math::

   \vec{x_o} = \tilde{T}(\vec{x_i}; \vec{q}) = T(\vec{x_i}; P(\vec{q}))

The reparametrized transformation stores the new parameters :math:`\vec{q}`. The underlying transformation
and parametrization are copied, and the parameters :math:`\vec{p} = P(\vec{q})` of the underlying
transformation are locked: they can only be modified through the reparametrized transformation.


ReparametrizedTransformation class
----------------------------------

.. autoclass:: ReparametrizedTransformation
   :no-members:
   :show-inheritance:


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.ReparametrizedTransformation` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.ReparametrizedTransformation` defines or redefines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ReparametrizedTransformation.input_dim
   ReparametrizedTransformation.n_parameters
   ReparametrizedTransformation.output_dim
   ReparametrizedTransformation.parametrization
   ReparametrizedTransformation.transformation


Methods
^^^^^^^

The :class:`pysdic.ReparametrizedTransformation` redefines the following method to update
the parameters :math:`\vec{p} = P(\vec{q})` of the underlying transformation:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ReparametrizedTransformation.set_parameters

The :class:`pysdic.ReparametrizedTransformation` implements
the following backend methods to apply and invert the
reparametrized transformation:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ReparametrizedTransformation._apply
   ReparametrizedTransformation._invert

