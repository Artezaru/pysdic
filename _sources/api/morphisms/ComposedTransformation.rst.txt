pysdic.ComposedTransformation
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.ComposedTransformation` provides a standardized way to manage composed transformations,
allowing for the combination of multiple transformations :math:`[T_1, T_2, \dots, T_n]` into a single
transformation :math:`T = T_n \circ \dots \circ T_1`.

The parameters of the composition are the concatenation of the parameters of the transformations,
in the order of the composition. The syntax ``T2 @ T1`` is equivalent to ``ComposedTransformation([T1, T2])``.


ComposedTransformation class
-----------------------------

.. autoclass:: ComposedTransformation
   :no-members:
   :show-inheritance:


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.ComposedTransformation` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.ComposedTransformation` defines or redefines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ComposedTransformation.input_dim
   ComposedTransformation.n_parameters
   ComposedTransformation.n_transformations
   ComposedTransformation.output_dim
   ComposedTransformation.parameters
   ComposedTransformation.transformations


Methods
^^^^^^^

The :class:`pysdic.ComposedTransformation` redefines the following method to split the
parameters between the transformations:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ComposedTransformation.set_parameters

The :class:`pysdic.ComposedTransformation` implements
the following backend methods to apply and invert the
composed transformation:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ComposedTransformation._apply
   ComposedTransformation._invert

.. note::

   The ``apply`` and ``invert`` methods of the composition accept the additional arguments
   ``kwargs`` (a list with one dictionary of keyword arguments per transformation, in the order
   :math:`[T_1, \dots, T_n]`) and ``bypass_dp`` (a sequence of booleans to skip the Jacobian
   with respect to the parameters of some transformations, whose columns are filled with NaN).

