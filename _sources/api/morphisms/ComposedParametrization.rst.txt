pysdic.ComposedParametrization
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.ComposedParametrization` provides a standardized way to manage composed parametrizations,
allowing for the combination of multiple parametrizations :math:`[P_1, P_2, \dots, P_n]` into a single
parametrization :math:`P = P_n \circ \dots \circ P_1`.

The syntax ``P2 @ P1`` is equivalent to ``ComposedParametrization([P1, P2])``.


ComposedParametrization class
-----------------------------

.. autoclass:: ComposedParametrization
   :no-members:
   :show-inheritance:


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.ComposedParametrization` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.ComposedParametrization` defines or redefines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ComposedParametrization.input_dim
   ComposedParametrization.n_parametrizations
   ComposedParametrization.output_dim
   ComposedParametrization.parametrizations


Methods
^^^^^^^

The :class:`pysdic.ComposedParametrization` implements
the following backend methods to encode and decode the
composed parametrization:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ComposedParametrization._encode
   ComposedParametrization._decode

.. note::

   The ``encode`` and ``decode`` methods of the composition accept the additional argument
   ``kwargs``: a list with one dictionary of keyword arguments per parametrization, in the order
   :math:`[P_1, \dots, P_n]` (the same order for ``decode``, even if the decoding is applied in reverse).

