pysdic.ParametrizationResult
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.ParametrizationResult` provides a standardized way to manage the results of parametrization operations:
the encoded or decoded parameters ``p`` and the Jacobian matrix ``jacobian_dp`` (or None if not computed).

.. seealso::

   - :class:`pysdic.Parametrization`: The base class for parametrizations, which defines the interface for encoding and decoding parameters.

ParametrizationResult class
---------------------------

.. autoclass:: ParametrizationResult
   :members:

