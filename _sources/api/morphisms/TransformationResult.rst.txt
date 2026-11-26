pysdic.TransformationResult
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.TransformationResult` provides a standardized way to manage the results of transformation operations:
the transformed points ``x`` and the Jacobian matrices ``jacobian_dx`` and ``jacobian_dp`` (or None if not computed).

.. seealso::

   - :class:`pysdic.Transformation`: The base class for transformations, which defines the interface for applying and inverting transformations.

TransformationResult class
---------------------------

.. autoclass:: TransformationResult
   :members:

