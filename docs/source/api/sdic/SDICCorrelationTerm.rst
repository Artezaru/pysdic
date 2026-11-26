pysdic.SDICCorrelationTerm
==========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


SDICCorrelationTerm class
-------------------------

.. autoclass:: SDICCorrelationTerm
   :no-members:

Instantiation and new terms
---------------------------

A :class:`SDICCorrelationTerm` is created with the unknowns (a :class:`pysdic.SDICUnknowns` created by the user),
the integration points, the pairings and its options (mode, gray level correction, weight, loss, sparse matrices,
caches). Its analysis is the analysis of the unknowns.

A term is **immutable**: the methods below create a new term with the same configuration except the given changes
(new caches and structures; the gray level groups given with ``gray_level_groups`` are shared). The previous term
stays valid on its own unknowns.

.. code-block:: python

   correlation_2 = correlation.with_unknowns(unknowns_2, pairings=pairings_2)  # Next frame
   correlation_3 = correlation.with_options(weight=10.0, use_cache=True)

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICCorrelationTerm.with_unknowns
   SDICCorrelationTerm.with_pairings
   SDICCorrelationTerm.with_integration_points
   SDICCorrelationTerm.with_options

Term and unknowns
-----------------

:meth:`SDICCorrelationTerm.pack`, :meth:`SDICCorrelationTerm.unpack` and :meth:`SDICCorrelationTerm.apply` are those of
:attr:`SDICCorrelationTerm.unknowns`: they work on the complete vector :math:`\vec{p}` (total displacements).

.. warning::

   **With a parametrization, always use** ``parametrization.reparametrized_pack()``, ``reparametrized_unpack(q)``
   **and** ``reparametrized_apply(q)`` on the vectors of the solver (see :class:`pysdic.SDICFixedParametrization`).

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICCorrelationTerm.term
   SDICCorrelationTerm.unknowns
   SDICCorrelationTerm.analysis
   SDICCorrelationTerm.n_unknowns
   SDICCorrelationTerm.pack
   SDICCorrelationTerm.unpack
   SDICCorrelationTerm.apply

Residuals
---------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICCorrelationTerm.compute_residuals

Configuration
-------------

Read-only properties (see the ``with_*`` methods to create a term with other values).

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICCorrelationTerm.integration_points
   SDICCorrelationTerm.pairings
   SDICCorrelationTerm.optimized_frames
   SDICCorrelationTerm.optimized_transformations
   SDICCorrelationTerm.options
   SDICCorrelationTerm.mode
   SDICCorrelationTerm.gl_correction
   SDICCorrelationTerm.gray_level_groups
   SDICCorrelationTerm.weight
   SDICCorrelationTerm.loss
   SDICCorrelationTerm.loss_scale
   SDICCorrelationTerm.filter_nans
   SDICCorrelationTerm.sparse
   SDICCorrelationTerm.use_cache
