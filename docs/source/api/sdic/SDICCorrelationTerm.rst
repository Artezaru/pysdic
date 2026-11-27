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

A :class:`SDICCorrelationTerm` is created with the analysis, the unknowns (a :class:`pysdic.SDICUnknowns` created by
the user, which does not know the analysis), the integration points, the pairings and its options (mode, gray level
correction, weight, loss, sparse matrices, caches).

A term is **immutable**: the methods below create a new term with the same configuration except the given changes
(new caches; the gray level groups given with ``gray_level_groups`` are shared). The shape functions, the weights and
the sparse structure of the integration points are shared too (the mesh and the points are locked), except by
:meth:`SDICCorrelationTerm.with_integration_points`. The previous term stays valid on its own unknowns.

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

The term does not build the vectors of unknowns: use the analysis on the complete vector :math:`\vec{p}`
(:meth:`pysdic.SDICAnalysis.pack_unknowns`, :meth:`pysdic.SDICAnalysis.unpack_unknowns`,
:meth:`pysdic.SDICAnalysis.apply_unknowns`), or the parametrization on the reduced vector :math:`\vec{q}`.

.. code-block:: python

   result = pysolvegn.solve([correlation.term], analysis.pack_unknowns(unknowns))
   analysis.apply_unknowns(unknowns, result.parameters)

.. seealso::

   :class:`pysdic.SDICUnknowns` for the order of :math:`\vec{p}`, and :class:`pysdic.SDICFixedParametrization` for
   the reduced vector :math:`\vec{q}` of a parametrization (``reparametrized_pack``, ``reparametrized_unpack``,
   ``reparametrized_apply``).

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICCorrelationTerm.term
   SDICCorrelationTerm.unknowns
   SDICCorrelationTerm.analysis

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