pysdic.SDICTemporalRegularizationTerm
=====================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


SDICTemporalRegularizationTerm class
------------------------------------

.. autoclass:: SDICTemporalRegularizationTerm
   :no-members:

Instantiation and new terms
---------------------------

A :class:`SDICTemporalRegularizationTerm` is created with the analysis and the unknowns of the problem (the same
:class:`pysdic.SDICAnalysis` and :class:`pysdic.SDICUnknowns` objects as the correlation term), the order of the derivative and its options. The analysis must
have ``times``. The term is **immutable**: the methods below create a new term.

.. code-block:: python

   regularization = SDICTemporalRegularizationTerm(analysis, unknowns, order=2, weight=1e-2)
   result = pysolvegn.solve([correlation.term, regularization.term], analysis.pack_unknowns(unknowns))
   analysis.apply_unknowns(unknowns, result.parameters)

   regularization_2 = regularization.with_unknowns(unknowns_2)              # New unknowns
   softer = regularization.with_options(order=1, weight=1e-1)               # Other options

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICTemporalRegularizationTerm.with_unknowns
   SDICTemporalRegularizationTerm.with_options

Term and unknowns
-----------------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICTemporalRegularizationTerm.term
   SDICTemporalRegularizationTerm.unknowns
   SDICTemporalRegularizationTerm.analysis

The term does not build the vectors of unknowns: they are built by the analysis (``pack_unknowns``,
``unpack_unknowns``, ``apply_unknowns``).

.. seealso::

   :class:`pysdic.SDICUnknowns` for the order of :math:`\vec{p}`, and :class:`pysdic.SDICFixedParametrization` for
   the reduced vector :math:`\vec{q}` of a parametrization (``reparametrized_pack``, ``reparametrized_unpack``,
   ``reparametrized_apply``).

Configuration
-------------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICTemporalRegularizationTerm.order
   SDICTemporalRegularizationTerm.weight
   SDICTemporalRegularizationTerm.options
   SDICTemporalRegularizationTerm.evaluated_frames