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

A :class:`SDICTemporalRegularizationTerm` is created with the unknowns of the problem (the same
:class:`pysdic.SDICUnknowns` as the correlation term), the order of the derivative and its options. The analysis must
have ``times``. The term is **immutable**: the methods below create a new term.

.. code-block:: python

   regularization = SDICTemporalRegularizationTerm(unknowns, order=2, weight=1e-2)
   result = pysolvegn.solve([correlation.term, regularization.term], unknowns.pack())

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
   SDICTemporalRegularizationTerm.pack

Configuration
-------------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICTemporalRegularizationTerm.order
   SDICTemporalRegularizationTerm.weight
   SDICTemporalRegularizationTerm.options
   SDICTemporalRegularizationTerm.evaluated_frames
