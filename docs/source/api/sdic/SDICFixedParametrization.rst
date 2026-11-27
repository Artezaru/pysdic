pysdic.SDICFixedParametrization
===============================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


SDICFixedParametrization class
------------------------------

.. autoclass:: SDICFixedParametrization
   :no-members:

Instantiation and new parametrizations
--------------------------------------

A :class:`SDICFixedParametrization` is created with the analysis and the unknowns of the problem (the same
:class:`pysdic.SDICAnalysis` and :class:`pysdic.SDICUnknowns` objects as its terms), the selected vertices and components. The values that are not selected
are read from the analysis **at each call**: after a convergence written in the analysis, the same parametrization
can be used again. It is **immutable**: the methods below create a new parametrization.

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICFixedParametrization.with_unknowns
   SDICFixedParametrization.with_options

Reduced vector
--------------

The solver optimizes the reduced vector :math:`\vec{q}` (the selected entries of :math:`\vec{p}`, total values).
**With a parametrization, always use these methods on the vectors of the solver**, never the ``*_unknowns`` methods
of the analysis (see the warning and the table in the description of the class above).

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICFixedParametrization.reparametrized_pack
   SDICFixedParametrization.reparametrized_unpack
   SDICFixedParametrization.reparametrized_apply
   SDICFixedParametrization.p_func

Attributes
----------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICFixedParametrization.parametrization
   SDICFixedParametrization.unknowns
   SDICFixedParametrization.analysis
   SDICFixedParametrization.indices
   SDICFixedParametrization.n_reparametrized
   SDICFixedParametrization.sparse