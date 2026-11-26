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

A :class:`SDICFixedParametrization` is created with the unknowns of the problem (the same
:class:`pysdic.SDICUnknowns` as its terms), the selected vertices and components. The values that are not selected
are read from the analysis **at each call**: after a convergence written in the analysis, the same parametrization
can be used again. It is **immutable**: the methods below create a new parametrization.

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICFixedParametrization.with_unknowns
   SDICFixedParametrization.with_options

Reduced vector
--------------

The solver optimizes the reduced vector :math:`\vec{q}` (the selected entries of :math:`\vec{p}`, total values):

.. warning::

   **With a parametrization, always use** :meth:`SDICFixedParametrization.reparametrized_pack`,
   :meth:`SDICFixedParametrization.reparametrized_unpack` **and** :meth:`SDICFixedParametrization.reparametrized_apply`
   on the vectors of the solver. ``unknowns.pack``, ``unknowns.unpack`` and ``unknowns.apply`` work on the complete
   vector :math:`\vec{p}`: giving them :math:`\vec{q}` raises an error (wrong size) or, worse, misplaces the values if
   the sizes happen to match.

   =========================  =================================================
   Without parametrization    With a parametrization
   =========================  =================================================
   ``unknowns.pack()``        ``parametrization.reparametrized_pack()``
   ``unknowns.unpack(p)``     ``parametrization.reparametrized_unpack(q)``
   ``unknowns.apply(p)``      ``parametrization.reparametrized_apply(q)``
   =========================  =================================================

.. code-block:: python

   parametrization = SDICFixedParametrization(unknowns, vertices=vertices, components=[0, 1])
   result = pysolvegn.solve([correlation.term], parametrization.reparametrized_pack(),
                            parametrization=parametrization.parametrization)
   parametrization.reparametrized_apply(result.parameters)  # NOT unknowns.apply(...)

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
