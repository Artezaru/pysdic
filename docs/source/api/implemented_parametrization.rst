Implemented Parametrizations
============================

The package ``pysdic`` provides a set of implemented parametrizations :math:`P` that can be used
to change the parametric representation of a transformation, mapping input parameters
:math:`\vec{p_i} \in \mathbb{R}^{n_i}` to output parameters :math:`\vec{p_o} \in \mathbb{R}^{n_o}`:

.. math::

    \vec{p_o} = P(\vec{p_i})

They are typically combined with a transformation :math:`T` in a :class:`pysdic.ReparametrizedTransformation`
to optimize the transformation with the new parameters :math:`\vec{p_i}`:
:math:`\tilde{T}(\vec{x_i}; \vec{p_i}) = T(\vec{x_i}; P(\vec{p_i}))`.

- :doc:`SubsetParametrization <morphisms/SubsetParametrization>`: Select a subset of the parameters to be optimized, the other parameters being kept fixed (:math:`\vec{p_o}[\mathcal{S}] = \vec{p_i}`).
- :doc:`AffineParametrization <morphisms/AffineParametrization>`: Affine map of the parameters :math:`\vec{p_o} = \mathbf{M} \vec{p_i} + \vec{p_0}`, where each column of :math:`\mathbf{M}` is a mode driven by one input parameter.

A :class:`pysdic.SubsetParametrization` is a particular :class:`pysdic.AffineParametrization` (with a selection matrix as
modes), implemented without dense matrix products. Parametrizations can be chained with :class:`pysdic.ComposedParametrization`
or with the syntax ``P2 @ P1``.

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/SubsetParametrization.rst
   morphisms/AffineParametrization.rst
