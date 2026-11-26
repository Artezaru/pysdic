Morphisms (Transformations + Parametrizations)
============================================================================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

A transformation is a mathematical operation that maps points
from one space :math:`\mathbb{R}^{n_i}` to another space :math:`\mathbb{R}^{n_o}`.
In the context of Stereo Digital Image Correlation (SDIC),
transformations are used to describe the relationship between
3D coordinates in the physical space and their
corresponding 2D coordinates in the image space.

Transformations
----------------------------

All transformation models derive from the abstract base
class :class:`pysdic.Transformation`,
which defines a common interface for applying and inverting transformations.

.. math::

    \vec{x_o} = T(\vec{x_i}; \vec{p}) \quad \vec{x_i} \in \mathbb{R}^{n_i}, \vec{x_o} \in \mathbb{R}^{n_o}, \vec{p} \in \mathbb{R}^{n_p}

- :doc:`Transformation <morphisms/Transformation>`: Abstract base class for all transformations in ``pysdic``.
- :doc:`TransformationResult <morphisms/TransformationResult>`: Data class for storing the results of applying or inverting a transformation, including transformed points and Jacobians.
- :doc:`IdentityTransformation <morphisms/IdentityTransformation>`: A transformation that maps input points to output points without any change (or the canonical projection or embedding if :math:`n_i \neq n_o`).

Each transformation :math:`T` defines a consistent interface composed of:

**Parameters**

- ``parameters`` (property): Optimisable parameters :math:`\vec{p}` of the transformation.
  Jacobians are computed with respect to these parameters. They are set with the property or with
  ``set_parameters``.
- ``is_locked`` and ``locked_message`` (properties): a transformation owned by another object of the package
  (e.g. the transformation of a :class:`pysdic.ReparametrizedTransformation`) is **locked** by its owner:
  setting its parameters raises a ``RuntimeError`` explaining why, unless
  ``set_parameters(..., override_lock=True)``. There is no public method to lock or unlock a transformation,
  and a copy (``copy``) is always unlocked (see :doc:`Core Tools <core>`).

**Core methods**

- ``apply``: Applies the transformation :math:`T` to input points.
  Returns a :class:`pysdic.TransformationResult` containing transformed points and associated Jacobians.

- ``invert``: Applies the inverse map :math:`T^{-1}`.
  Returns a :class:`pysdic.TransformationResult` containing inverse-mapped points and corresponding Jacobians.

**Capability flags**

- ``has_apply``, ``has_apply_dx``, ``has_apply_dp``, ``has_invert``, ``has_invert_dx``, ``has_invert_dp``:
  whether the methods and their Jacobians are available.
- ``is_left_invertible``, ``is_right_invertible``, ``is_invertible``: whether ``invert`` is a left
  and/or right inverse of ``apply``.

The Jacobians and the inverse map can be verified numerically with the ``check_*`` methods.

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/Transformation.rst
   morphisms/TransformationResult.rst
   morphisms/IdentityTransformation.rst


Parametrizations
----------------------------

Parametrizations allow to change the parametric representation
of the parameters of a transformation,
without changing the underlying transformation itself.

.. math::

    \vec{p_o} = P(\vec{p_i}) \quad \vec{p_i} \in \mathbb{R}^{n_i}, \vec{p_o} \in \mathbb{R}^{n_o}

such that the transformation is expressed with the new parameters :math:`\vec{p_i}`:

.. math::

    \tilde{T}(\vec{x_i}; \vec{p_i}) = T(\vec{x_i}; P(\vec{p_i}))

- :doc:`Parametrization <morphisms/Parametrization>`: Abstract base class for all parametrizations in ``pysdic``.
- :doc:`ParametrizationResult <morphisms/ParametrizationResult>`: Data class for storing the results of encoding or decoding parameters, including transformed parameters and Jacobians.

Each parametrization :math:`P` defines a consistent interface composed of:

**Core methods**

- ``encode``: Applies the parametrization :math:`P` to input parameters.
  Returns a :class:`pysdic.ParametrizationResult` containing transformed parameters and associated Jacobians.

- ``decode``: Applies the decoding map :math:`P^{-1}`.
  Returns a :class:`pysdic.ParametrizationResult` containing inverse-mapped parameters and corresponding Jacobians.

**Capability flags**

- ``has_encode``, ``has_encode_dp``, ``has_decode``, ``has_decode_dp``: whether the methods and their Jacobians are available.
- ``is_left_invertible``, ``is_right_invertible``, ``is_invertible``: whether ``decode`` is a left
  and/or right inverse of ``encode``.

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/Parametrization.rst
   morphisms/ParametrizationResult.rst


Composing transformations and parametrizations
-----------------------------------------------

To facilitate the construction of complex transformation models,
the package provides classes for:

- :doc:`ComposedTransformation <morphisms/ComposedTransformation>`: Composing multiple transformations :math:`[T_1, T_2, \dots, T_k]` into a single transformation :math:`T = T_k \circ \dots \circ T_2 \circ T_1` (:math:`T_1` is applied first). The syntax ``T2 @ T1`` is equivalent to ``ComposedTransformation([T1, T2])``.
- :doc:`ComposedParametrization <morphisms/ComposedParametrization>`: Composing multiple parametrizations :math:`[P_1, P_2, \dots, P_k]` into a single parametrization :math:`P = P_k \circ \dots \circ P_2 \circ P_1` (:math:`P_1` is applied first). The syntax ``P2 @ P1`` is equivalent to ``ComposedParametrization([P1, P2])``.
- :doc:`ReparametrizedTransformation <morphisms/ReparametrizedTransformation>`: Applying a parametrization to the parameters of a transformation :math:`\tilde{T}(\vec{x_i}; \vec{q}) = T(\vec{x_i}; P(\vec{q}))`.

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/ComposedTransformation.rst
   morphisms/ComposedParametrization.rst
   morphisms/ReparametrizedTransformation.rst