pysdic.Parametrization
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Parametrization` provides a standardized way to manage parametrizations
:math:`P` mapping input parameters :math:`\vec{p_i} \in \mathbb{R}^{n_i}` to output parameters
:math:`\vec{p_o} \in \mathbb{R}^{n_o}`, allowing for the modification of the parametric
representation of a transformation without changing the underlying transformation
(see :class:`pysdic.ReparametrizedTransformation`).

.. math::

   \vec{p_o} = P(\vec{p_i})


Parametrization class
------------------------

.. autoclass:: Parametrization
   :no-members:
   :show-inheritance:

Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Parametrization` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.Parametrization` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.input_dim
   Parametrization.output_dim

The capabilities of a parametrization are exposed by the following read-only flags:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.has_encode
   Parametrization.has_encode_dp
   Parametrization.has_decode
   Parametrization.has_decode_dp
   Parametrization.is_invertible
   Parametrization.is_left_invertible
   Parametrization.is_right_invertible


Methods
^^^^^^^

The :class:`pysdic.Parametrization` defines the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.copy


Applying the parametrization methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform the parametrization from input parameters :math:`\vec{p_i}` to output parameters
:math:`\vec{p_o}` (``encode``) or the decoding map from :math:`\vec{p_o}` to :math:`\vec{p_i}` (``decode``),
use the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.encode
   Parametrization.decode

The output of the parametrization is a :class:`pysdic.ParametrizationResult` object, which
contains the encoded or decoded parameters (``p``) and the Jacobian matrix (``jacobian_dp``)
if requested.

.. note::

   Only single parameter vectors are supported: shape :math:`(n_i,)` for ``encode`` and
   :math:`(n_o,)` for ``decode`` (no batch dimension).

.. note::

   You can also use the ``__call__`` method to apply the parametrization,
   which is an alias for the ``encode`` method: ``parametrization(p_i)``.

.. note::

   Two parametrizations can be composed with the syntax ``P2 @ P1``, which is equivalent to
   ``ComposedParametrization([P1, P2])`` (see :class:`pysdic.ComposedParametrization`).


Numerical checks
^^^^^^^^^^^^^^^^

The implementation of a parametrization (Jacobians and decoding map) can be verified numerically
with the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.check_encode_dp
   Parametrization.check_decode_dp
   Parametrization.check_left_invertible
   Parametrization.check_right_invertible


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.Parametrization` inherits serialization methods from the following mixin classes:

- Class :class:`pysdic.core.serialization.JsonSerializerMixin`
- Class :class:`pysdic.core.serialization.Hdf5SerializerMixin`

.. seealso::

   The documentation :doc:`Serialization Mixins <../core>` for more details on
   the available serialization methods and formats.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Parametrization.from_dict
   Parametrization.to_dict

