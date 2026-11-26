pysdic.Transformation
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Transformation` provides a standardized way to manage parametric transformations
:math:`T` from :math:`\mathbb{R}^{n_i}` to :math:`\mathbb{R}^{n_o}`, where the transformation is
defined by a set of parameters :math:`\vec{p} \in \mathbb{R}^{n_p}`.

.. math::

   \vec{x_o} = T(\vec{x_i}; \vec{p})


Transformation class
------------------------

.. autoclass:: Transformation
   :no-members:
   :show-inheritance:

Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Transformation` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.Transformation` defines the following attributes:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.input_dim
   Transformation.is_locked
   Transformation.is_setup
   Transformation.locked_message
   Transformation.n_parameters
   Transformation.output_dim
   Transformation.parameters

The capabilities of a transformation are exposed by the following read-only flags:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.has_apply
   Transformation.has_apply_dx
   Transformation.has_apply_dp
   Transformation.has_invert
   Transformation.has_invert_dx
   Transformation.has_invert_dp
   Transformation.is_invertible
   Transformation.is_left_invertible
   Transformation.is_right_invertible


Methods
^^^^^^^

The :class:`pysdic.Transformation` defines the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.copy
   Transformation.set_parameters

.. note::

   A locked transformation (see :attr:`Transformation.is_locked`) cannot be modified with
   :meth:`Transformation.set_parameters` unless ``override_lock=True`` is given.
   The lock is managed internally by the objects that own the transformation.


Applying the transformation methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform the transformation from :math:`\mathbb{R}^{n_i}` to :math:`\mathbb{R}^{n_o}`
(``apply``) or its inverse map from :math:`\mathbb{R}^{n_o}` to :math:`\mathbb{R}^{n_i}` (``invert``),
use the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.apply
   Transformation.invert

The output of the transformation is a :class:`pysdic.TransformationResult` object, which
contains the transformed points (``x``) and the Jacobian matrices with respect to the
points (``jacobian_dx``) and to the parameters (``jacobian_dp``) if requested.

.. note::

   You can also use the ``__call__`` method to apply the transformation,
   which is an alias for the ``apply`` method: ``transformation(x_i)``.

.. note::

   Two transformations can be composed with the syntax ``T2 @ T1``, which is equivalent to
   ``ComposedTransformation([T1, T2])`` (see :class:`pysdic.ComposedTransformation`).


Numerical checks
^^^^^^^^^^^^^^^^

The implementation of a transformation (Jacobians and inverse map) can be verified numerically
with the following methods:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.check_apply_dx
   Transformation.check_apply_dp
   Transformation.check_invert_dx
   Transformation.check_invert_dp
   Transformation.check_left_invertible
   Transformation.check_right_invertible


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.Transformation` inherits serialization methods from the following mixin classes:

- Class :class:`pysdic.core.serialization.JsonSerializerMixin`
- Class :class:`pysdic.core.serialization.Hdf5SerializerMixin`

.. seealso::

   The documentation :doc:`Serialization Mixins <../core>` for more details on
   the available serialization methods and formats.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Transformation.from_dict
   Transformation.to_dict