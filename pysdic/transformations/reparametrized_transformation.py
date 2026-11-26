"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from numpy.typing import ArrayLike

import numpy

from ..parametrizations.parametrization import Parametrization
from .transformation import Transformation


# Reasons of the permanent locks of the transformation and the parametrization owned by a
# ReparametrizedTransformation (added after "Cannot ...: this <class> is locked.")
_TRANSFORMATION_LOCK_MESSAGE = (
    "It is used by a ReparametrizedTransformation, which keeps its parameters p equal to "
    "P(q), where P is the parametrization and q the parameters of the "
    "ReparametrizedTransformation. Set q instead (reparametrized.parameters = q): p is then "
    "updated automatically. The transformation stays locked as long as the "
    "ReparametrizedTransformation exists. To get a transformation that can be modified "
    "independently, use transformation.copy(), or give a copy to the "
    "ReparametrizedTransformation: ReparametrizedTransformation(transformation.copy(), "
    "parametrization)."
)

_PARAMETRIZATION_LOCK_MESSAGE = (
    "It is used by a ReparametrizedTransformation, which assumes that it never changes (the "
    "parameters p = P(q) of its transformation and the cached Jacobian dP/dq depend on it). "
    "The parametrization stays locked as long as the ReparametrizedTransformation exists. To "
    "use a modified parametrization, create a new ReparametrizedTransformation with a modified copy "
    "(parametrization.copy()), or give a copy to the ReparametrizedTransformation: "
    "ReparametrizedTransformation(transformation, parametrization.copy())."
)


def _already_locked_error(kind: str, obj: Any, copy_call: str) -> ValueError:
    r"""
    Build the error raised when the transformation or the parametrization given to a
    :class:`pysdic.ReparametrizedTransformation` is already locked.

    .. important::

        Internal use only.

    """
    message = obj.locked_message
    reason = (
        f"Reason given by the lock: {message}" if message else "No reason was given with the lock."
    )
    return ValueError(
        f"Cannot create the ReparametrizedTransformation: the {kind} "
        f"({obj.__class__.__name__}) is locked, so it is already used by another object that "
        f"relies on it (for example another ReparametrizedTransformation or an SDIC analysis). "
        f"{reason} The ReparametrizedTransformation must own its {kind} and lock it "
        f"permanently, which would break this other object. Give a copy instead: "
        f"ReparametrizedTransformation({copy_call})."
    )


class ReparametrizedTransformation(Transformation):
    r"""
    A :class:`pysdic.Transformation` whose **parameters** are described by a
    :class:`pysdic.Parametrization`.

    Lets :math:`T` be a transformation that maps input points :math:`\vec{x_i}` to
    output points :math:`\vec{x_o}` with parameters :math:`\vec{p}`:

    .. math::

        \vec{x_o} = T(\vec{x_i}; \vec{p})

    In some cases, it can be useful to express the parameters :math:`\vec{p}` with
    another representation :math:`\vec{q}` related to the original parameters
    by a parametrization :math:`P`:

    .. math::

        \vec{p} = P(\vec{q})

    The reparametrized transformation :math:`\tilde{T}` has the parameters :math:`\vec{q}`:

    .. math::

        \vec{x_o} = \tilde{T}(\vec{x_i}; \vec{q}) = T(\vec{x_i}; P(\vec{q}))

    For example, a :class:`pysdic.SubsetParametrization` allows to optimize only some parameters
    of a transformation while the others are kept fixed.

    The Jacobian with respect to the input points is the one of :math:`T`, and the Jacobian with
    respect to the parameters is obtained with the chain rule:

    .. math::

        \mathbf{J}_q = \frac{\partial \tilde{T}}{\partial \vec{q}} = \frac{\partial T}{\partial \vec{p}} \cdot \frac{\partial P}{\partial \vec{q}}

    .. seealso::

        - Class :class:`pysdic.Transformation` for the base class of transformations.
        - Class :class:`pysdic.Parametrization` for the base class of parametrizations.

    .. note::

        The reparametrized transformation stores the parameters :math:`\vec{q}`. Each time they
        are set, the parameters :math:`\vec{p} = P(\vec{q})` of the underlying transformation
        are updated.

    .. important::

        The transformation and the parametrization are **not copied**: the reparametrized
        transformation uses the given objects and **locks them permanently** (see
        ``is_locked`` and ``locked_message``), with messages explaining why:

        - the parameters :math:`\vec{p}` of the transformation can only be modified through
          the parameters :math:`\vec{q}` of the reparametrized transformation, so that they
          are always equal to :math:`P(\vec{q})`;
        - the parametrization cannot be modified (for the parametrizations with modifiable
          constants), since :math:`\vec{p}` and the cached Jacobian
          :math:`\frac{\partial P}{\partial \vec{q}}` depend on it.

        To keep a modifiable transformation (or parametrization), give a copy:
        ``ReparametrizedTransformation(transformation.copy(), parametrization)``.

        A transformation or a parametrization that is already locked (for example by another
        reparametrized transformation) is refused. If one of them is unlocked from outside
        (with the internal method ``_set_locked``), the reparametrized transformation refuses
        any later modification of :math:`\vec{q}`.

    .. note::

        Locking the reparametrized transformation itself (by an owner of the package) only
        locks its parameters :math:`\vec{q}`: the transformation and the parametrization stay
        locked in any case. A copy (``copy``) is unlocked and owns copies of the transformation
        and of the parametrization.

    .. note::

        The capability flags are derived from the flags of the transformation and the
        parametrization. The Jacobians with respect to the parameters are only available
        if the parametrization supports ``encode`` with Jacobian (``has_encode_dp``).


    Parameters
    ----------
    transformation : Transformation
        The transformation :math:`T` to reparametrize. The number of parameters of the
        transformation must match the output dimension of the parametrization.
        The transformation is not copied: it must not be locked, and it is locked
        permanently (see above).

    parametrization : Parametrization
        The parametrization :math:`P` mapping the new parameters :math:`\vec{q}` to the
        parameters :math:`\vec{p}` of the transformation. The parametrization must implement
        ``encode``. The parametrization is not copied: it must not be locked, and it is
        locked permanently (see above).

    parameters : Optional[ArrayLike], optional (default: ``None``)
        A 1D array of shape :math:`(n_q,)` containing the new parameters :math:`\vec{q}`, with
        :math:`n_q` the input dimension of the parametrization. The parameters of the
        transformation are then replaced by :math:`P(\vec{q})`. If None, the parameters
        :math:`\vec{q}` are obtained from the current parameters of the transformation
        (see below).


    Initialization of the parameters
    --------------------------------

    - If ``parameters`` is given, :math:`\vec{q}` is set to it and the parameters of the
      transformation are replaced by :math:`\vec{p} = P(\vec{q})`.
    - If ``parameters`` is None and the parameters :math:`\vec{p}` of the transformation are set,
      the parameters :math:`\vec{q} = P^{-1}(\vec{p})` are obtained with the ``decode`` method of
      the parametrization. The parameters :math:`\vec{p}` must be in the image of :math:`P`, i.e.
      :math:`P(P^{-1}(\vec{p})) = \vec{p}`, otherwise a ``ValueError`` is raised to avoid silently
      modifying the parameters of the transformation. If the parametrization does not implement
      ``decode``, a ``ValueError`` is raised too: the parameters of the transformation are never
      discarded silently.
    - If ``parameters`` is None and the parameters of the transformation are not set, the
      reparametrized transformation is not set up either (set its parameters later).


    Examples
    --------
    Optimize only the translation along :math:`x` of a transformation with parameters
    :math:`(t_x, t_y)`, the translation along :math:`y` being fixed:

    .. code-block:: python

        transformation = ScaledTranslation2D(parameters=[1.0, 2.0])
        parametrization = SubsetParametrization(
            mask=[True, False],
            fixed_values=[0.0, 2.0],  # The value of t_y, and 0.0 for the optimized t_x
        )

        reparametrized = ReparametrizedTransformation(transformation, parametrization)
        reparametrized.parameters                     # array([1.])
        reparametrized.parameters = [5.0]
        transformation.parameters                     # array([5., 2.]) (the same object)
        transformation.parameters = [0.0, 0.0]        # RuntimeError: locked by the reparametrized transformation

    """

    def __init__(
        self,
        transformation: Transformation,
        parametrization: Parametrization,
        *,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        if not isinstance(transformation, Transformation):
            raise TypeError(
                f"transformation must be an instance of Transformation, got {type(transformation)}"
            )
        if not isinstance(parametrization, Parametrization):
            raise TypeError(
                f"parametrization must be an instance of Parametrization, got {type(parametrization)}"
            )
        if not parametrization.has_encode:
            raise ValueError(
                f"The parametrization of type '{parametrization.__class__.__name__}' must "
                f"implement encode to reparametrize a transformation."
            )
        if transformation.n_parameters != parametrization.output_dim:
            raise ValueError(
                f"Output dimension of the parametrization ({parametrization.output_dim}) must match "
                f"the number of parameters of the transformation ({transformation.n_parameters})."
            )
        if transformation.is_locked:
            raise _already_locked_error(
                "transformation", transformation, "transformation.copy(), parametrization"
            )
        if parametrization.is_locked:
            raise _already_locked_error(
                "parametrization", parametrization, "transformation, parametrization.copy()"
            )

        # The objects are shared (not copied): they are locked permanently below
        self._transformation = transformation
        self._parametrization = parametrization
        self._jacobian_encode: Optional[numpy.ndarray] = None  # Cache of dP/dq at q

        # Capability flags derived from the transformation and the parametrization
        self._has_apply = transformation.has_apply
        self._has_apply_dx = transformation.has_apply_dx
        self._has_apply_dp = transformation.has_apply_dp and parametrization.has_encode_dp
        self._has_invert = transformation.has_invert
        self._has_invert_dx = transformation.has_invert_dx
        self._has_invert_dp = (
            transformation.has_invert_dp and parametrization.has_encode_dp
        )
        self._is_left_invertible = transformation.is_left_invertible
        self._is_right_invertible = transformation.is_right_invertible

        if parameters is None and self._transformation.parameters is not None:
            # Never discard the parameters of the transformation: q = P^{-1}(p)
            parameters = self._decode_initial_parameters()

        # Sets q and p = P(q) (with override_lock: nothing is locked yet). If q is not set,
        # the parameters of the transformation are not set either.
        super().__init__(parameters=parameters)

        # Locked only once everything succeeded: a failed creation leaves them unlocked
        self._transformation._set_locked(True, message=_TRANSFORMATION_LOCK_MESSAGE)
        self._parametrization._set_locked(True, message=_PARAMETRIZATION_LOCK_MESSAGE)

    def _decode_initial_parameters(self) -> numpy.ndarray:
        # Decode the parameters q from the parameters p (set) of the transformation
        p = self._transformation.parameters
        transformation_name = self._transformation.__class__.__name__
        parametrization_name = self._parametrization.__class__.__name__

        if not self._parametrization.has_decode:
            raise ValueError(
                f"Cannot create the ReparametrizedTransformation: the transformation "
                f"({transformation_name}) has parameters p, but the parametrization "
                f"({parametrization_name}) cannot compute the corresponding parameters q (it does "
                f"not implement decode). Give the parameters q with the 'parameters' argument: "
                f"ReparametrizedTransformation(transformation, parametrization, parameters=q) "
                f"(the parameters of the transformation are then replaced by P(q)). The parameters "
                f"of the transformation are never discarded silently: to start without parameters, "
                f"unset them first with transformation.parameters = None."
            )

        q = self._parametrization.decode(p).p
        p_encoded = self._parametrization.encode(q).p
        if not numpy.allclose(p_encoded, p, rtol=1e-8, atol=1e-10):
            difference = float(numpy.max(numpy.abs(p_encoded - p)))
            raise ValueError(
                f"Cannot create the ReparametrizedTransformation: the parameters p of the "
                f"transformation ({transformation_name}) are not in the image of the "
                f"parametrization ({parametrization_name}): encode(decode(p)) differs from p "
                f"(maximum difference {difference:.3e}). Using q = decode(p) would silently replace "
                f"p by P(decode(p)). Either give the parameters q with the 'parameters' argument "
                f"(p is then replaced by P(q)), or use a parametrization able to represent p (for "
                f"a SubsetParametrization, fix the other parameters to their current values: "
                f"fixed_values=numpy.where(mask, 0.0, transformation.parameters))."
            )
        return q

    # ------
    # Properties
    # ------
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters :math:`n_q` of the reparametrized transformation,
        i.e. the input dimension of the parametrization.

        Returns
        -------
        int
            The number of parameters of the reparametrized transformation.

        """
        return self._parametrization.input_dim

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the transformation.

        Returns
        -------
        int
            The input dimension of the transformation.

        """
        return self._transformation.input_dim

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the transformation.

        Returns
        -------
        int
            The output dimension of the transformation.

        """
        return self._transformation.output_dim

    @property
    def transformation(self) -> Transformation:
        r"""
        [Get] the underlying transformation :math:`T`.

        .. note::

            The parameters of the underlying transformation are locked permanently and
            always equal to :math:`P(\vec{q})`. They can only be modified through the
            reparametrized transformation.

        Returns
        -------
        Transformation
            The underlying transformation (the object given at the creation, not a copy).

        """
        return self._transformation

    @property
    def parametrization(self) -> Parametrization:
        r"""
        [Get] the parametrization :math:`P` mapping the new parameters :math:`\vec{q}` to the
        parameters :math:`\vec{p}` of the transformation.

        .. note::

            The parametrization is locked permanently: it cannot be modified.

        Returns
        -------
        Parametrization
            The parametrization (the object given at the creation, not a copy).

        """
        return self._parametrization

    # ------
    # Lock
    # ------
    def _check_unlocked(self, action: str, *, override_lock: bool = False) -> None:
        # Own lock of the reparametrized transformation, then the ownership of the
        # transformation and of the parametrization (locked permanently at the creation)
        super()._check_unlocked(action, override_lock=override_lock)
        if override_lock:
            return
        for kind, obj, consequence in (
            ("transformation", self._transformation,
             "its parameters may have been modified and may no longer be equal to P(q)"),
            ("parametrization", self._parametrization,
             "it may have been modified, and the parameters p = P(q) of the transformation and "
             "the cached Jacobian dP/dq may then be wrong"),
        ):
            if not obj.is_locked:
                raise RuntimeError(
                    f"Cannot {action}: the {kind} ({obj.__class__.__name__}) of this "
                    f"ReparametrizedTransformation was unlocked from outside (with the internal "
                    f"method _set_locked(False)): {consequence}. The ReparametrizedTransformation "
                    f"cannot guarantee its consistency anymore. Create a new "
                    f"ReparametrizedTransformation with copies of the transformation and of the "
                    f"parametrization: ReparametrizedTransformation(transformation.copy(), "
                    f"parametrization.copy(), parameters=q)."
                )

    def set_parameters(
        self, value: Optional[ArrayLike], *, override_lock: bool = False
    ) -> None:
        r"""
        Set the parameters :math:`\vec{q}` of the reparametrized transformation, with an option
        to override the lock.

        The parameters :math:`\vec{p} = P(\vec{q})` of the underlying transformation are
        updated accordingly.

        Parameters
        ----------
        value : Optional[ArrayLike]
            A 1D array of shape :math:`(n_q,)` containing the parameters to set,
            or None to unset the parameters.

        override_lock : bool, optional (default: ``False``)
            Whether to override the lock and allow setting parameters even if they are locked.

        Raises
        ------
        ValueError
            If the input value is not a 1D array or does not have the correct shape.

        RuntimeError
            If the parameters are locked and ``override_lock`` is False.

        """
        super().set_parameters(value, override_lock=override_lock)  # Checks the locks, stores q
        self._jacobian_encode = None  # Invalidate the cache of dP/dq

        q = self._parameters
        p = None if q is None else self._parametrization.encode(q).p
        self._transformation.set_parameters(p, override_lock=True)  # Owned: locked permanently

    def _get_jacobian_encode(self) -> numpy.ndarray:
        # Jacobian dP/dq at the current parameters q, computed once per value of q
        if self._jacobian_encode is None:
            self._jacobian_encode = self._parametrization.encode(
                self._parameters, dp=True
            ).jacobian_dp  # (n_p, n_q)
        return self._jacobian_encode

    # ------
    # Serialization and copy
    # ------
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "transformation": self._transformation.to_dict(),
            "parametrization": self._parametrization.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> ReparametrizedTransformation:
        if "transformation" not in data:
            raise ValueError("Input dictionary must contain a 'transformation' key.")
        if "parametrization" not in data:
            raise ValueError("Input dictionary must contain a 'parametrization' key.")
        transformation = Transformation.from_dict(data["transformation"])
        parametrization = Parametrization.from_dict(data["parametrization"])
        return cls(
            transformation=transformation,
            parametrization=parametrization,
            parameters=data.get("parameters", None),
        )

    def _copy(self) -> ReparametrizedTransformation:
        # Unlocked copies of the transformation and of the parametrization, owned (and locked)
        # by the new reparametrized transformation
        return self.__class__(
            transformation=self._transformation.copy(),
            parametrization=self._parametrization.copy(),
            parameters=self.parameters,
        )

    # ------
    # Apply and invert
    # ------
    def _apply(
        self,
        x_i: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the transformation :math:`T` to the input points :math:`\vec{x_i}` with the
        parameters :math:`\vec{p} = P(\vec{q})`.

        .. math::

            \vec{x_o} = T(\vec{x_i}; P(\vec{q}))

        The Jacobian with respect to the parameters is computed using the chain rule:

        .. math::

            \mathbf{J}_q = \frac{\partial T}{\partial \vec{p}} \cdot \frac{\partial P}{\partial \vec{q}}

        Parameters
        ----------
        x_i : numpy.ndarray
            The input points to transform, with shape :math:`(n_\text{points}, n_i)`.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters :math:`\vec{q}`.

        **kwargs
            Additional arguments passed to the ``apply`` method of the transformation.


        Returns
        -------
        numpy.ndarray
            The transformed output points, with shape :math:`(n_\text{points}, n_o)`.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the input points if ``dx=True``, with shape
            :math:`(n_\text{points}, n_o, n_i)`, or None if ``dx=False``.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the parameters :math:`\vec{q}` if ``dp=True``,
            with shape :math:`(n_\text{points}, n_o, n_q)`, or None if ``dp=False``.

        """
        result = self._transformation.apply(x_i, dx=dx, dp=dp, **kwargs)

        jacobian_dp = None
        if dp:
            jacobian_dp = (
                result.jacobian_dp @ self._get_jacobian_encode()
            )  # (n_points, n_o, n_p) @ (n_p, n_q) -> (n_points, n_o, n_q)

        return result.x, result.jacobian_dx, jacobian_dp

    def _invert(
        self,
        x_o: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse map :math:`T^{-1}` to the output points :math:`\vec{x_o}` with the
        parameters :math:`\vec{p} = P(\vec{q})`.

        .. math::

            \vec{x_i} = T^{-1}(\vec{x_o}; P(\vec{q}))

        The Jacobian with respect to the parameters is computed using the chain rule:

        .. math::

            \mathbf{J}_q = \frac{\partial T^{-1}}{\partial \vec{p}} \cdot \frac{\partial P}{\partial \vec{q}}

        Parameters
        ----------
        x_o : numpy.ndarray
            The output points to invert, with shape :math:`(n_\text{points}, n_o)`.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the output points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters :math:`\vec{q}`.

        **kwargs
            Additional arguments passed to the ``invert`` method of the transformation.


        Returns
        -------
        numpy.ndarray
            The inverted input points, with shape :math:`(n_\text{points}, n_i)`.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the output points if ``dx=True``, with shape
            :math:`(n_\text{points}, n_i, n_o)`, or None if ``dx=False``.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the parameters :math:`\vec{q}` if ``dp=True``,
            with shape :math:`(n_\text{points}, n_i, n_q)`, or None if ``dp=False``.

        """
        result = self._transformation.invert(x_o, dx=dx, dp=dp, **kwargs)

        jacobian_dp = None
        if dp:
            jacobian_dp = (
                result.jacobian_dp @ self._get_jacobian_encode()
            )  # (n_points, n_i, n_p) @ (n_p, n_q) -> (n_points, n_i, n_q)

        return result.x, result.jacobian_dx, jacobian_dp