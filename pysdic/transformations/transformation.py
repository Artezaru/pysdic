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
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
from numpy.typing import ArrayLike

import numpy
import datetime
import inspect

from ..__version__ import __version__
from ..core.serialization import JsonSerializerMixin, Hdf5SerializerMixin
from ..core.lockable import LockableMixin
from .._optionals import RequiresOptionalMixin
from .._validation import (
    _validate_bool,
    _validate_real,
    _validate_str,
    _validate_tolerances,
    _validate_kwargs,
    _validate_kwargs_sequence,
    _as_array,
    _as_batch_array,
)


@dataclass(slots=True, frozen=True)
class TransformationResult:
    r"""
    A class to represent the result of a transformation.

    This class is used to store the results of a transformation, including the
    transformed points and the Jacobian matrices.

    .. seealso::

        - :class:`pysdic.Transformation` for the transformations that produce
          a :class:`pysdic.TransformationResult` object.

    Lets :math:`T` be a transformation that maps input points :math:`\vec{x_i}`
    from an input space :math:`\mathbb{R}^{n_i}` to output points :math:`\vec{x_o}`
    in an output space :math:`\mathbb{R}^{n_o}` with parameters :math:`\vec{p}`.
    Then the result of applying the transformation can be represented as a
    :class:`pysdic.TransformationResult` object with the following attributes:

    Parameters
    ----------
    x : numpy.ndarray
        The transformed points :math:`\vec{x_o}` after applying the transformation.
        Shape :math:`(\dots, n_o)`.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the input points :math:`\vec{x_i}`,
        or None if not computed.
        Shape :math:`(\dots, n_o, n_i)`.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the parameters of the transformation :math:`\vec{p}`,
        or None if not computed.
        Shape :math:`(\dots, n_o, n_p)`.

    .. note::

        For the result of the ``invert`` method, the roles of the input and output
        spaces are swapped: ``x`` has shape :math:`(\dots, n_i)`, ``jacobian_dx`` has shape
        :math:`(\dots, n_i, n_o)` and ``jacobian_dp`` has shape :math:`(\dots, n_i, n_p)`.

    """

    x: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dp: Optional[numpy.ndarray] = None


class Transformation(
    LockableMixin, RequiresOptionalMixin, JsonSerializerMixin, Hdf5SerializerMixin
):
    r"""
    Abstract base class representing a spatial parametric transformation :math:`T` that
    maps points :math:`\vec{x_i}` from an input space :math:`\mathbb{R}^{n_i}` to an
    output space :math:`\mathbb{R}^{n_o}`. The transformation is defined by a set of
    parameters :math:`\vec{p} \in \mathbb{R}^{n_p}` that control the mapping.

    .. math::

        \vec{x_o} = T(\vec{x_i}; \vec{p})

    .. note::

        - A **parameter** is a variable for which the transformation can be
          differentiated with respect to, and is typically optimized during fitting or
          registration processes.

        - The ``parameters`` are copied when they are set and are exposed as read-only
          views to prevent accidental modifications.

        - The parameters can be locked (see :class:`pysdic.core.lockable.LockableMixin`)
          when the transformation is owned by another object of the package that assumes
          they do not change (for example a :class:`pysdic.ReparametrizedTransformation`).
          A locked transformation refuses ``parameters = ...`` and ``set_parameters``
          (unless ``override_lock=True``): ``is_locked`` and ``locked_message`` tell whether
          it is locked and why. A copy is always unlocked.

    All transformations can depend on **constants**, which are fixed values
    that influence the transformation but are not subject to optimization or
    differentiation.

    A transformation can be applied to input points :math:`\vec{x_i} \in \mathbb{R}^{n_i}`
    using the ``apply`` method, which returns a :class:`pysdic.TransformationResult`
    object containing the transformed points :math:`\vec{x_o} \in \mathbb{R}^{n_o}` and
    the Jacobian matrices :math:`\mathbf{J}_x = \frac{\partial T}{\partial \vec{x_i}}` (if requested)
    and :math:`\mathbf{J}_p = \frac{\partial T}{\partial \vec{p}}` (if requested).

    A transformation can also define an inverse map :math:`T^{-1}` using the ``invert`` method,
    which returns a :class:`pysdic.TransformationResult` object
    containing the inverted points :math:`\vec{x_i} \in \mathbb{R}^{n_i}` and the Jacobian matrices
    :math:`\mathbf{J}_x = \frac{\partial T^{-1}}{\partial \vec{x_o}}` (if requested)
    and :math:`\mathbf{J}_p = \frac{\partial T^{-1}}{\partial \vec{p}}` (if requested).

    .. seealso::

        - Class :class:`pysdic.TransformationResult` for the result returned by the
          ``apply`` and ``invert`` methods.

    .. note::

        When :math:`n_i \neq n_o`, the map :math:`T` cannot be bijective and :math:`T^{-1}`
        denotes a left inverse or a right inverse of :math:`T` (see the flags
        ``is_left_invertible`` and ``is_right_invertible`` below).

    .. note::

        The transformation acts independently on each point: the output point
        :math:`\vec{x_o}` only depends on the corresponding input point :math:`\vec{x_i}`
        and on the parameters :math:`\vec{p}`.

    .. seealso::

        To change the parametric space of the transformation (ie :math:`\vec{p} = P(\vec{p_i})`),
        use the following classes:

        - Class :class:`pysdic.Parametrization` for the parametrizations that can encode the parameters of a transformation.
        - Class :class:`pysdic.ReparametrizedTransformation` to allow a transformation to be associated with a parametrization to encode its parameters.

    .. seealso::

        To compose transformations (ie :math:`T = T_2 \circ T_1`),
        use the following classes:

        - Class :class:`pysdic.ComposedTransformation` to allow multiple transformations to be composed together.
        - The syntax ``T2 @ T1`` which is equivalent to ``ComposedTransformation([T1, T2])``.


    Parameters
    ----------
    parameters : Optional[ArrayLike] (default: ``None``)
        A 1D array of shape :math:`(n_p,)` containing the parameters :math:`\vec{p}` of the transformation.
        If None, the parameters are not set (and the transformation is not set up) unless
        :math:`n_p = 0`.


    Capability flags
    ----------------

    Each transformation declares its capabilities with the following boolean
    class/instance attributes. All flags default to ``False``.

    - ``_has_apply``: the method ``_apply`` is implemented (i.e. :math:`T` can be evaluated).
    - ``_has_apply_dx``: the method ``_apply`` can compute :math:`\frac{\partial T}{\partial \vec{x_i}}`.
    - ``_has_apply_dp``: the method ``_apply`` can compute :math:`\frac{\partial T}{\partial \vec{p}}`.
    - ``_has_invert``: the method ``_invert`` is implemented (i.e. :math:`T^{-1}` can be evaluated).
    - ``_has_invert_dx``: the method ``_invert`` can compute :math:`\frac{\partial T^{-1}}{\partial \vec{x_o}}`.
    - ``_has_invert_dp``: the method ``_invert`` can compute :math:`\frac{\partial T^{-1}}{\partial \vec{p}}`.
    - ``_is_left_invertible``: ``invert`` is a left inverse of ``apply``, i.e.
      :math:`T^{-1}(T(\vec{x_i})) = \vec{x_i}` for all :math:`\vec{x_i} \in \mathbb{R}^{n_i}`.
    - ``_is_right_invertible``: ``invert`` is a right inverse of ``apply``, i.e.
      :math:`T(T^{-1}(\vec{x_o})) = \vec{x_o}` for all :math:`\vec{x_o} \in \mathbb{R}^{n_o}`.

    The flags must be consistent:

    - ``_has_apply_dx`` and ``_has_apply_dp`` require ``_has_apply``.
    - ``_has_invert_dx`` and ``_has_invert_dp`` require ``_has_invert``.
    - ``_is_left_invertible`` and ``_is_right_invertible`` require both ``_has_apply`` and ``_has_invert``.

    The consistency is checked at initialization by ``Transformation.__init__``.
    The flags ``_is_left_invertible`` and ``_is_right_invertible`` are declarations
    that can be verified numerically with the methods ``check_left_invertible`` and
    ``check_right_invertible``.


    Implementation details
    ----------------------

    - All subclasses must have an ``__init__`` method taking ``parameters`` as an argument
      and calling ``super().__init__(parameters=parameters)`` at the end, once the constants
      are set (the number of parameters can depend on the constants). Optional class-methods
      can be implemented for specific constructions.

    - Subclasses with ``_has_apply=True`` must implement the method ``_apply``, and
      subclasses with ``_has_invert=True`` must implement the method ``_invert``.
      These methods are called by the public methods ``apply`` and ``invert``
      respectively, which handle the validation of the inputs and outputs of the
      transformation with optional Jacobian computation. If a flag is ``False``,
      the corresponding method does not need to be implemented.

    - All subclasses must implement the properties ``input_dim`` and ``output_dim``
      to return the input and output dimensions of the transformation.

    - All subclasses must implement the property ``n_parameters`` to return the number of parameters
      :math:`n_p` of the transformation (must not depend on the **parameters** array).

    - All subclasses must implement the method ``_to_dict``, ``_from_dict``, and
      ``_copy`` called by the public methods, ``from_dict``, ``to_dict``
      and ``copy`` will handle copying and the serialization and
      deserialization of the **constants** of the transformation.

    - All subclasses must set the capability flags described above, either as class attributes
      (if they do not depend on the constants) or as instance attributes before calling
      ``super().__init__()`` (if they depend on the constants).

    - Subclasses that cannot work without an optional package declare it in the class attribute
      ``_required_optionals`` (e.g. ``_required_optionals = ("pyzernike",)``): an ``ImportError``
      is raised when the class is instantiated if the package is missing, and the class method
      ``is_available()`` returns False (see :class:`pysdic._optionals.RequiresOptionalMixin`).

    Here is a minimal example of how to implement a simple subclass of :class:`pysdic.Transformation`
    representing a 2D scaled translation :math:`\vec{x_o} = \mathbf{S} \vec{x_i} + \vec{p}` with
    the translation :math:`\vec{p}` as **parameters** and the diagonal scaling factors
    :math:`\mathbf{S} = \text{diag}(s_x, s_y)` as **constants**:

    .. code-block:: python

        class ScaledTranslation2D(Transformation):

            _has_apply = True
            _has_apply_dx = True
            _has_apply_dp = True
            _has_invert = True
            _has_invert_dx = True
            _has_invert_dp = True
            _is_left_invertible = True
            _is_right_invertible = True

            def __init__(
                self,
                parameters: Optional[ArrayLike] = None,
                scales: ArrayLike = (1.0, 1.0),
            ):
                scales = numpy.array(scales, dtype=float, copy=True)
                if scales.shape != (2,) or numpy.any(scales == 0.0):
                    raise ValueError("scales must be 2 non-zero values to be invertible.")
                scales.flags.writeable = False
                self._scales = scales
                super().__init__(parameters=parameters)  # Check the flags and set the parameters

            @property
            def scales(self) -> numpy.ndarray:
                return self._scales  # Read-only constant

            @property
            def n_parameters(self) -> int:
                return 2  # translation in x and y

            @property
            def input_dim(self) -> int:
                return 2  # 2D input points

            @property
            def output_dim(self) -> int:
                return 2  # 2D output points

            def _to_dict(self) -> Dict[str, Any]:
                return {"scales": self.scales.tolist()}  # 'parameters' is added automatically

            @classmethod
            def _from_dict(cls, data: Dict[str, Any]) -> ScaledTranslation2D:
                return cls(parameters=data["parameters"], scales=data["scales"])

            def _copy(self) -> ScaledTranslation2D:
                return self.__class__(parameters=self.parameters, scales=self.scales)

            def _apply(self, x_i: numpy.ndarray, *, dx: bool = False, dp: bool = False, **kwargs):
                n_points = x_i.shape[0]
                x_o = x_i * self.scales + self.parameters  # (n_points, 2)
                jacobian_dx = numpy.tile(numpy.diag(self.scales), (n_points, 1, 1)) if dx else None  # (n_points, 2, 2)
                jacobian_dp = numpy.tile(numpy.eye(2), (n_points, 1, 1)) if dp else None  # (n_points, 2, 2)
                return x_o, jacobian_dx, jacobian_dp

            def _invert(self, x_o: numpy.ndarray, *, dx: bool = False, dp: bool = False, **kwargs):
                n_points = x_o.shape[0]
                x_i = (x_o - self.parameters) / self.scales  # (n_points, 2)
                jacobian_dx = numpy.tile(numpy.diag(1.0 / self.scales), (n_points, 1, 1)) if dx else None  # (n_points, 2, 2)
                jacobian_dp = numpy.tile(-numpy.diag(1.0 / self.scales), (n_points, 1, 1)) if dp else None  # (n_points, 2, 2)
                return x_i, jacobian_dx, jacobian_dp

    The implementation can then be verified numerically:

    .. code-block:: python

        T = ScaledTranslation2D(parameters=[1.0, 2.0], scales=[2.0, 3.0])
        T.apply([[1.0, 1.0]]).x                              # array([[3., 5.]])
        T.check_apply_dx([[1.0, 1.0]], raise_error=True)     # True
        T.check_apply_dp([[1.0, 1.0]], raise_error=True)     # True
        T.check_left_invertible([[1.0, 1.0]], check_dx=True, check_dp=True)  # True

    """

    __subclass_registry: ClassVar[Dict[str, type]] = {}

    # Capability flags (conservative defaults, overridden by subclasses)
    _has_apply: bool = False
    _has_apply_dx: bool = False
    _has_apply_dp: bool = False
    _has_invert: bool = False
    _has_invert_dx: bool = False
    _has_invert_dp: bool = False
    _is_left_invertible: bool = False
    _is_right_invertible: bool = False

    # Explanation of the errors raised by a transformation locked without a message
    _default_locked_message: str = (
        "It is used by another object of the package that assumes that its parameters do "
        "not change. Work on a copy (transformation.copy()), or use "
        "set_parameters(..., override_lock=True) to force setting the parameters if you are "
        "sure this will not cause inconsistencies."
    )

    def __init__(
        self,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        self._parameters: Optional[numpy.ndarray] = None
        self._assert_flags_set()
        if parameters is not None or self.n_parameters == 0:
            self.set_parameters(parameters, override_lock=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "Transformation":
            raise TypeError(
                "subclass of Transformation cannot be named 'Transformation' to avoid conflicts with the base class."
            )
        if cls.__name__ in cls.__subclass_registry:
            raise ValueError(
                f"A subclass of Transformation with name '{cls.__name__}' is already registered."
            )
        cls.__subclass_registry[cls.__name__] = cls

    @classmethod
    def __get_subclass_by_name(cls, name: str) -> type:
        if not name in cls.__subclass_registry:
            raise ValueError(
                f"No subclass of Transformation with name '{name}' is registered."
            )
        return cls.__subclass_registry[name]

    def _assert_flags_set(self) -> None:

        required_flags = {
            "_has_apply": "indicate whether the transformation implements apply().",
            "_has_apply_dx": "indicate whether the transformation supports Jacobian dx computation in apply().",
            "_has_apply_dp": "indicate whether the transformation supports Jacobian dp computation in apply().",
            "_has_invert": "indicate whether the transformation implements invert().",
            "_has_invert_dx": "indicate whether the transformation supports Jacobian dx computation in invert().",
            "_has_invert_dp": "indicate whether the transformation supports Jacobian dp computation in invert().",
            "_is_left_invertible": "indicate whether invert() is a left inverse of apply().",
            "_is_right_invertible": "indicate whether invert() is a right inverse of apply().",
        }

        for attr, description in required_flags.items():

            value = getattr(self, attr)

            if not isinstance(value, bool):
                raise TypeError(
                    f"Attribute '{attr}' of subclass "
                    f"{self.__class__.__name__} must be a bool to {description} "
                    f"Got {type(value)}."
                )

        # Consistency of the flags
        if (self._has_apply_dx or self._has_apply_dp) and not self._has_apply:
            raise ValueError(
                f"Subclass {self.__class__.__name__} sets '_has_apply_dx=True' or "
                f"'_has_apply_dp=True' but '_has_apply=False'."
            )
        if (self._has_invert_dx or self._has_invert_dp) and not self._has_invert:
            raise ValueError(
                f"Subclass {self.__class__.__name__} sets '_has_invert_dx=True' or "
                f"'_has_invert_dp=True' but '_has_invert=False'."
            )
        if (self._is_left_invertible or self._is_right_invertible) and not (
            self._has_apply and self._has_invert
        ):
            raise ValueError(
                f"Subclass {self.__class__.__name__} is declared left or right invertible "
                f"but does not implement both apply ('_has_apply') and invert ('_has_invert')."
            )

    # ------
    # To be implemented by subclasses
    # ------
    @property
    @abstractmethod
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters :math:`n_p` of the transformation.

        Returns
        -------
        int
            The number of parameters of the transformation.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the property 'n_parameters' "
            f"to return the number of parameters of the transformation."
        )

    @property
    @abstractmethod
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the transformation.

        Returns
        -------
        int
            The input dimension of the transformation.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the property 'input_dim' "
            f"to return the input dimension of the transformation."
        )

    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the transformation.

        Returns
        -------
        int
            The output dimension of the transformation.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the property 'output_dim' "
            f"to return the output dimension of the transformation."
        )

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        r"""
        Serialize the constants of the transformation to a dictionary.

        .. note::

            The **parameters** must not be included in the returned dictionary: they are
            serialized separately by the ``to_dict`` method.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the constants of the transformation for
            serialization in the 'to_dict' method.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the method '_to_dict' "
            f"to return a dictionary of constants for serialization in the 'to_dict' method."
        )

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Transformation:
        r"""
        Create a transformation instance from a dictionary.

        .. note::

            The input dictionary contains a key ``'parameters'`` with the parameters
            array to initialize the transformation with (or None), and all the
            additional constants returned by the ``_to_dict`` method of the transformation.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the parameters and constants for the transformation.

        Returns
        -------
        :class:`pysdic.Transformation`
            A new instance of the transformation initialized with the data from the dictionary.

        """
        raise NotImplementedError(
            f"Subclass {cls.__name__} must implement the class method '_from_dict' "
            f"to create a transformation instance from a dictionary for deserialization in the 'from_dict' method."
        )

    @abstractmethod
    def _copy(self) -> Transformation:
        r"""
        Create a copy of the transformation.

        Returns
        -------
        :class:`pysdic.Transformation`
            A new instance of the transformation with the same parameters and constants.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the method '_copy' "
            f"to create a copy of the transformation for the 'copy' method."
        )

    def _apply(
        self,
        x_i: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the transformation to input points :math:`\vec{x_i}` with optional Jacobian computation.

        .. note::

            This method must be implemented by subclasses setting ``_has_apply=True``.
            The input array may share memory with the user's array and must not be
            modified in place.

        Parameters
        ----------
        x_i : numpy.ndarray
            A 2D array of shape :math:`(n_\text{points}, n_i)` containing the input points to transform.

        dx : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to input points.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to parameters.

        **kwargs
            Additional arguments passed to the underlying ``_apply`` method.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:
            - The transformed output points as a 2D array of shape :math:`(n_\text{points}, n_o)`.
            - The Jacobian with respect to input points as a 3D array of shape :math:`(n_\text{points}, n_o, n_i)` if computed, otherwise None.
            - The Jacobian with respect to parameters as a 3D array of shape :math:`(n_\text{points}, n_o, n_p)` if computed, otherwise None.

        """
        if self._has_apply:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} sets '_has_apply=True' but does not "
                f"implement the method '_apply' to apply the transformation to input "
                f"points with optional Jacobian computation."
            )
        raise NotImplementedError(
            f"Transformation of type '{self.__class__.__name__}' does not implement "
            f"apply ('_has_apply=False')."
        )

    def _invert(
        self,
        x_o: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse map :math:`T^{-1}` to output points :math:`\vec{x_o}` with optional Jacobian computation.

        .. note::

            This method must be implemented by subclasses setting ``_has_invert=True``.
            The input array may share memory with the user's array and must not be
            modified in place.

        Parameters
        ----------
        x_o : numpy.ndarray
            A 2D array of shape :math:`(n_\text{points}, n_o)` containing the output points to invert.

        dx : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to output points.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to parameters.

        **kwargs
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:
            - The inverted input points as a 2D array of shape :math:`(n_\text{points}, n_i)`.
            - The Jacobian with respect to output points as a 3D array of shape :math:`(n_\text{points}, n_i, n_o)` if computed, otherwise None.
            - The Jacobian with respect to parameters as a 3D array of shape :math:`(n_\text{points}, n_i, n_p)` if computed, otherwise None.

        """
        if self._has_invert:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} sets '_has_invert=True' but does not "
                f"implement the method '_invert' to invert the transformation for output "
                f"points with optional Jacobian computation."
            )
        raise NotImplementedError(
            f"Transformation of type '{self.__class__.__name__}' does not implement "
            f"invert ('_has_invert=False')."
        )

    # ------
    # Properties and setters for parameters
    # ------
    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] or [Set] the parameters of the transformation.

        .. note::

            If the number of parameters :math:`n_p` is greater than zero,
            the **parameters** array is a 1D array of shape :math:`(n_p,)`
            containing the parameter values or ``None`` if the parameters are not set.

            If :math:`n_p` is zero, the **parameters** attribute is always
            an empty array of shape :math:`(0,)` for optimal compatibility
            with optimization algorithms instead of being set to ``None``.

        .. note::

            The parameters are copied when they are set, and the returned array is a
            read-only view to prevent accidental modifications.

        .. seealso::

            - Method ``set_parameters`` to set the parameters of a locked transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            A 1D array of shape :math:`(n_p,)` containing the parameters
            of the transformation, or None if not set.

        """
        if self._parameters is None:
            return None
        view = self._parameters.view()
        view.flags.writeable = False
        return view

    @parameters.setter
    def parameters(self, value: Optional[ArrayLike]) -> None:
        self.set_parameters(value, override_lock=False)

    def set_parameters(
        self, value: Optional[ArrayLike], *, override_lock: bool = False
    ) -> None:
        r"""
        Set the parameters of the transformation, with an option to override the lock.

        The parameters of a transformation are locked when it is owned by another object of
        the package, to avoid inconsistencies (for example by a
        :class:`pysdic.ReparametrizedTransformation`, see ``locked_message``).

        Parameters
        ----------
        value : Optional[ArrayLike]
            A 1D array of shape :math:`(n_p,)` containing the parameters to set
            for the transformation, or None to unset the parameters. The parameters
            must be finite (no NaN or inf).

        override_lock : bool, optional (default: ``False``)
            Whether to override the lock and allow setting parameters even if they are locked.

        Raises
        ------
        ValueError
            If the input value is not a 1D array, does not have the correct shape or is not finite.

        RuntimeError
            If the parameters are locked and ``override_lock`` is False.

        """
        override_lock = _validate_bool(override_lock, "override_lock")
        self._check_unlocked("set the parameters", override_lock=override_lock)

        if value is None:
            # Transformations without parameters always have an empty parameters array
            self._parameters = (
                numpy.empty((0,), dtype=float) if self.n_parameters == 0 else None
            )
            return

        self._parameters = self._validate_parameters(value)

    # -----
    # Flags
    # -----
    @property
    def is_setup(self) -> bool:
        r"""
        [Get] whether the transformation is fully set up and ready to be applied or inverted.

        A transformation is considered fully set up if its parameters are set
        (i.e., not None).

        Returns
        -------
        bool
            True if the transformation is fully set up, False otherwise.

        """
        return self.parameters is not None

    @property
    def has_apply(self) -> bool:
        r"""
        [Get] whether the transformation implements the apply method.

        Returns
        -------
        bool
            True if the transformation implements the apply method, False otherwise.

        """
        return self._has_apply

    @property
    def has_apply_dx(self) -> bool:
        r"""
        [Get] whether the apply method can compute the Jacobian with respect to input points.

        Returns
        -------
        bool
            True if the apply method can compute the Jacobian with respect to input points, False otherwise.

        """
        return self._has_apply_dx

    @property
    def has_apply_dp(self) -> bool:
        r"""
        [Get] whether the apply method can compute the Jacobian with respect to parameters.

        Returns
        -------
        bool
            True if the apply method can compute the Jacobian with respect to parameters, False otherwise.

        """
        return self._has_apply_dp

    @property
    def has_invert(self) -> bool:
        r"""
        [Get] whether the transformation implements the invert method.

        Returns
        -------
        bool
            True if the transformation implements the invert method, False otherwise.

        """
        return self._has_invert

    @property
    def has_invert_dx(self) -> bool:
        r"""
        [Get] whether the invert method can compute the Jacobian with respect to output points.

        Returns
        -------
        bool
            True if the invert method can compute the Jacobian with respect to output points, False otherwise.

        """
        return self._has_invert_dx

    @property
    def has_invert_dp(self) -> bool:
        r"""
        [Get] whether the invert method can compute the Jacobian with respect to parameters.

        Returns
        -------
        bool
            True if the invert method can compute the Jacobian with respect to parameters, False otherwise.

        """
        return self._has_invert_dp

    @property
    def is_left_invertible(self) -> bool:
        r"""
        [Get] whether the invert method is a left inverse of the apply method.

        .. math::

            T^{-1}(T(\vec{x_i})) = \vec{x_i} \quad \forall \vec{x_i} \in \mathbb{R}^{n_i}

        Returns
        -------
        bool
            True if the invert method is a left inverse of the apply method, False otherwise.

        """
        return self._is_left_invertible

    @property
    def is_right_invertible(self) -> bool:
        r"""
        [Get] whether the invert method is a right inverse of the apply method.

        .. math::

            T(T^{-1}(\vec{x_o})) = \vec{x_o} \quad \forall \vec{x_o} \in \mathbb{R}^{n_o}

        Returns
        -------
        bool
            True if the invert method is a right inverse of the apply method, False otherwise.

        """
        return self._is_right_invertible

    @property
    def is_invertible(self) -> bool:
        r"""
        [Get] whether the transformation is invertible (i.e. both left and right invertible).

        In this case :math:`T` is a bijection and :math:`n_i = n_o`.

        Returns
        -------
        bool
            True if the transformation is both left and right invertible, False otherwise.

        """
        return self._is_left_invertible and self._is_right_invertible

    # ------
    # Internal validation helpers
    # ------
    def _validate_parameters(self, value: ArrayLike) -> numpy.ndarray:
        # Copy the parameters (n_p,) to protect the user's array
        return _as_array(
            value, "parameters", numpy.float64, shape=(self.n_parameters,), finite=True, copy=True
        )

    def _set_parameters_at(
        self,
        index: Union[int, slice],
        value: Any,
        name: str,
        *,
        hint: Optional[str] = None,
    ) -> None:
        # Set some components of the parameters, e.g. one coefficient (int index) or a block (slice).
        # The value is validated (a finite real number for an int index, an array with the size of
        # the slice otherwise), and the parameters are set with the 'parameters' setter: the lock
        # and the checks of the subclass (set_parameters) are respected.
        parameters = self.parameters
        if parameters is None:
            raise ValueError(
                f"Cannot set {name} while the parameters are not set. Set all the parameters first"
                + (f" ({hint})." if hint else ".")
            )
        if isinstance(index, slice):
            size = len(range(self.n_parameters)[index])
            value = _as_array(value, name, numpy.float64, shape=(size,), finite=True)
        else:
            value = _validate_real(value, name, finite=True)
        parameters = parameters.copy()
        parameters[index] = value
        self.parameters = parameters

    def _validate_points(
        self, x: ArrayLike, dim: int, name: str
    ) -> Tuple[numpy.ndarray, Tuple[int, ...]]:
        # Reshape the points (..., dim) into a 2D array (n_points, dim) without copy if possible.
        # A single point must be given with shape (1, dim).
        x = _as_batch_array(x, name, dim, numpy.float64, finite=False, min_ndim=2)
        return x.reshape(-1, dim), x.shape[:-1]

    def _validate_output(
        self,
        output: Any,
        *,
        dx: bool,
        dp: bool,
        n_points: int,
        dim_out: int,
        dim_in: int,
        method: str,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        name = self.__class__.__name__
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError(
                f"Method '{method}' of '{name}' must return a tuple (x, jacobian_dx, jacobian_dp), "
                f"got {type(output)}."
            )
        x, jacobian_dx, jacobian_dp = output

        x = numpy.asarray(x, dtype=float)
        if x.shape != (n_points, dim_out):
            raise RuntimeError(
                f"Method '{method}' of '{name}' returned points with shape {x.shape}, "
                f"expected {(n_points, dim_out)}."
            )

        jacobians = []
        for flag, jacobian, n_cols, label in (
            (dx, jacobian_dx, dim_in, "jacobian_dx"),
            (dp, jacobian_dp, self.n_parameters, "jacobian_dp"),
        ):
            if not flag:
                jacobians.append(None)  # Discard any Jacobian computed while not requested
                continue
            if jacobian is None:
                raise RuntimeError(
                    f"Method '{method}' of '{name}' returned None for '{label}' while it was requested."
                )
            jacobian = numpy.asarray(jacobian, dtype=float)
            expected_shape = (n_points, dim_out, n_cols)
            if jacobian.shape != expected_shape:
                raise RuntimeError(
                    f"Method '{method}' of '{name}' returned '{label}' with shape "
                    f"{jacobian.shape}, expected {expected_shape}."
                )
            jacobians.append(jacobian)

        return x, jacobians[0], jacobians[1]

    # ------
    # Methods
    # ------
    def copy(self) -> Transformation:
        r"""
        Create a copy of the transformation.

        The copy is always **unlocked** (see ``is_locked``), even if the transformation is
        locked: it can be modified independently of the original.

        .. note::

            The transformations owned by a copied :class:`pysdic.ReparametrizedTransformation`
            are locked by the new reparametrized transformation, as for any reparametrized
            transformation.

        Returns
        -------
        Transformation
            A new instance of the transformation with the same parameters and constants.

        """
        new = self._copy()
        new._set_locked(False)
        return new

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transformation:
        r"""
        Create a transformation instance from a dictionary.

        The class instantiated is given by the ``'type'`` key of the dictionary:

        - If called on :class:`pysdic.Transformation`, the ``'type'`` key is required and
          any registered subclass can be instantiated.
        - If called on a subclass ``cls``, the ``'type'`` key is optional (defaults to ``cls``)
          and must designate ``cls`` or one of its subclasses.

        .. code-block:: python

            T = Transformation.from_dict(data)       # Any registered transformation
            T = MyTransformation.from_dict(data)     # MyTransformation or a subclass of it

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the parameters and constants for the transformation.

        Returns
        -------
        Transformation
            A new instance of the transformation initialized with the data from the dictionary.

        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data)}")
        if not "context" in data:
            raise ValueError("Input dictionary must contain a 'context' key.")
        if data["context"] != "Transformation":
            raise ValueError(
                f"Input dictionary has 'context' key '{data['context']}' that does not match 'Transformation'."
            )

        t_type = _validate_str(data.get("type", None), "type", allow_none=True)
        if cls is Transformation:
            # Dispatch to the correct subclass based on the 'type' key in the input dictionary
            if t_type is None:
                raise ValueError(
                    "Input dictionary must contain a 'type' key indicating the transformation type."
                )
            t_class = cls.__get_subclass_by_name(t_type)
        elif t_type is None:
            t_class = cls
        else:
            t_class = cls.__get_subclass_by_name(t_type)
            if not issubclass(t_class, cls):
                raise ValueError(
                    f"Input dictionary has 'type' key '{t_type}' that is not the class "
                    f"'{cls.__name__}' or one of its subclasses."
                )

        if inspect.isabstract(t_class):
            raise TypeError(
                f"Cannot deserialize the abstract transformation class '{t_class.__name__}'. "
                f"The 'type' key must designate a concrete subclass."
            )

        if not "parameters" in data:
            raise ValueError("Input dictionary must contain a 'parameters' key.")
        if not "kwargs" in data:
            raise ValueError("Input dictionary must contain a 'kwargs' key.")
        kwargs = data["kwargs"]  # dict of kwargs for the transformation constructor
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"'kwargs' key in input dictionary must be a dictionary, got {type(kwargs)}"
            )

        # Copy to protect the input dictionary and add the parameters for the constructor
        kwargs = dict(kwargs)
        kwargs["parameters"] = data["parameters"]

        return t_class._from_dict(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Serialize the transformation to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parameters and constants of the transformation,
            including a 'type' key indicating the transformation type and
            the metadata 'version' and 'datetime' of the serialization.

        """
        parameters = self.parameters
        return {
            "type": self.__class__.__name__,
            "context": "Transformation",
            "version": __version__,
            "datetime": datetime.datetime.now().isoformat(),
            "parameters": parameters.tolist() if parameters is not None else None,
            "kwargs": self._to_dict(),
        }

    def apply(
        self,
        x_i: ArrayLike,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> TransformationResult:
        r"""
        Apply the transformation to the given points :math:`\vec{x_i} \in \mathbb{R}^{n_i}`
        to return the transformed points :math:`\vec{x_o} \in \mathbb{R}^{n_o}`.

        .. math::

            \vec{x_o} = T(\vec{x_i}; \vec{p})

        .. important::

            - The apply method can only be used if the transformation
              implements it (i.e., ``has_apply`` must be True).
            - The transformation must be fully set up (i.e., ``is_setup`` must be True).
            - The jacobian ``dx`` and ``dp`` can only be computed if the transformation
              supports it (i.e., ``has_apply_dx`` and ``has_apply_dp`` must be True).

        .. note::

            This method is called by ``__call__`` to allow the transformation to be
            applied directly to points using the syntax ``transformation(x_i)``.


        Parameters
        ----------
        x_i : ArrayLike
            Input points :math:`\vec{x_i}` with shape :math:`(\dots, n_i)`
            to which the transformation will be applied. A single point must be reshaped
            to :math:`(1, n_i)` before being passed to this method.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points
            (i.e., :math:`\mathbf{J}_x = \frac{\partial T}{\partial \vec{x_i}}`).

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters
            (i.e., :math:`\mathbf{J}_p = \frac{\partial T}{\partial \vec{p}}`).

        **kwargs
            Additional arguments passed to the underlying ``_apply`` method.


        Returns
        -------
        :class:`pysdic.TransformationResult`
            Contains the transformed points and the Jacobian matrices (if requested).
            The transformed points can be accessed using the ``x`` attribute with shape
            :math:`(\dots, n_o)`. The Jacobian matrices can be accessed using the ``jacobian_dx``
            and ``jacobian_dp`` attributes with shapes :math:`(\dots, n_o, n_i)` and
            :math:`(\dots, n_o, n_p)`, or None if not computed.


        Jacobians
        ---------
        Two Jacobian matrices can optionally be computed:

        - ``dx``: Jacobian of the transformed points with respect to the input points.
          Shape: :math:`(\dots, n_o, n_i)` where :math:`n_i` is the input dimension
          and :math:`n_o` is the output dimension.
          Example:

            .. code-block:: python

                jacobian_dx[..., 0, 0]  # ∂x_o/∂x_i
                jacobian_dx[..., 0, 1]  # ∂x_o/∂y_i
                jacobian_dx[..., 1, 0]  # ∂y_o/∂x_i
                jacobian_dx[..., 1, 1]  # ∂y_o/∂y_i

        - ``dp``: Jacobian of the transformed points with respect to the transformation parameters.
          Shape: :math:`(\dots, n_o, n_p)` where :math:`n_p` is the number of parameters.
          Example:

            .. code-block:: python

                jacobian_dp[..., 0, 0]  # ∂x_o/∂λ_1
                jacobian_dp[..., 0, 1]  # ∂x_o/∂λ_2
                jacobian_dp[..., 1, 0]  # ∂y_o/∂λ_1
                jacobian_dp[..., 1, 1]  # ∂y_o/∂λ_2

        The Jacobians are only computed if ``dx`` or ``dp`` are True.


        Developer Notes
        ----------------
        Subclasses must implement the ``_apply`` method. This method should:

        1. Accept a NumPy array of shape :math:`(n_\text{points}, n_i)`.
        2. Return three objects:

            - ``x_o``: shape :math:`(n_\text{points}, n_o)`.
            - ``jacobian_dx``: shape :math:`(n_\text{points}, n_o, n_i)` if ``dx=True``, else None.
            - ``jacobian_dp``: shape :math:`(n_\text{points}, n_o, n_p)` if ``dp=True``, else None.

        The shapes of the outputs of ``_apply`` are checked, and a ``RuntimeError`` is
        raised if they do not match the expected shapes.

        """
        if not self.has_apply:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not implement apply."
            )

        dx = _validate_bool(dx, "dx")
        if dx and not self.has_apply_dx:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to the input points (dx) in the apply method."
            )

        dp = _validate_bool(dp, "dp")
        if dp and not self.has_apply_dp:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to the parameters (dp) in the apply method."
            )

        if not self.is_setup:
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before "
                "transforming points."
            )

        # Reshape the input points into a 2D array for _apply method
        x_i, batch_shape = self._validate_points(x_i, self.input_dim, "x_i")
        n_points = x_i.shape[0]

        # Apply the transformation
        output = self._apply(
            x_i, dx=dx, dp=dp, **kwargs
        )  # (n_points, n_o), (n_points, n_o, n_i), (n_points, n_o, n_p)

        # Validate the outputs of the transformation
        x_o, jacobian_dx, jacobian_dp = self._validate_output(
            output,
            dx=dx,
            dp=dp,
            n_points=n_points,
            dim_out=self.output_dim,
            dim_in=self.input_dim,
            method="_apply",
        )

        # Reshape the outputs to the original batch shape
        return TransformationResult(
            x=x_o.reshape(*batch_shape, self.output_dim),
            jacobian_dx=(
                jacobian_dx.reshape(*batch_shape, self.output_dim, self.input_dim)
                if jacobian_dx is not None
                else None
            ),
            jacobian_dp=(
                jacobian_dp.reshape(*batch_shape, self.output_dim, self.n_parameters)
                if jacobian_dp is not None
                else None
            ),
        )

    def __call__(
        self,
        x_i: ArrayLike,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> TransformationResult:
        r"""
        Allow the transformation to be called directly on input points.

        This method simply calls the ``apply`` method, allowing the transformation to be
        applied using the syntax ``transformation(x_i)`` instead of ``transformation.apply(x_i)``.

        Parameters
        ----------
        x_i : ArrayLike
            Input points to which the transformation will be applied with shape :math:`(\dots, n_i)`.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters.

        **kwargs
            Additional arguments passed to the underlying ``_apply`` method.


        Returns
        -------
        :class:`pysdic.TransformationResult`
            The result of applying the transformation, containing the transformed points and optional Jacobians.

        """
        return self.apply(x_i, dx=dx, dp=dp, **kwargs)

    def invert(
        self,
        x_o: ArrayLike,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> TransformationResult:
        r"""
        Apply the inverse map :math:`T^{-1}` to the given output points :math:`\vec{x_o} \in \mathbb{R}^{n_o}`
        to return the corresponding input points :math:`\vec{x_i} \in \mathbb{R}^{n_i}`.

        .. math::

            \vec{x_i} = T^{-1}(\vec{x_o}; \vec{p})

        .. important::

            - The invert method can only be used if the transformation
              implements it (i.e., ``has_invert`` must be True).
            - The transformation must be fully set up (i.e., ``is_setup`` must be True).
            - The jacobian ``dx`` and ``dp`` can only be computed if the transformation
              supports it (i.e., ``has_invert_dx`` and ``has_invert_dp`` must be True).

        .. note::

            The relation between ``invert`` and ``apply`` is given by the flags
            ``is_left_invertible`` (:math:`T^{-1} \circ T = \text{Id}`) and
            ``is_right_invertible`` (:math:`T \circ T^{-1} = \text{Id}`).


        Parameters
        ----------
        x_o : ArrayLike
            Output points :math:`\vec{x_o}` with shape :math:`(\dots, n_o)`
            for which to compute the inverse transformation. A single point must be reshaped
            to :math:`(1, n_o)` before being passed to this method.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the output points
            (i.e., :math:`\mathbf{J}_x = \frac{\partial T^{-1}}{\partial \vec{x_o}}`).

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters
            (i.e., :math:`\mathbf{J}_p = \frac{\partial T^{-1}}{\partial \vec{p}}`).

        **kwargs
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        :class:`pysdic.TransformationResult`
            Contains the inverted input points and the Jacobian matrices (if requested).
            The inverted input points can be accessed using the ``x`` attribute with shape
            :math:`(\dots, n_i)`. The Jacobian matrices can be accessed using the ``jacobian_dx``
            and ``jacobian_dp`` attributes with shapes :math:`(\dots, n_i, n_o)` and
            :math:`(\dots, n_i, n_p)`, or None if not computed.


        Jacobians
        ---------
        Two Jacobian matrices can optionally be computed:

        - ``dx``: Jacobian of the inverted input points with respect to the output points.
          Shape: :math:`(\dots, n_i, n_o)` where :math:`n_i` is the input dimension
          and :math:`n_o` is the output dimension.
          Example:

            .. code-block:: python

                jacobian_dx[..., 0, 0]  # ∂x_i/∂x_o
                jacobian_dx[..., 0, 1]  # ∂x_i/∂y_o
                jacobian_dx[..., 1, 0]  # ∂y_i/∂x_o
                jacobian_dx[..., 1, 1]  # ∂y_i/∂y_o

        - ``dp``: Jacobian of the inverted input points with respect to the transformation parameters.
          Shape: :math:`(\dots, n_i, n_p)` where :math:`n_p` is the number of parameters.
          Example:

            .. code-block:: python

                jacobian_dp[..., 0, 0]  # ∂x_i/∂λ_1
                jacobian_dp[..., 0, 1]  # ∂x_i/∂λ_2
                jacobian_dp[..., 1, 0]  # ∂y_i/∂λ_1
                jacobian_dp[..., 1, 1]  # ∂y_i/∂λ_2

        The Jacobians are only computed if ``dx`` or ``dp`` are True.


        Developer Notes
        -----------------
        Subclasses must implement the ``_invert`` method. This method should:

        1. Accept a NumPy array of shape :math:`(n_\text{points}, n_o)`.
        2. Return three objects:

            - ``x_i``: shape :math:`(n_\text{points}, n_i)`.
            - ``jacobian_dx``: shape :math:`(n_\text{points}, n_i, n_o)` if ``dx=True``, else None.
            - ``jacobian_dp``: shape :math:`(n_\text{points}, n_i, n_p)` if ``dp=True``, else None.

        The shapes of the outputs of ``_invert`` are checked, and a ``RuntimeError`` is
        raised if they do not match the expected shapes.

        """
        if not self.has_invert:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not implement invert."
            )

        dx = _validate_bool(dx, "dx")
        if dx and not self.has_invert_dx:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to the output points (dx) in the invert method."
            )

        dp = _validate_bool(dp, "dp")
        if dp and not self.has_invert_dp:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to the parameters (dp) in the invert method."
            )

        if not self.is_setup:
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before "
                "inverting points."
            )

        # Reshape the output points into a 2D array for _invert method
        x_o, batch_shape = self._validate_points(x_o, self.output_dim, "x_o")
        n_points = x_o.shape[0]

        # Invert the transformation
        output = self._invert(
            x_o, dx=dx, dp=dp, **kwargs
        )  # (n_points, n_i), (n_points, n_i, n_o), (n_points, n_i, n_p)

        # Validate the outputs of the inverse map
        x_i, jacobian_dx, jacobian_dp = self._validate_output(
            output,
            dx=dx,
            dp=dp,
            n_points=n_points,
            dim_out=self.input_dim,
            dim_in=self.output_dim,
            method="_invert",
        )

        # Reshape the outputs to the original batch shape
        return TransformationResult(
            x=x_i.reshape(*batch_shape, self.input_dim),
            jacobian_dx=(
                jacobian_dx.reshape(*batch_shape, self.input_dim, self.output_dim)
                if jacobian_dx is not None
                else None
            ),
            jacobian_dp=(
                jacobian_dp.reshape(*batch_shape, self.input_dim, self.n_parameters)
                if jacobian_dp is not None
                else None
            ),
        )

    def __matmul__(self, other: Transformation) -> ComposedTransformation:
        r"""
        Compose two transformations using the syntax ``T2 @ T1``.

        .. math::

            (T_2 \circ T_1)(\vec{x_i}) = T_2(T_1(\vec{x_i}))

        This is equivalent to ``ComposedTransformation([T1, T2])``: the right operand
        :math:`T_1` is applied first. The output dimension of :math:`T_1` must match
        the input dimension of :math:`T_2`.

        .. code-block:: python

            T = T3 @ T2 @ T1  # ComposedTransformation([ComposedTransformation([T1, T2]), T3])

        Parameters
        ----------
        other : Transformation
            The transformation :math:`T_1` applied first.

        Returns
        -------
        :class:`pysdic.ComposedTransformation`
            The composition :math:`T_2 \circ T_1`.

        """
        if not isinstance(other, Transformation):
            return NotImplemented
        return ComposedTransformation([other, self])

    # ------
    # Numerical checks
    # ------
    @staticmethod
    def _finite_difference_jacobian_dx(
        func: Callable[[numpy.ndarray], numpy.ndarray],
        x: numpy.ndarray,
        n_out: int,
        eps: float,
    ) -> numpy.ndarray:
        # Central finite differences with respect to the points (n_points, n_in).
        # All the points are perturbed at once since the transformation acts pointwise.
        n_points, n_in = x.shape
        jacobian = numpy.empty((n_points, n_out, n_in), dtype=float)
        for k in range(n_in):
            h = eps * numpy.maximum(1.0, numpy.abs(x[:, k]))  # (n_points,)
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[:, k] += h
            x_minus[:, k] -= h
            step = (x_plus[:, k] - x_minus[:, k])[:, None]  # (n_points, 1)
            jacobian[:, :, k] = (func(x_plus) - func(x_minus)) / step
        return jacobian

    def _finite_difference_jacobian_dp(
        self,
        func: Callable[[], numpy.ndarray],
        n_points: int,
        n_out: int,
        eps: float,
    ) -> numpy.ndarray:
        # Central finite differences with respect to the parameters.
        # The parameters are temporarily modified and always restored.
        parameters = numpy.array(self.parameters, dtype=float, copy=True)
        jacobian = numpy.empty((n_points, n_out, self.n_parameters), dtype=float)
        try:
            for k in range(self.n_parameters):
                h = eps * max(1.0, abs(parameters[k]))
                p_plus = parameters.copy()
                p_minus = parameters.copy()
                p_plus[k] += h
                p_minus[k] -= h
                self.set_parameters(p_plus, override_lock=True)
                f_plus = func()
                self.set_parameters(p_minus, override_lock=True)
                f_minus = func()
                jacobian[:, :, k] = (f_plus - f_minus) / (p_plus[k] - p_minus[k])
        finally:
            self.set_parameters(parameters, override_lock=True)
        return jacobian

    def _compare_arrays(
        self,
        actual: numpy.ndarray,
        expected: numpy.ndarray,
        *,
        rtol: float,
        atol: float,
        context: str,
        raise_error: bool,
    ) -> bool:
        passed = bool(numpy.allclose(actual, expected, rtol=rtol, atol=atol))
        if not passed and raise_error:
            error = numpy.abs(actual - expected)
            excess = error - (atol + rtol * numpy.abs(expected))  # > 0 where the check fails
            index = numpy.unravel_index(numpy.argmax(excess), error.shape)
            raise AssertionError(
                f"[{self.__class__.__name__}] {context} check failed: absolute error "
                f"{error[index]:.3e} at index {tuple(int(i) for i in index)} "
                f"(actual={actual[index]:.6e}, expected={expected[index]:.6e}, "
                f"rtol={rtol}, atol={atol})."
            )
        return passed

    def check_apply_dx(
        self,
        x_i: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the apply method with respect to the input points against
        central finite differences at the given input points :math:`\vec{x_i}`.

        .. math::

            \frac{\partial T}{\partial x_{i,k}} \approx \frac{T(\vec{x_i} + h_k \vec{e_k}) - T(\vec{x_i} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |x_{i,k}|)`.

        .. note::

            This check is only valid at the given points. It is recommended to perform
            it on points covering the domain of interest.

        Parameters
        ----------
        x_i : ArrayLike
            Input points :math:`\vec{x_i}` with shape :math:`(\dots, n_i)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_apply`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_apply_dx:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian dx in the apply method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        x_i, _ = self._validate_points(x_i, self.input_dim, "x_i")

        jacobian = self.apply(x_i, dx=True, **kwargs).jacobian_dx  # (n_points, n_o, n_i)
        jacobian_fd = self._finite_difference_jacobian_dx(
            lambda x: self.apply(x, **kwargs).x, x_i, self.output_dim, eps
        )  # (n_points, n_o, n_i)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Apply Jacobian (dx)",
            raise_error=raise_error,
        )

    def check_apply_dp(
        self,
        x_i: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the apply method with respect to the parameters against
        central finite differences at the given input points :math:`\vec{x_i}` and the
        current parameters :math:`\vec{p}`.

        .. math::

            \frac{\partial T}{\partial p_k} \approx \frac{T(\vec{x_i}; \vec{p} + h_k \vec{e_k}) - T(\vec{x_i}; \vec{p} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |p_k|)`.

        .. note::

            The parameters are temporarily modified during the check (even if they are locked)
            and are always restored at the end.

        Parameters
        ----------
        x_i : ArrayLike
            Input points :math:`\vec{x_i}` with shape :math:`(\dots, n_i)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_apply`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_apply_dp:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian dp in the apply method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        x_i, _ = self._validate_points(x_i, self.input_dim, "x_i")

        jacobian = self.apply(x_i, dp=True, **kwargs).jacobian_dp  # (n_points, n_o, n_p)
        jacobian_fd = self._finite_difference_jacobian_dp(
            lambda: self.apply(x_i, **kwargs).x, x_i.shape[0], self.output_dim, eps
        )  # (n_points, n_o, n_p)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Apply Jacobian (dp)",
            raise_error=raise_error,
        )

    def check_invert_dx(
        self,
        x_o: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the invert method with respect to the output points against
        central finite differences at the given output points :math:`\vec{x_o}`.

        .. math::

            \frac{\partial T^{-1}}{\partial x_{o,k}} \approx \frac{T^{-1}(\vec{x_o} + h_k \vec{e_k}) - T^{-1}(\vec{x_o} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |x_{o,k}|)`.

        Parameters
        ----------
        x_o : ArrayLike
            Output points :math:`\vec{x_o}` with shape :math:`(\dots, n_o)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_invert_dx:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian dx in the invert method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        x_o, _ = self._validate_points(x_o, self.output_dim, "x_o")

        jacobian = self.invert(x_o, dx=True, **kwargs).jacobian_dx  # (n_points, n_i, n_o)
        jacobian_fd = self._finite_difference_jacobian_dx(
            lambda x: self.invert(x, **kwargs).x, x_o, self.input_dim, eps
        )  # (n_points, n_i, n_o)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Invert Jacobian (dx)",
            raise_error=raise_error,
        )

    def check_invert_dp(
        self,
        x_o: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the invert method with respect to the parameters against
        central finite differences at the given output points :math:`\vec{x_o}` and the
        current parameters :math:`\vec{p}`.

        .. math::

            \frac{\partial T^{-1}}{\partial p_k} \approx \frac{T^{-1}(\vec{x_o}; \vec{p} + h_k \vec{e_k}) - T^{-1}(\vec{x_o}; \vec{p} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |p_k|)`.

        .. note::

            The parameters are temporarily modified during the check (even if they are locked)
            and are always restored at the end.

        Parameters
        ----------
        x_o : ArrayLike
            Output points :math:`\vec{x_o}` with shape :math:`(\dots, n_o)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_invert_dp:
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian dp in the invert method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        x_o, _ = self._validate_points(x_o, self.output_dim, "x_o")

        jacobian = self.invert(x_o, dp=True, **kwargs).jacobian_dp  # (n_points, n_i, n_p)
        jacobian_fd = self._finite_difference_jacobian_dp(
            lambda: self.invert(x_o, **kwargs).x, x_o.shape[0], self.input_dim, eps
        )  # (n_points, n_i, n_p)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Invert Jacobian (dp)",
            raise_error=raise_error,
        )

    def check_left_invertible(
        self,
        x_i: ArrayLike,
        *,
        check_dx: bool = False,
        check_dp: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        apply_kwargs: Optional[Dict[str, Any]] = None,
        invert_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        r"""
        Check numerically that the invert method is a left inverse of the apply method
        at the given input points :math:`\vec{x_i}`.

        .. math::

            T^{-1}(T(\vec{x_i})) = \vec{x_i}

        If ``check_dx`` is True, the consistency of the Jacobians with respect to the points
        is also checked using the chain rule:

        .. math::

            \frac{\partial T^{-1}}{\partial \vec{x_o}}(T(\vec{x_i})) \cdot \frac{\partial T}{\partial \vec{x_i}}(\vec{x_i}) = \mathbf{I}_{n_i}

        If ``check_dp`` is True, the consistency of the Jacobians with respect to the parameters
        is also checked by differentiating the identity with respect to :math:`\vec{p}`:

        .. math::

            \frac{\partial T^{-1}}{\partial \vec{x_o}}(T(\vec{x_i})) \cdot \frac{\partial T}{\partial \vec{p}}(\vec{x_i}) + \frac{\partial T^{-1}}{\partial \vec{p}}(T(\vec{x_i})) = \mathbf{0}

        .. note::

            This check is only valid at the given points. It is recommended to perform
            it on points covering the domain of interest.

        Parameters
        ----------
        x_i : ArrayLike
            Input points :math:`\vec{x_i}` with shape :math:`(\dots, n_i)` at which to perform the check.

        check_dx : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians with respect to the points.

        check_dp : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians with respect to the parameters.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        apply_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_apply`` method.

        invert_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        bool
            True if the invert method behaves as a left inverse at the given points, False otherwise.

        """
        if not (self.has_apply and self.has_invert):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must implement both "
                f"apply and invert to check the left inverse."
            )
        check_dx = _validate_bool(check_dx, "check_dx")
        check_dp = _validate_bool(check_dp, "check_dp")
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        if check_dx and not (self.has_apply_dx and self.has_invert_dx):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must support Jacobian dx "
                f"computation in both apply and invert to check the Jacobians (check_dx=True)."
            )
        if check_dp and not (
            self.has_apply_dp and self.has_invert_dp and self.has_invert_dx
        ):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must support Jacobian dp "
                f"computation in both apply and invert, and Jacobian dx computation in invert, "
                f"to check the Jacobians (check_dp=True)."
            )
        apply_kwargs = _validate_kwargs(apply_kwargs, "apply_kwargs")
        invert_kwargs = _validate_kwargs(invert_kwargs, "invert_kwargs")
        x_i, _ = self._validate_points(x_i, self.input_dim, "x_i")

        applied = self.apply(x_i, dx=check_dx, dp=check_dp, **apply_kwargs)
        inverted = self.invert(
            applied.x, dx=check_dx or check_dp, dp=check_dp, **invert_kwargs
        )

        passed = self._compare_arrays(
            inverted.x,
            x_i,
            rtol=rtol,
            atol=atol,
            context="Left inverse (invert(apply(x_i)) = x_i)",
            raise_error=raise_error,
        )

        if check_dx:
            product = inverted.jacobian_dx @ applied.jacobian_dx  # (n_points, n_i, n_i)
            passed_dx = self._compare_arrays(
                product,
                numpy.broadcast_to(numpy.eye(self.input_dim), product.shape),
                rtol=rtol,
                atol=atol,
                context="Left inverse Jacobians (J_invert_dx @ J_apply_dx = I)",
                raise_error=raise_error,
            )
            passed = passed and passed_dx

        if check_dp:
            total = (
                inverted.jacobian_dx @ applied.jacobian_dp + inverted.jacobian_dp
            )  # (n_points, n_i, n_p)
            passed_dp = self._compare_arrays(
                total,
                numpy.zeros_like(total),
                rtol=rtol,
                atol=atol,
                context="Left inverse Jacobians (J_invert_dx @ J_apply_dp + J_invert_dp = 0)",
                raise_error=raise_error,
            )
            passed = passed and passed_dp

        return passed

    def check_right_invertible(
        self,
        x_o: ArrayLike,
        *,
        check_dx: bool = False,
        check_dp: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        apply_kwargs: Optional[Dict[str, Any]] = None,
        invert_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        r"""
        Check numerically that the invert method is a right inverse of the apply method
        at the given output points :math:`\vec{x_o}`.

        .. math::

            T(T^{-1}(\vec{x_o})) = \vec{x_o}

        If ``check_dx`` is True, the consistency of the Jacobians with respect to the points
        is also checked using the chain rule:

        .. math::

            \frac{\partial T}{\partial \vec{x_i}}(T^{-1}(\vec{x_o})) \cdot \frac{\partial T^{-1}}{\partial \vec{x_o}}(\vec{x_o}) = \mathbf{I}_{n_o}

        If ``check_dp`` is True, the consistency of the Jacobians with respect to the parameters
        is also checked by differentiating the identity with respect to :math:`\vec{p}`:

        .. math::

            \frac{\partial T}{\partial \vec{x_i}}(T^{-1}(\vec{x_o})) \cdot \frac{\partial T^{-1}}{\partial \vec{p}}(\vec{x_o}) + \frac{\partial T}{\partial \vec{p}}(T^{-1}(\vec{x_o})) = \mathbf{0}

        .. note::

            This check is only valid at the given points. It is recommended to perform
            it on points covering the domain of interest.

        Parameters
        ----------
        x_o : ArrayLike
            Output points :math:`\vec{x_o}` with shape :math:`(\dots, n_o)` at which to perform the check.

        check_dx : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians with respect to the points.

        check_dp : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians with respect to the parameters.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        apply_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_apply`` method.

        invert_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_invert`` method.


        Returns
        -------
        bool
            True if the invert method behaves as a right inverse at the given points, False otherwise.

        """
        if not (self.has_apply and self.has_invert):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must implement both "
                f"apply and invert to check the right inverse."
            )
        check_dx = _validate_bool(check_dx, "check_dx")
        check_dp = _validate_bool(check_dp, "check_dp")
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        if check_dx and not (self.has_apply_dx and self.has_invert_dx):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must support Jacobian dx "
                f"computation in both apply and invert to check the Jacobians (check_dx=True)."
            )
        if check_dp and not (
            self.has_apply_dp and self.has_invert_dp and self.has_apply_dx
        ):
            raise ValueError(
                f"Transformation of type '{self.__class__.__name__}' must support Jacobian dp "
                f"computation in both apply and invert, and Jacobian dx computation in apply, "
                f"to check the Jacobians (check_dp=True)."
            )
        apply_kwargs = _validate_kwargs(apply_kwargs, "apply_kwargs")
        invert_kwargs = _validate_kwargs(invert_kwargs, "invert_kwargs")
        x_o, _ = self._validate_points(x_o, self.output_dim, "x_o")

        inverted = self.invert(x_o, dx=check_dx, dp=check_dp, **invert_kwargs)
        applied = self.apply(
            inverted.x, dx=check_dx or check_dp, dp=check_dp, **apply_kwargs
        )

        passed = self._compare_arrays(
            applied.x,
            x_o,
            rtol=rtol,
            atol=atol,
            context="Right inverse (apply(invert(x_o)) = x_o)",
            raise_error=raise_error,
        )

        if check_dx:
            product = applied.jacobian_dx @ inverted.jacobian_dx  # (n_points, n_o, n_o)
            passed_dx = self._compare_arrays(
                product,
                numpy.broadcast_to(numpy.eye(self.output_dim), product.shape),
                rtol=rtol,
                atol=atol,
                context="Right inverse Jacobians (J_apply_dx @ J_invert_dx = I)",
                raise_error=raise_error,
            )
            passed = passed and passed_dx

        if check_dp:
            total = (
                applied.jacobian_dx @ inverted.jacobian_dp + applied.jacobian_dp
            )  # (n_points, n_o, n_p)
            passed_dp = self._compare_arrays(
                total,
                numpy.zeros_like(total),
                rtol=rtol,
                atol=atol,
                context="Right inverse Jacobians (J_apply_dx @ J_invert_dp + J_apply_dp = 0)",
                raise_error=raise_error,
            )
            passed = passed and passed_dp

        return passed

    def __str__(self) -> str:
        r"""
        Return a string representation of the transformation.

        .. code-block:: text

            [<TransformationClassName>] (Transformation)
                Input Dimension: <input_dim>
                Output Dimension: <output_dim>
                Number of Parameters: <n_parameters>
                Is Setup: <is_setup>
                Parameters:
                    p_0: <value of parameter 0>
                    p_1: <value of parameter 1>
                    ...
                Kwargs:
                    key_0: <value of constant 0>
                    key_1: <value of constant 1>
                    ...

        Returns
        -------
        str
            A string representation of the transformation.
        """
        repr_string = f"[{self.__class__.__name__}] (Transformation)\n"
        repr_string += f"\tInput Dimension: {self.input_dim}\n"
        repr_string += f"\tOutput Dimension: {self.output_dim}\n"
        repr_string += f"\tNumber of Parameters: {self.n_parameters}\n"
        repr_string += f"\tIs Setup: {self.is_setup}\n"
        parameters = self.parameters
        if self.n_parameters > 0 and parameters is not None:
            repr_string += f"\tParameters:\n"
            for i in range(self.n_parameters):
                repr_string += f"\t\tp_{i}: {parameters[i]}\n"
        data = self._to_dict()
        if len(data) > 0:
            repr_string += f"\tKwargs:\n"
            for key, value in data.items():
                repr_string += f"\t\t{key}: {value}\n"

        return repr_string


class ComposedTransformation(Transformation):
    r"""
    A composition of multiple :class:`pysdic.Transformation` instances.

    This class allows you to compose multiple transformations together, where the output of one
    transformation is fed as the input to the next. The resulting composition behaves like a single
    transformation that applies the sequence of transformations.

    For example, if you have two transformations :math:`T_1` and :math:`T_2`, you can create a composition
    :math:`T = T_2 \circ T_1` such that:

    .. math::

        T(\vec{x_i}; \vec{p}) = T_2(T_1(\vec{x_i}; \vec{p}_1); \vec{p}_2)

    where :math:`\vec{x_i}` are the input points to the composition, which are first transformed by
    :math:`T_1` and then by :math:`T_2` to produce the final output points. The inverse map is applied
    in the reverse order:

    .. math::

        T^{-1}(\vec{x_o}; \vec{p}) = T_1^{-1}(T_2^{-1}(\vec{x_o}; \vec{p}_2); \vec{p}_1)

    The parameters of the composition are the concatenation of the parameters of the
    transformations, in the order of the composition:

    .. math::

        \vec{p} = \begin{bmatrix} \vec{p}_1 & \vec{p}_2 & \dots & \vec{p}_n \end{bmatrix}

    .. seealso::

        - Class :class:`pysdic.Transformation` for the base class of transformations.
        - The syntax ``T2 @ T1`` which is equivalent to ``ComposedTransformation([T1, T2])``.

    .. note::

        The number of transformations in the composition must be at least 1 and cannot
        be changed after initialization. In addition, the transformations
        stored in the composition cannot be changed after initialization.

    .. note::

        Locking the composition (by an owner of the package) also locks all its
        transformations. The parameters of the composition cannot be set if the composition
        or one of its transformations is locked (unless ``override_lock=True``).

    .. note::

        The composition does not store its own parameters: getting the ``parameters`` of the
        composition reads the parameters of the transformations, and setting them splits the
        array and sets the parameters of each transformation. The transformations are not
        copied, so modifying the parameters of a transformation also modifies the parameters
        of the composition.

    .. note::

        The capability flags of the composition are derived from the flags of the
        transformations: a flag is True only if it is True for all the transformations.
        In addition, the Jacobian with respect to the parameters of a transformation
        must be propagated through the transformations applied after it, so:

        - ``has_apply_dp`` also requires ``has_apply_dx`` for :math:`T_2, \dots, T_n`.
        - ``has_invert_dp`` also requires ``has_invert_dx`` for :math:`T_1, \dots, T_{n-1}`.

    Parameters
    ----------
    transformations : List[Transformation]
        A list of transformation instances to compose together. The output dimension of each
        transformation must match the input dimension of the next.
        The order of the list is :math:`[T_1, T_2, ..., T_n]` where :math:`T_1` is the first
        transformation applied to the input points and :math:`T_n` is the last transformation
        that produces the final output points.

    parameters : Optional[ArrayLike], optional (default: ``None``)
        A 1D array of shape :math:`(n_p,)` with :math:`n_p = \sum_k n_{p,k}` to set the
        parameters of all the transformations. If None, the current parameters of the
        transformations are kept.

    """

    def __init__(
        self,
        transformations: List[Transformation],
        *,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        if not isinstance(transformations, (list, tuple)):
            raise TypeError(
                f"transformations must be a list or tuple of Transformation instances, got {type(transformations)}"
            )
        transformations = list(transformations)  # Convert to list if it's a tuple
        if len(transformations) == 0:
            raise ValueError("transformations list cannot be empty.")
        for i, t in enumerate(transformations):
            if not isinstance(t, Transformation):
                raise TypeError(
                    f"All elements in transformations must be instances of Transformation, got {type(t)} at index {i}"
                )
            if i > 0 and t.input_dim != transformations[i - 1].output_dim:
                raise ValueError(
                    f"Output dimension of transformation at index {i-1} ({transformations[i - 1].output_dim}) "
                    f"must match input dimension of transformation at index {i} ({t.input_dim})."
                )

        self._transformations = transformations

        # Capability flags derived from the transformations
        self._has_apply = all(t.has_apply for t in transformations)
        self._has_apply_dx = all(t.has_apply_dx for t in transformations)
        self._has_apply_dp = all(t.has_apply_dp for t in transformations) and all(
            t.has_apply_dx for t in transformations[1:]
        )
        self._has_invert = all(t.has_invert for t in transformations)
        self._has_invert_dx = all(t.has_invert_dx for t in transformations)
        self._has_invert_dp = all(t.has_invert_dp for t in transformations) and all(
            t.has_invert_dx for t in transformations[:-1]
        )
        self._is_left_invertible = all(t.is_left_invertible for t in transformations)
        self._is_right_invertible = all(t.is_right_invertible for t in transformations)

        super().__init__(parameters=parameters)

    # ------
    # Properties
    # ------
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters :math:`n_p` of the composition, i.e. the sum of the
        number of parameters of the transformations.

        Returns
        -------
        int
            The number of parameters of the composition.

        """
        return sum(t.n_parameters for t in self._transformations)

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the composition, i.e. the input
        dimension of the first transformation :math:`T_1`.

        Returns
        -------
        int
            The input dimension of the composition.

        """
        return self._transformations[0].input_dim

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the composition, i.e. the output
        dimension of the last transformation :math:`T_n`.

        Returns
        -------
        int
            The output dimension of the composition.

        """
        return self._transformations[-1].output_dim

    @property
    def transformations(self) -> List[Transformation]:
        r"""
        [Get] the list of transformations that are composed together.

        Returns
        -------
        List[Transformation]
            The list of transformation instances that make up the composition.
        """
        return list(
            self._transformations
        )  # Return a copy to prevent external modification

    @property
    def n_transformations(self) -> int:
        r"""
        [Get] the number of transformations in the composition.

        Returns
        -------
        int
            The number of transformations in the composition.
        """
        return len(self._transformations)

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] or [Set] the parameters of the composition.

        The parameters of the composition are the concatenation of the parameters
        of the transformations, in the order of the composition.

        .. note::

            The returned array is read-only. Setting the parameters splits the array
            and sets the parameters of each transformation (see ``set_parameters``).

        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the composition with shape :math:`(n_p,)`, or None
            if the parameters of at least one transformation are not set.
        """
        params = [t.parameters for t in self._transformations]
        if any(p is None for p in params):
            return None
        params = numpy.concatenate(params, axis=0)
        params.flags.writeable = False
        return params

    @parameters.setter
    def parameters(self, value: Optional[ArrayLike]) -> None:
        self.set_parameters(value, override_lock=False)

    def set_parameters(
        self, value: Optional[ArrayLike], *, override_lock: bool = False
    ) -> None:
        r"""
        Set the parameters of the composition, with an option to override the lock.

        The input parameters are split according to the number of parameters of each
        transformation and set to each transformation in the sequence.

        .. note::

            The lock of the composition and the locks of the transformations are both
            respected, unless ``override_lock`` is True. All the locks and the input are
            checked before any modification: if an error is raised, the parameters of
            all the transformations are unchanged.

        Parameters
        ----------
        value : Optional[ArrayLike]
            A 1D array of shape :math:`(n_p,)` containing the parameters to set,
            or None to unset the parameters of all the transformations.

        override_lock : bool, optional (default: ``False``)
            Whether to override the locks and allow setting parameters even if they are locked.

        Raises
        ------
        ValueError
            If the input value is not a 1D array, does not have the correct shape or is not finite.

        RuntimeError
            If the parameters are locked and ``override_lock`` is False.

        """
        override_lock = _validate_bool(override_lock, "override_lock")
        # Locks of the composition and of all the transformations, before any modification
        self._check_unlocked("set the parameters", override_lock=override_lock)
        if value is not None:
            value = self._validate_parameters(value)

        # Everything is checked: the transformations are set without checking their locks again
        if value is None:
            for t in self._transformations:
                t.set_parameters(None, override_lock=True)
            return

        start = 0
        for t in self._transformations:
            end = start + t.n_parameters
            t.set_parameters(value[start:end], override_lock=True)
            start = end

    # ------
    # Lock
    # ------
    @property
    def is_locked(self) -> bool:
        r"""
        [Get] Whether the composition or at least one of its transformations is locked.

        .. note::

            Use :meth:`is_locked_composition` to get the lock state of the composition
            alone.

        Returns
        -------
        bool
            True if the parameters of the composition cannot be set without
            ``override_lock=True``.

        """
        return self.is_locked_composition or any(
            t.is_locked for t in self._transformations
        )

    @property
    def is_locked_composition(self) -> bool:
        r"""
        [Get] Whether the composition itself is locked (without looking at its
        transformations).

        Returns
        -------
        bool
            True if the composition is locked.

        """
        return getattr(self, "_locked", False)

    @property
    def locked_message(self) -> Optional[str]:
        r"""
        [Get] The explanation given when the composition was locked, or else the
        message of the first locked transformation.

        Returns
        -------
        Optional[str]
            The message of the lock, or None if nothing is locked or the lock has no
            message.

        """
        if self.is_locked_composition:
            return LockableMixin.locked_message.fget(self)
        for t in self._transformations:
            if t.is_locked:
                return t.locked_message
        return None

    def _set_locked(self, locked: bool = True, *, message: Optional[str] = None) -> None:
        r"""
        Lock or unlock the composition and all its transformations.

        The lock is propagated to the transformations (recursively for nested
        compositions) with the same ``message``.

        .. important::

            Internal use only (see :class:`pysdic.core.lockable.LockableMixin`).

        .. warning::

            The transformations are shared, not copied. Unlocking the composition also
            unlocks its transformations, even if they were locked by another owner: only
            unlock new objects (as ``copy`` does).

        Parameters
        ----------
        locked : bool, optional (default: ``True``)
            True to lock the composition and its transformations, False to unlock them.

        message : Optional[str], optional (default: ``None``)
            The reason of the lock, added to the errors raised when setting the
            parameters. Ignored when ``locked`` is False.

        Raises
        ------
        TypeError
            If ``locked`` is not a boolean or ``message`` is not a string.

        ValueError
            If ``message`` is an empty string.

        """
        super()._set_locked(locked, message=message)  # validates the inputs first
        for t in self._transformations:
            t._set_locked(locked, message=message)

    def _check_unlocked(self, action: str, *, override_lock: bool = False) -> None:
        # Check the composition, then each transformation (recursively): the error names
        # the locked object and gives its own message
        super()._check_unlocked(action, override_lock=override_lock)
        for t in self._transformations:
            t._check_unlocked(action, override_lock=override_lock)

    # ------
    # Serialization and copy
    # ------
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "transformations": [t.to_dict() for t in self._transformations],
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> ComposedTransformation:
        if not "transformations" in data:
            raise ValueError("Input dictionary must contain a 'transformations' key.")
        transformations_data = data["transformations"]
        if not isinstance(transformations_data, (tuple, list)):
            raise TypeError(
                f"'transformations' key in input dictionary must be a list or tuple, got {type(transformations_data)}"
            )
        transformations = [
            Transformation.from_dict(t_data) for t_data in transformations_data
        ]
        return cls(
            transformations=transformations, parameters=data.get("parameters", None)
        )

    def _copy(self) -> ComposedTransformation:
        return self.__class__(
            transformations=[t.copy() for t in self._transformations],
        )

    # ------
    # Apply and invert
    # ------
    def _active_dp_list(
        self, dp: bool, bypass_dp: Optional[Sequence[bool]]
    ) -> List[bool]:
        # Return for each transformation whether its Jacobian dp is computed
        if bypass_dp is None:
            bypass_dp = [False] * self.n_transformations
        if not isinstance(bypass_dp, (list, tuple, numpy.ndarray)):
            raise TypeError(
                f"bypass_dp must be a sequence of booleans, got {type(bypass_dp)}"
            )
        if len(bypass_dp) != self.n_transformations:
            raise ValueError(
                f"Length of bypass_dp must match the number of transformations in the composition ({self.n_transformations}), got {len(bypass_dp)}"
            )
        bypass_dp = [_validate_bool(b, f"bypass_dp[{i}]") for i, b in enumerate(bypass_dp)]
        return [bool(dp) and not b for b in bypass_dp]

    def _parameter_columns(self) -> List[slice]:
        # The columns of the parameters of each transformation in the Jacobian dp of the composition
        columns, start = [], 0
        for t in self._transformations:
            columns.append(slice(start, start + t.n_parameters))
            start += t.n_parameters
        return columns

    def _fill_jacobian_dp(
        self,
        jacobian_dp: numpy.ndarray,
        active: List[bool],
        blocks: List[Optional[numpy.ndarray]],
        last: int,
        last_result: TransformationResult,
    ) -> None:
        # Write the Jacobians of the parameters in their columns of the preallocated jacobian_dp
        # (n_points, n_rows, n_p), at the last step 'last' of the chain (T_n for apply, T_1 for invert):
        # - the blocks of the previous steps are propagated through the last step directly in their
        #   columns (matmul with out=, no temporary array),
        # - the block of the last step is copied in its columns,
        # - the columns of the bypassed blocks are filled with NaN (nothing is computed).
        for j, columns in enumerate(self._parameter_columns()):
            if columns.stop == columns.start:
                continue
            if not active[j]:
                jacobian_dp[:, :, columns] = numpy.nan
            elif j == last:
                jacobian_dp[:, :, columns] = last_result.jacobian_dp
            else:
                numpy.matmul(last_result.jacobian_dx, blocks[j], out=jacobian_dp[:, :, columns])

    def _apply(
        self,
        x_i: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        kwargs: Optional[List[Optional[Dict[str, Any]]]] = None,
        bypass_dp: Optional[Sequence[bool]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the composition of transformations to the input points :math:`\vec{x_i}`
        to produce the output points :math:`\vec{x_o}`.

        .. math::

            \vec{x_o} = T_n(T_{n-1}(...T_1(\vec{x_i})...))

        Lets :math:`\vec{y}_0 = \vec{x_i}` and :math:`\vec{y}_k = T_k(\vec{y}_{k-1}; \vec{p}_k)`
        be the intermediate points. The Jacobian with respect to the input points is computed using
        the chain rule across the sequence of transformations:

        .. math::

            \mathbf{J}_x = \frac{\partial T_n}{\partial \vec{y}_{n-1}} \cdot ... \cdot \frac{\partial T_2}{\partial \vec{y}_1} \cdot \frac{\partial T_1}{\partial \vec{x_i}}

        The Jacobian with respect to the parameters is the concatenation of the Jacobians with
        respect to the parameters of each transformation, propagated through the transformations
        applied after it:

        .. math::

            \mathbf{J}_p = \begin{bmatrix}
            \frac{\partial T}{\partial \vec{p}_1} & \frac{\partial T}{\partial \vec{p}_2} & \dots & \frac{\partial T}{\partial \vec{p}_n}
            \end{bmatrix}
            \quad \text{with} \quad
            \frac{\partial T}{\partial \vec{p}_k} = \frac{\partial T_n}{\partial \vec{y}_{n-1}} \cdot ... \cdot \frac{\partial T_{k+1}}{\partial \vec{y}_k} \cdot \frac{\partial T_k}{\partial \vec{p}_k}

        Parameters
        ----------
        x_i : numpy.ndarray
            The input points to transform, with shape :math:`(n_\text{points}, n_i)`
            where :math:`n_i` is the input dimension of the first transformation in
            the composition.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters.

        kwargs : Optional[List[Optional[Dict[str, Any]]]], optional (default: ``None``)
            A list of dictionaries containing additional keyword arguments to pass to each
            transformation's apply method. The list must have the same length as the number
            of transformations in the composition, in the order :math:`[T_1, T_2, ..., T_n]`.
            ``None`` elements are replaced by empty dictionaries.

        bypass_dp : Optional[Sequence[bool]], optional (default: ``None``)
            A sequence of booleans with the same length as the number of transformations, in the
            order :math:`[T_1, T_2, ..., T_n]`. If ``dp=True``, the Jacobian with respect to the
            parameters of the transformations with ``True`` is not computed (nor propagated) and its
            columns in ``jacobian_dp`` are filled with NaN. If None, no Jacobian is bypassed.
            Ignored if ``dp=False``.


        Returns
        -------
        numpy.ndarray
            The transformed output points, with shape :math:`(n_\text{points}, n_o)` where :math:`n_o`
            is the output dimension of the last transformation in the composition.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the input points if ``dx=True``, with shape
            :math:`(n_\text{points}, n_o, n_i)`, or None if ``dx=False``.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the parameters if ``dp=True``, with shape
            :math:`(n_\text{points}, n_o, n_p)`, or None if ``dp=False``, where
            :math:`n_p` is the total number of parameters of the composition.

        """
        kwargs = _validate_kwargs_sequence(kwargs, self.n_transformations, "kwargs")
        active = self._active_dp_list(dp, bypass_dp)  # dp computed for each transformation

        x = x_i
        jacobian_dx = None
        jacobian_dp_blocks = [None] * self.n_transformations  # Jacobians of the current points w.r.t. each p_k
        last = self.n_transformations - 1
        # Preallocated Jacobian dp (n_points, n_o, n_p): only the requested blocks are computed, directly
        # in their columns at the last step (no concatenation)
        jacobian_dp = numpy.empty((x_i.shape[0], self.output_dim, self.n_parameters)) if dp else None

        for k, (t, kwarg) in enumerate(zip(self._transformations, kwargs)):
            # The Jacobian dx of T_k is needed to propagate the Jacobians of the previous parameters
            propagate = any(active[:k])
            result = t.apply(x, dx=dx or propagate, dp=active[k], **kwarg)
            x = result.x

            if k == last:
                if dp:
                    self._fill_jacobian_dp(jacobian_dp, active, jacobian_dp_blocks, last, result)
            else:
                if propagate:
                    for j in range(k):
                        if active[j]:
                            jacobian_dp_blocks[j] = (
                                result.jacobian_dx @ jacobian_dp_blocks[j]
                            )  # (n_points, n_out_k, n_p_j)
                if active[k]:
                    jacobian_dp_blocks[k] = result.jacobian_dp  # (n_points, n_out_k, n_p_k)

            if dx:
                jacobian_dx = (
                    result.jacobian_dx
                    if jacobian_dx is None
                    else result.jacobian_dx @ jacobian_dx
                )  # (n_points, n_out_k, n_i)

        return x, jacobian_dx, jacobian_dp  # (n_points, n_o), (n_points, n_o, n_i), (n_points, n_o, n_p)

    def _invert(
        self,
        x_o: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        kwargs: Optional[List[Optional[Dict[str, Any]]]] = None,
        bypass_dp: Optional[Sequence[bool]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse of the composition of transformations to the output points :math:`\vec{x_o}`
        to produce the input points :math:`\vec{x_i}`.

        .. math::

            \vec{x_i} = T_1^{-1}(T_2^{-1}(...T_n^{-1}(\vec{x_o})...))

        Lets :math:`\vec{z}_n = \vec{x_o}` and :math:`\vec{z}_{k-1} = T_k^{-1}(\vec{z}_k; \vec{p}_k)`
        be the intermediate points. The Jacobian with respect to the output points is computed using
        the chain rule across the sequence of transformations in reverse:

        .. math::

            \mathbf{J}_x = \frac{\partial T_1^{-1}}{\partial \vec{z}_1} \cdot ... \cdot \frac{\partial T_{n-1}^{-1}}{\partial \vec{z}_{n-1}} \cdot \frac{\partial T_n^{-1}}{\partial \vec{x_o}}

        The Jacobian with respect to the parameters is the concatenation of the Jacobians with
        respect to the parameters of each transformation, propagated through the inverse maps
        applied after it:

        .. math::

            \mathbf{J}_p = \begin{bmatrix}
            \frac{\partial T^{-1}}{\partial \vec{p}_1} & \frac{\partial T^{-1}}{\partial \vec{p}_2} & \dots & \frac{\partial T^{-1}}{\partial \vec{p}_n}
            \end{bmatrix}
            \quad \text{with} \quad
            \frac{\partial T^{-1}}{\partial \vec{p}_k} = \frac{\partial T_1^{-1}}{\partial \vec{z}_1} \cdot ... \cdot \frac{\partial T_{k-1}^{-1}}{\partial \vec{z}_{k-1}} \cdot \frac{\partial T_k^{-1}}{\partial \vec{p}_k}

        Parameters
        ----------
        x_o : numpy.ndarray
            The output points to invert, with shape :math:`(n_\text{points}, n_o)`
            where :math:`n_o` is the output dimension of the last transformation in
            the composition.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the output points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters.

        kwargs : Optional[List[Optional[Dict[str, Any]]]], optional (default: ``None``)
            A list of dictionaries containing additional keyword arguments to pass to each
            transformation's invert method. The list must have the same length as the number
            of transformations in the composition, in the order :math:`[T_1, T_2, ..., T_n]`
            (the same order as for applying, even if the inversion is applied in reverse).
            ``None`` elements are replaced by empty dictionaries.

        bypass_dp : Optional[Sequence[bool]], optional (default: ``None``)
            A sequence of booleans with the same length as the number of transformations, in the
            order :math:`[T_1, T_2, ..., T_n]`. If ``dp=True``, the Jacobian with respect to the
            parameters of the transformations with ``True`` is not computed (nor propagated) and its
            columns in ``jacobian_dp`` are filled with NaN. If None, no Jacobian is bypassed.
            Ignored if ``dp=False``.


        Returns
        -------
        numpy.ndarray
            The inverted input points, with shape :math:`(n_\text{points}, n_i)` where :math:`n_i`
            is the input dimension of the first transformation in the composition.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the output points if ``dx=True``, with shape
            :math:`(n_\text{points}, n_i, n_o)`, or None if ``dx=False``.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the parameters if ``dp=True``, with shape
            :math:`(n_\text{points}, n_i, n_p)`, or None if ``dp=False``, where
            :math:`n_p` is the total number of parameters of the composition.

        """
        kwargs = _validate_kwargs_sequence(kwargs, self.n_transformations, "kwargs")
        active = self._active_dp_list(dp, bypass_dp)  # dp computed for each transformation

        x = x_o
        jacobian_dx = None
        jacobian_dp_blocks = [None] * self.n_transformations  # Filled from T_n to T_1
        # Preallocated Jacobian dp (n_points, n_i, n_p): only the requested blocks are computed, directly
        # in their columns at the last step T_1^{-1} (no concatenation)
        jacobian_dp = numpy.empty((x_o.shape[0], self.input_dim, self.n_parameters)) if dp else None

        for k in reversed(range(self.n_transformations)):
            t = self._transformations[k]
            # The Jacobian dx of T_k^{-1} is needed to propagate the Jacobians of the previous parameters
            propagate = any(active[k + 1 :])
            result = t.invert(x, dx=dx or propagate, dp=active[k], **kwargs[k])
            x = result.x

            if k == 0:
                if dp:
                    self._fill_jacobian_dp(jacobian_dp, active, jacobian_dp_blocks, 0, result)
            else:
                if propagate:
                    for j in range(k + 1, self.n_transformations):
                        if active[j]:
                            jacobian_dp_blocks[j] = (
                                result.jacobian_dx @ jacobian_dp_blocks[j]
                            )  # (n_points, n_in_k, n_p_j)
                if active[k]:
                    jacobian_dp_blocks[k] = result.jacobian_dp  # (n_points, n_in_k, n_p_k)

            if dx:
                jacobian_dx = (
                    result.jacobian_dx
                    if jacobian_dx is None
                    else result.jacobian_dx @ jacobian_dx
                )  # (n_points, n_in_k, n_o)

        return x, jacobian_dx, jacobian_dp  # (n_points, n_i), (n_points, n_i, n_o), (n_points, n_i, n_p)