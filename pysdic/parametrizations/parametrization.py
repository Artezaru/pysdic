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
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple
from numpy.typing import ArrayLike

import numpy
import datetime
import inspect

from ..__version__ import __version__
from ..core.serialization import JsonSerializerMixin, Hdf5SerializerMixin
from ..core.lockable import LockableMixin
from .._validation import (
    _validate_bool,
    _validate_real,
    _validate_str,
    _validate_tolerances,
    _validate_kwargs,
    _validate_kwargs_sequence,
    _as_array,
)


@dataclass(slots=True, frozen=True)
class ParametrizationResult:
    r"""
    A class to represent the result of a parametrization.

    This class is used to store the results of a parametrization, including the
    encoded parameters and the Jacobian matrices.

    .. seealso::

        - :class:`pysdic.Parametrization` for the parametrizations that
          produce a :class:`pysdic.ParametrizationResult` object.

    Lets :math:`P` be a parametrization that maps input parameters :math:`\vec{p_i}`
    from an input parametric space :math:`\mathbb{R}^{n_i}` to output parameters :math:`\vec{p_o}`
    in an output parametric space :math:`\mathbb{R}^{n_o}`. Then the result of applying the
    parametrization can be represented as a :class:`pysdic.ParametrizationResult` object with the following attributes:

    Parameters
    ----------
    p : numpy.ndarray
        The encoded parameters :math:`\vec{p_o}` after applying the parametrization.
        Shape :math:`(n_o,)`.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the input parameters :math:`\vec{p_i}`.
        Shape :math:`(n_o, n_i)`.

    """

    p: numpy.ndarray
    jacobian_dp: Optional[numpy.ndarray] = None



class Parametrization(LockableMixin, JsonSerializerMixin, Hdf5SerializerMixin):
    r"""
    Abstract base class representing a parametric transformation :math:`P` that
    maps parameters :math:`\vec{p_i}` from an input parametric space
    :math:`\mathbb{R}^{n_i}` to an output parametric space :math:`\mathbb{R}^{n_o}`.

    .. math::

        \vec{p_o} = P(\vec{p_i})

    All parametrizations can depend on **constants**, which are fixed values
    that influence the parametrization.

    .. note::

        A parametrization is locked (see :class:`pysdic.core.lockable.LockableMixin`) when
        it is owned by another object of the package that assumes it does not change (for
        example a :class:`pysdic.ReparametrizedTransformation`). A locked parametrization
        refuses the modifications of its constants (for the subclasses with modifiable
        constants): ``is_locked`` and ``locked_message`` tell whether it is locked and why.
        A copy is always unlocked.

    A parametrization can be applied to input parameters :math:`\vec{p_i} \in \mathbb{R}^{n_i}`
    using the ``encode`` method, which returns a :class:`pysdic.ParametrizationResult`
    object containing the encoded parameters :math:`\vec{p_o} \in \mathbb{R}^{n_o}` and
    the Jacobian matrix :math:`\mathbf{J}_p = \frac{\partial P}{\partial \vec{p_i}}` (if requested).

    A parametrization can also define a decoding map :math:`P^{-1}` using the ``decode`` method,
    which returns a :class:`pysdic.ParametrizationResult` object
    containing the decoded parameters :math:`\vec{p_i} \in \mathbb{R}^{n_i}` and the Jacobian matrix
    :math:`\mathbf{J}_p = \frac{\partial P^{-1}}{\partial \vec{p_o}}` (if requested).

    .. seealso::
 
        - Class :class:`pysdic.ParametrizationResult` for the result returned by the
          ``encode`` and ``decode`` methods.

    .. note::

        When :math:`n_i \neq n_o`, the map :math:`P` cannot be bijective and :math:`P^{-1}`
        denotes a left inverse or a right inverse of :math:`P` (see the flags
        ``is_left_invertible`` and ``is_right_invertible`` below).

    .. note::

        Only single parameter vectors are supported: shape :math:`(n_i,)` for ``encode``
        and :math:`(n_o,)` for ``decode``. Batched parameters with shape
        :math:`(\dots, n_i)` or :math:`(\dots, n_o)` are not supported.

    .. seealso::

        To compose parametrizations (ie :math:`P = P_2 \circ P_1`),
        use the following classes:

        - Class :class:`pysdic.ComposedParametrization` to allow multiple parametrizations to be composed together.
        - The syntax ``P2 @ P1`` which is equivalent to ``ComposedParametrization([P1, P2])``.

    .. seealso::

        To change the parametric space of the transformation (ie :math:`\vec{p_o} = P(\vec{p_i})`),
        use the following classes:

        - Class :class:`pysdic.Transformation` : the base class for transformations.
        - Class :class:`pysdic.ReparametrizedTransformation` to allow a transformation
          to be associated with a parametrization to encode its parameters.


    Capability flags
    ----------------

    Each parametrization declares its capabilities with the following boolean
    class/instance attributes. All flags default to ``False``.

    - ``_has_encode``: the method ``_encode`` is implemented (i.e. :math:`P` can be evaluated).
    - ``_has_encode_dp``: the method ``_encode`` can compute :math:`\frac{\partial P}{\partial \vec{p_i}}`.
    - ``_has_decode``: the method ``_decode`` is implemented (i.e. :math:`P^{-1}` can be evaluated).
    - ``_has_decode_dp``: the method ``_decode`` can compute :math:`\frac{\partial P^{-1}}{\partial \vec{p_o}}`.
    - ``_is_left_invertible``: ``decode`` is a left inverse of ``encode``, i.e.
      :math:`P^{-1}(P(\vec{p_i})) = \vec{p_i}` for all :math:`\vec{p_i} \in \mathbb{R}^{n_i}`.
    - ``_is_right_invertible``: ``decode`` is a right inverse of ``encode``, i.e.
      :math:`P(P^{-1}(\vec{p_o})) = \vec{p_o}` for all :math:`\vec{p_o} \in \mathbb{R}^{n_o}`.

    The flags must be consistent:

    - ``_has_encode_dp`` requires ``_has_encode``.
    - ``_has_decode_dp`` requires ``_has_decode``.
    - ``_is_left_invertible`` and ``_is_right_invertible`` require both ``_has_encode`` and ``_has_decode``.

    The consistency is checked at initialization by ``Parametrization.__init__``.
    The flags ``_is_left_invertible`` and ``_is_right_invertible`` are declarations
    that can be verified numerically with the methods ``check_left_invertible`` and
    ``check_right_invertible``.


    Implementation details
    ----------------------

    - All subclasses of :class:`pysdic.Parametrization` must call ``super().__init__()``
      at the end of their ``__init__`` method to check the consistency of the flags.

    - Subclasses with ``_has_encode=True`` must implement the method ``_encode``, and
      subclasses with ``_has_decode=True`` must implement the method ``_decode``.
      These methods are called by the public methods ``encode`` and ``decode``
      respectively, which handle the validation of the inputs and outputs of the
      parametrization with optional Jacobian computation. If a flag is ``False``,
      the corresponding method does not need to be implemented.

    - All subclasses must implement the properties ``input_dim`` and ``output_dim``
      to return the input and output dimensions of the parametrization.

    - All subclasses must implement the method ``_to_dict``, ``_from_dict``, and
      ``_copy`` called by the public methods, ``from_dict``, ``to_dict``
      and ``copy`` will handle copying and the serialization and
      deserialization of the **constants** of the parametrization.

    - All subclasses must set the capability flags described above, either as class attributes
      (if they do not depend on the constants) or as instance attributes before calling
      ``super().__init__()`` (if they depend on the constants).

    - The constants should be fixed at the initialization (read-only properties). If a
      subclass allows to modify a constant after the initialization, each method that
      modifies it must call ``self._check_unlocked("set the <name>")`` before any
      modification, so that a locked parametrization cannot be modified.

    Here is a minimal example of how to implement a simple subclass of :class:`pysdic.Parametrization`
    representing a scaling of parameters :math:`p_o = s \cdot p_i` with a fixed scale
    factor :math:`s` as a **constant**:

    .. code-block:: python

        class ScalingParametrization(Parametrization):

            _has_encode = True
            _has_encode_dp = True
            _has_decode = True
            _has_decode_dp = True
            _is_left_invertible = True
            _is_right_invertible = True

            def __init__(
                self,
                scale_factor: float = 1.0
            ):
                scale_factor = float(scale_factor)
                if scale_factor == 0.0:
                    raise ValueError("scale_factor must be non-zero to be invertible.")
                self._scale_factor = scale_factor
                super().__init__()  # Check the consistency of the flags

            @property
            def scale_factor(self) -> float:
                return self._scale_factor  # Read-only constant

            @property
            def input_dim(self) -> int:
                return 1  # 1D input parameters

            @property
            def output_dim(self) -> int:
                return 1  # 1D output parameters

            def _to_dict(self) -> Dict[str, Any]:
                return {"scale_factor": self.scale_factor}

            @classmethod
            def _from_dict(cls, data: Dict[str, Any]) -> ScalingParametrization:
                return cls(scale_factor=data["scale_factor"])

            def _copy(self) -> ScalingParametrization:
                return self.__class__(scale_factor=self.scale_factor)

            def _encode(self, p_i: numpy.ndarray, *, dp: bool = False, **kwargs):
                p_o = self.scale_factor * p_i  # (1,)
                jacobian_dp = numpy.array([[self.scale_factor]]) if dp else None  # (1, 1)
                return p_o, jacobian_dp

            def _decode(self, p_o: numpy.ndarray, *, dp: bool = False, **kwargs):
                p_i = p_o / self.scale_factor  # (1,)
                jacobian_dp = numpy.array([[1.0 / self.scale_factor]]) if dp else None  # (1, 1)
                return p_i, jacobian_dp

    The implementation can then be verified numerically:

    .. code-block:: python

        P = ScalingParametrization(scale_factor=2.0)
        P.encode([3.0]).p                              # array([6.])
        P.check_encode_dp([3.0], raise_error=True)     # True
        P.check_left_invertible([3.0], check_dp=True)  # True

    """

    __subclass_registry: ClassVar[Dict[str, type]] = {}

    # Capability flags (conservative defaults, overridden by subclasses)
    _has_encode: bool = False
    _has_encode_dp: bool = False
    _has_decode: bool = False
    _has_decode_dp: bool = False
    _is_left_invertible: bool = False
    _is_right_invertible: bool = False

    # Explanation of the errors raised by a parametrization locked without a message
    _default_locked_message: str = (
        "It is used by another object of the package that assumes that it does not change. "
        "Work on a copy (parametrization.copy()) to get a parametrization that can be "
        "modified."
    )

    def __init__(
        self,
    ) -> None:
        self._assert_flags_set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "Parametrization":
            raise TypeError(
                "subclass of Parametrization cannot be named 'Parametrization' to avoid conflicts with the base class."
            )
        if cls.__name__ in cls.__subclass_registry:
            raise ValueError(
                f"A subclass of Parametrization with name '{cls.__name__}' is already registered."
            )
        cls.__subclass_registry[cls.__name__] = cls

    @classmethod
    def __get_subclass_by_name(cls, name: str) -> type:
        if not name in cls.__subclass_registry:
            raise ValueError(
                f"No subclass of Parametrization with name '{name}' is registered."
            )
        return cls.__subclass_registry[name]

    def _assert_flags_set(self) -> None:

        required_flags = {
            "_has_encode": "indicate whether the parametrization implements encode().",
            "_has_encode_dp": "indicate whether the parametrization supports Jacobian computation in encode().",
            "_has_decode": "indicate whether the parametrization implements decode().",
            "_has_decode_dp": "indicate whether the parametrization supports Jacobian computation in decode().",
            "_is_left_invertible": "indicate whether decode() is a left inverse of encode().",
            "_is_right_invertible": "indicate whether decode() is a right inverse of encode().",
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
        if self._has_encode_dp and not self._has_encode:
            raise ValueError(
                f"Subclass {self.__class__.__name__} sets '_has_encode_dp=True' "
                f"but '_has_encode=False'."
            )
        if self._has_decode_dp and not self._has_decode:
            raise ValueError(
                f"Subclass {self.__class__.__name__} sets '_has_decode_dp=True' "
                f"but '_has_decode=False'."
            )
        if (self._is_left_invertible or self._is_right_invertible) and not (
            self._has_encode and self._has_decode
        ):
            raise ValueError(
                f"Subclass {self.__class__.__name__} is declared left or right invertible "
                f"but does not implement both encode ('_has_encode') and decode ('_has_decode')."
            )

    # ------
    # To be implemented by subclasses
    # ------
    @property
    @abstractmethod
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the parametrization.

        Returns
        -------
        int
            The input dimension of the parametrization.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the property 'input_dim' "
            f"to return the input dimension of the parametrization."
        )

    @property
    @abstractmethod
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the parametrization.

        Returns
        -------
        int
            The output dimension of the parametrization.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the property 'output_dim' "
            f"to return the output dimension of the parametrization."
        )

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        r"""
        Serialize the parametrization to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the constants of the parametrization for
            serialization in the 'to_dict' method.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the method '_to_dict' "
            f"to return a dictionary of constants for serialization in the 'to_dict' method."
        )

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Parametrization:
        r"""
        Deserialize the parametrization from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the parameters and constants for the parametrization.

        Returns
        -------
        :class:`pysdic.Parametrization`
            An instance of the parametrization initialized with the data from the dictionary.

        """
        raise NotImplementedError(
            f"Subclass {cls.__name__} must implement the class method '_from_dict' "
            f"to return an instance of the parametrization initialized with the data "
            f"extracted from the input dictionary for deserialization in the 'from_dict' method."
        )

    @abstractmethod
    def _copy(self) -> Parametrization:
        r"""
        Return a copy of the parametrization.

        Returns
        -------
        :class:`pysdic.Parametrization`
            A copy of the parametrization.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the method '_copy' "
            f"to return a copy of the parametrization."
        )

    def _encode(
        self, p_i: numpy.ndarray, *, dp: bool = False, **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Apply the parametrization to input parameters :math:`\vec{p_i}`.

        .. note::

            This method must be implemented by subclasses setting ``_has_encode=True``.
            The input array is a copy owned by the parametrization and can be safely
            modified in place.

        Parameters
        ----------
        p_i : numpy.ndarray
            A 1D array of shape :math:`(n_i,)` containing the input parameters to encode.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the input parameters.

        **kwargs
            Additional arguments passed to the underlying ``_encode`` method.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray]]
            A tuple containing:
            - The encoded output parameters as a 1D array of shape :math:`(n_o,)`.
            - The Jacobian matrix with respect to the input parameters as a 2D array of
              shape :math:`(n_o, n_i)`, or None if not computed.

        """
        if self._has_encode:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} sets '_has_encode=True' but does not "
                f"implement the method '_encode' to apply the parametrization to input "
                f"parameters with optional Jacobian computation."
            )
        raise NotImplementedError(
            f"Parametrization of type '{self.__class__.__name__}' does not implement "
            f"encode ('_has_encode=False')."
        )

    def _decode(
        self, p_o: numpy.ndarray, *, dp: bool = False, **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Apply the decoding map :math:`P^{-1}` to output parameters :math:`\vec{p_o}`.

        .. note::

            This method must be implemented by subclasses setting ``_has_decode=True``.
            The input array is a copy owned by the parametrization and can be safely
            modified in place.

        Parameters
        ----------
        p_o : numpy.ndarray
            A 1D array of shape :math:`(n_o,)` containing the output parameters to decode.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the output parameters.

        **kwargs
            Additional arguments passed to the underlying ``_decode`` method.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray]]
            A tuple containing:
            - The decoded input parameters as a 1D array of shape :math:`(n_i,)`.
            - The Jacobian matrix with respect to the output parameters as a 2D array of
              shape :math:`(n_i, n_o)`, or None if not computed.

        """
        if self._has_decode:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} sets '_has_decode=True' but does not "
                f"implement the method '_decode' to decode output parameters with "
                f"optional Jacobian computation."
            )
        raise NotImplementedError(
            f"Parametrization of type '{self.__class__.__name__}' does not implement "
            f"decode ('_has_decode=False')."
        )

    # ------
    # Flags
    # ------
    @property
    def has_encode(self) -> bool:
        r"""
        [Get] whether the parametrization implements the encode method.

        Returns
        -------
        bool
            True if the parametrization implements the encode method, False otherwise.

        """
        return self._has_encode

    @property
    def has_encode_dp(self) -> bool:
        r"""
        [Get] whether the parametrization supports Jacobian computation in the encode method.

        Returns
        -------
        bool
            True if the parametrization supports Jacobian computation in the encode method, False otherwise.

        """
        return self._has_encode_dp

    @property
    def has_decode(self) -> bool:
        r"""
        [Get] whether the parametrization implements the decode method.

        Returns
        -------
        bool
            True if the parametrization implements the decode method, False otherwise.

        """
        return self._has_decode

    @property
    def has_decode_dp(self) -> bool:
        r"""
        [Get] whether the parametrization supports Jacobian computation in the decode method.

        Returns
        -------
        bool
            True if the parametrization supports Jacobian computation in the decode method, False otherwise.

        """
        return self._has_decode_dp

    @property
    def is_left_invertible(self) -> bool:
        r"""
        [Get] whether the decode method is a left inverse of the encode method.

        .. math::

            P^{-1}(P(\vec{p_i})) = \vec{p_i} \quad \forall \vec{p_i} \in \mathbb{R}^{n_i}

        Returns
        -------
        bool
            True if the decode method is a left inverse of the encode method, False otherwise.

        """
        return self._is_left_invertible

    @property
    def is_right_invertible(self) -> bool:
        r"""
        [Get] whether the decode method is a right inverse of the encode method.

        .. math::

            P(P^{-1}(\vec{p_o})) = \vec{p_o} \quad \forall \vec{p_o} \in \mathbb{R}^{n_o}

        Returns
        -------
        bool
            True if the decode method is a right inverse of the encode method, False otherwise.

        """
        return self._is_right_invertible

    @property
    def is_invertible(self) -> bool:
        r"""
        [Get] whether the parametrization is invertible (i.e. both left and right invertible).

        In this case :math:`P` is a bijection and :math:`n_i = n_o`.

        Returns
        -------
        bool
            True if the parametrization is both left and right invertible, False otherwise.

        """
        return self._is_left_invertible and self._is_right_invertible

    # ------
    # Internal validation helpers
    # ------
    def _validate_parameters(
        self, p: ArrayLike, dim: int, name: str
    ) -> numpy.ndarray:
        # Copy the parameters to protect the user's array from in-place modifications.
        # Batched parameters (ndim > 1) are not supported.
        return _as_array(p, name, numpy.float64, shape=(dim,), finite=False, copy=True)

    def _validate_output(
        self,
        output: Any,
        *,
        dp: bool,
        dim: int,
        jacobian_shape: Tuple[int, int],
        method: str,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        name = self.__class__.__name__
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError(
                f"Method '{method}' of '{name}' must return a tuple (p, jacobian_dp), "
                f"got {type(output)}."
            )
        p, jacobian_dp = output

        p = numpy.asarray(p, dtype=float)
        if p.shape != (dim,):
            raise RuntimeError(
                f"Method '{method}' of '{name}' returned parameters with shape {p.shape}, "
                f"expected ({dim},)."
            )

        if dp:
            if jacobian_dp is None:
                raise RuntimeError(
                    f"Method '{method}' of '{name}' returned None for the Jacobian "
                    f"while dp=True."
                )
            jacobian_dp = numpy.asarray(jacobian_dp, dtype=float)
            if jacobian_dp.shape != jacobian_shape:
                raise RuntimeError(
                    f"Method '{method}' of '{name}' returned a Jacobian with shape "
                    f"{jacobian_dp.shape}, expected {jacobian_shape}."
                )
        else:
            jacobian_dp = None  # Discard any Jacobian computed while not requested

        return p, jacobian_dp

    # ------
    # Methods
    # ------
    def copy(self) -> Parametrization:
        r"""
        Create a copy of the parametrization.

        The copy is always **unlocked** (see ``is_locked``), even if the parametrization is
        locked: it can be modified independently of the original.

        Returns
        -------
        Parametrization
            A new instance of the parametrization with the same constants.

        """
        new = self._copy()
        new._set_locked(False)
        return new

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Parametrization:
        r"""
        Create a parametrization instance from a dictionary.

        The class instantiated is given by the ``'type'`` key of the dictionary:

        - If called on :class:`pysdic.Parametrization`, the ``'type'`` key is required and
          any registered subclass can be instantiated.
        - If called on a subclass ``cls``, the ``'type'`` key is optional (defaults to ``cls``)
          and must designate ``cls`` or one of its subclasses.

        .. code-block:: python

            P = Parametrization.from_dict(data)       # Any registered parametrization
            P = MyParametrization.from_dict(data)     # MyParametrization or a subclass of it

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the constants for the parametrization.

        Returns
        -------
        Parametrization
            A new instance of the parametrization initialized with the data from the dictionary.

        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data)}")
        if not "context" in data:
            raise ValueError("Input dictionary must contain a 'context' key.")
        if data["context"] != "Parametrization":
            raise ValueError(
                f"Input dictionary has 'context' key '{data['context']}' that does not match 'Parametrization'."
            )

        t_type = _validate_str(data.get("type", None), "type", allow_none=True)
        if cls is Parametrization:
            # Dispatch to the correct subclass based on the 'type' key in the input dictionary
            if t_type is None:
                raise ValueError(
                    "Input dictionary must contain a 'type' key indicating the parametrization type."
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
                f"Cannot deserialize the abstract parametrization class '{t_class.__name__}'. "
                f"The 'type' key must designate a concrete subclass."
            )

        if not "kwargs" in data:
            raise ValueError("Input dictionary must contain a 'kwargs' key.")
        kwargs = data["kwargs"]  # dict of kwargs for the parametrization constructor
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"'kwargs' key in input dictionary must be a dictionary, got {type(kwargs)}"
            )
        return t_class._from_dict(dict(kwargs))  # Copy to protect the input dictionary

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Serialize the parametrization to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the constants of the parametrization,
            including a 'type' key indicating the parametrization type and
            the metadata 'version' and 'datetime' of the serialization.

        """
        return {
            "type": self.__class__.__name__,
            "context": "Parametrization",
            "version": __version__,
            "datetime": datetime.datetime.now().isoformat(),
            "kwargs": self._to_dict(),
        }

    def encode(
        self,
        p_i: ArrayLike,
        *,
        dp: bool = False,
        **kwargs,
    ) -> ParametrizationResult:
        r"""
        Apply the parametrization to the given parameters :math:`\vec{p_i} \in \mathbb{R}^{n_i}`
        to return the encoded parameters :math:`\vec{p_o} \in \mathbb{R}^{n_o}`.

        .. math::

            \vec{p_o} = P(\vec{p_i})


        .. important::

            - The encode method can only be used if the parametrization
              implements it (i.e., ``has_encode`` must be True).
            - The jacobian ``dp`` can only be computed if the parametrization
              supports it (i.e., ``has_encode_dp`` must be True).

        .. note::

            This method is called by ``__call__`` to allow the parametrization to be
            applied directly to parameters using the syntax ``parametrization(p_i)``.


        Parameters
        ----------
        p_i : ArrayLike
            Input parameters :math:`\vec{p_i}` with shape :math:`(n_i,)`
            to which the parametrization will be applied.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the input parameters.
            (i.e., :math:`\mathbf{J}_p = \frac{\partial P}{\partial \vec{p_i}}`).

        **kwargs
            Additional arguments passed to the underlying ``_encode`` method.


        Returns
        -------
        :class:`pysdic.ParametrizationResult`
            Contains the encoded parameters accessible using the ``p`` attribute with shape
            :math:`(n_o,)`, and the Jacobian matrix with respect to the input parameters
            accessible using the ``jacobian_dp`` attribute with shape
            :math:`(n_o, n_i)` if computed, or None otherwise.


        Notes
        ----------------
        The jacobian ``dp`` has the following structure:

        .. code-block:: python

            jacobian_dp[0, 0]  # ∂p_o1/∂p_i1
            jacobian_dp[0, 1]  # ∂p_o1/∂p_i2
            jacobian_dp[1, 0]  # ∂p_o2/∂p_i1
            jacobian_dp[1, 1]  # ∂p_o2/∂p_i2
            jacobian_dp[j, i]  # ∂p_oj/∂p_ii


        Developers notes
        -------------------
        Subclasses must implement the ``_encode`` method. This method should:

        1. Accept a NumPy array of shape :math:`(n_i,)`.
        2. Return :math:`\vec{p_o}` Numpy array of shape :math:`(n_o,)` and
           optionally the Jacobian matrix with respect to the input parameters
           as a 2D array of shape :math:`(n_o, n_i)` if ``dp`` is True, or None otherwise.

        The shapes of the outputs of ``_encode`` are checked, and a ``RuntimeError`` is
        raised if they do not match the expected shapes.

        """
        if not self.has_encode:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not implement encode."
            )

        dp = _validate_bool(dp, "dp")
        if dp and not self.has_encode_dp:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to input parameters (dp) in the encode method."
            )

        # Validate the input parameters as a 1D array for _encode method
        p_i = self._validate_parameters(p_i, self.input_dim, "p_i")

        # Apply the parametrization
        output = self._encode(p_i, dp=dp, **kwargs)  # (n_o,); (n_o, n_i) or None

        # Validate the outputs of the parametrization
        p_o, jacobian_dp = self._validate_output(
            output,
            dp=dp,
            dim=self.output_dim,
            jacobian_shape=(self.output_dim, self.input_dim),
            method="_encode",
        )

        # Return the result as a ParametrizationResult object
        return ParametrizationResult(p=p_o, jacobian_dp=jacobian_dp)

    def __call__(
        self,
        p_i: ArrayLike,
        *,
        dp: bool = False,
        **kwargs,
    ) -> ParametrizationResult:
        r"""
        Allow the parametrization to be applied directly to parameters using the syntax ``parametrization(p_i)``.

        This method is an alias for the ``encode`` method, allowing the parametrization to be called directly on input parameters.

        Parameters
        ----------
        p_i : ArrayLike
            Input parameters :math:`\vec{p_i}` with shape :math:`(n_i,)`
            to which the parametrization will be applied.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the input parameters.
            (i.e., :math:`\mathbf{J}_p = \frac{\partial P}{\partial \vec{p_i}}`).

        **kwargs
            Additional arguments passed to the underlying ``_encode`` method.


        Returns
        -------
        :class:`pysdic.ParametrizationResult`
            Contains the encoded parameters accessible using the ``p`` attribute with shape
            :math:`(n_o,)`, and the Jacobian matrix with respect to the input parameters
            accessible using the ``jacobian_dp`` attribute with shape
            :math:`(n_o, n_i)` if computed, or None otherwise.

        """
        return self.encode(p_i, dp=dp, **kwargs)

    def decode(
        self,
        p_o: ArrayLike,
        *,
        dp: bool = False,
        **kwargs,
    ) -> ParametrizationResult:
        r"""
        Apply the decoding map :math:`P^{-1}` to the output parameters :math:`\vec{p_o} \in \mathbb{R}^{n_o}`
        to return the decoded parameters :math:`\vec{p_i} \in \mathbb{R}^{n_i}`.

        .. math::

            \vec{p_i} = P^{-1}(\vec{p_o})

        .. important::

            - The decode method can only be used if the parametrization
              implements it (i.e., ``has_decode`` must be True).
            - The jacobian ``dp`` can only be computed if the parametrization
              supports it (i.e., ``has_decode_dp`` must be True).

        .. note::

            The relation between ``decode`` and ``encode`` is given by the flags
            ``is_left_invertible`` (:math:`P^{-1} \circ P = \text{Id}`) and
            ``is_right_invertible`` (:math:`P \circ P^{-1} = \text{Id}`).


        Parameters
        ----------
        p_o : ArrayLike
            Output parameters :math:`\vec{p_o}` with shape :math:`(n_o,)`
            to which the decoding map will be applied.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the output parameters.
            (i.e., :math:`\mathbf{J}_p = \frac{\partial P^{-1}}{\partial \vec{p_o}}`).

        **kwargs
            Additional arguments passed to the underlying ``_decode`` method.


        Returns
        -------
        :class:`pysdic.ParametrizationResult`
            Contains the decoded parameters accessible using the ``p`` attribute with shape
            :math:`(n_i,)`, and the Jacobian matrix with respect to the output parameters
            accessible using the ``jacobian_dp`` attribute with shape
            :math:`(n_i, n_o)` if computed, or None otherwise.


        Notes
        ----------------
        The jacobian ``dp`` has the following structure:

        .. code-block:: python

            jacobian_dp[0, 0]  # ∂p_i1/∂p_o1
            jacobian_dp[0, 1]  # ∂p_i1/∂p_o2
            jacobian_dp[1, 0]  # ∂p_i2/∂p_o1
            jacobian_dp[1, 1]  # ∂p_i2/∂p_o2
            jacobian_dp[i, j]  # ∂p_ii/∂p_oj


        Developers notes
        -------------------
        Subclasses must implement the ``_decode`` method. This method should:

        1. Accept a NumPy array of shape :math:`(n_o,)`.
        2. Return :math:`\vec{p_i}` Numpy array of shape :math:`(n_i,)` and
           optionally the Jacobian matrix with respect to the output parameters
           as a 2D array of shape :math:`(n_i, n_o)` if ``dp`` is True, or None otherwise.

        The shapes of the outputs of ``_decode`` are checked, and a ``RuntimeError`` is
        raised if they do not match the expected shapes.

        """
        if not self.has_decode:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not implement decode."
            )

        dp = _validate_bool(dp, "dp")
        if dp and not self.has_decode_dp:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian with respect to output parameters (dp) in the decode method."
            )

        # Validate the output parameters as a 1D array for _decode method
        p_o = self._validate_parameters(p_o, self.output_dim, "p_o")

        # Decode the parameters
        output = self._decode(p_o, dp=dp, **kwargs)  # (n_i,); (n_i, n_o) or None

        # Validate the outputs of the decoding map
        p_i, jacobian_dp = self._validate_output(
            output,
            dp=dp,
            dim=self.input_dim,
            jacobian_shape=(self.input_dim, self.output_dim),
            method="_decode",
        )

        # Return the result as a ParametrizationResult object
        return ParametrizationResult(p=p_i, jacobian_dp=jacobian_dp)

    def __matmul__(self, other: Parametrization) -> ComposedParametrization:
        r"""
        Compose two parametrizations using the syntax ``P2 @ P1``.

        .. math::

            (P_2 \circ P_1)(\vec{p_i}) = P_2(P_1(\vec{p_i}))

        This is equivalent to ``ComposedParametrization([P1, P2])``: the right operand
        :math:`P_1` is applied first. The output dimension of :math:`P_1` must match
        the input dimension of :math:`P_2`.

        .. code-block:: python

            P = P3 @ P2 @ P1  # ComposedParametrization([ComposedParametrization([P1, P2]), P3])

        Parameters
        ----------
        other : Parametrization
            The parametrization :math:`P_1` applied first.

        Returns
        -------
        :class:`pysdic.ComposedParametrization`
            The composition :math:`P_2 \circ P_1`.

        """
        if not isinstance(other, Parametrization):
            return NotImplemented
        return ComposedParametrization([other, self])

    # ------
    # Numerical checks
    # ------
    @staticmethod
    def _finite_difference_jacobian(
        func: Callable[[numpy.ndarray], numpy.ndarray],
        x: numpy.ndarray,
        n_out: int,
        eps: float,
    ) -> numpy.ndarray:
        # Central finite differences with a step relative to the magnitude of x
        jacobian = numpy.empty((n_out, x.size), dtype=float)
        for k in range(x.size):
            h = eps * max(1.0, abs(x[k]))
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += h
            x_minus[k] -= h
            jacobian[:, k] = (func(x_plus) - func(x_minus)) / (x_plus[k] - x_minus[k])
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
            index = numpy.unravel_index(numpy.argmax(error), error.shape)
            raise AssertionError(
                f"[{self.__class__.__name__}] {context} check failed: maximum absolute error "
                f"{error[index]:.3e} at index {tuple(int(i) for i in index)} "
                f"(actual={actual[index]:.6e}, expected={expected[index]:.6e}, "
                f"rtol={rtol}, atol={atol})."
            )
        return passed

    def check_encode_dp(
        self,
        p_i: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the encode method against central finite differences
        at the given input parameters :math:`\vec{p_i}`.

        .. math::

            \frac{\partial P}{\partial p_{i,k}} \approx \frac{P(\vec{p_i} + h_k \vec{e_k}) - P(\vec{p_i} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |p_{i,k}|)`.

        .. note::

            This check is only valid at the given point. It is recommended to perform
            it at several points of the parametric space.

        Parameters
        ----------
        p_i : ArrayLike
            Input parameters :math:`\vec{p_i}` with shape :math:`(n_i,)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_encode`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_encode_dp:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian in the encode method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        p_i = self._validate_parameters(p_i, self.input_dim, "p_i")

        jacobian = self.encode(p_i, dp=True, **kwargs).jacobian_dp  # (n_o, n_i)
        jacobian_fd = self._finite_difference_jacobian(
            lambda p: self.encode(p, **kwargs).p, p_i, self.output_dim, eps
        )  # (n_o, n_i)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Encode Jacobian (dp)",
            raise_error=raise_error,
        )

    def check_decode_dp(
        self,
        p_o: ArrayLike,
        *,
        eps: float = 1e-6,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        **kwargs,
    ) -> bool:
        r"""
        Check the Jacobian of the decode method against central finite differences
        at the given output parameters :math:`\vec{p_o}`.

        .. math::

            \frac{\partial P^{-1}}{\partial p_{o,k}} \approx \frac{P^{-1}(\vec{p_o} + h_k \vec{e_k}) - P^{-1}(\vec{p_o} - h_k \vec{e_k})}{2 h_k}

        with :math:`h_k = \varepsilon \max(1, |p_{o,k}|)`.

        .. note::

            This check is only valid at the given point. It is recommended to perform
            it at several points of the parametric space.

        Parameters
        ----------
        p_o : ArrayLike
            Output parameters :math:`\vec{p_o}` with shape :math:`(n_o,)` at which to check the Jacobian.

        eps : float, optional (default: ``1e-6``)
            The relative step :math:`\varepsilon` of the finite differences.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        **kwargs
            Additional arguments passed to the underlying ``_decode`` method.


        Returns
        -------
        bool
            True if the analytical Jacobian matches the finite differences, False otherwise.

        """
        if not self.has_decode_dp:
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' does not support "
                f"computing the Jacobian in the decode method, cannot check it."
            )
        eps = _validate_real(eps, "eps", positive=True, finite=True)
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        p_o = self._validate_parameters(p_o, self.output_dim, "p_o")

        jacobian = self.decode(p_o, dp=True, **kwargs).jacobian_dp  # (n_i, n_o)
        jacobian_fd = self._finite_difference_jacobian(
            lambda p: self.decode(p, **kwargs).p, p_o, self.input_dim, eps
        )  # (n_i, n_o)

        return self._compare_arrays(
            jacobian,
            jacobian_fd,
            rtol=rtol,
            atol=atol,
            context="Decode Jacobian (dp)",
            raise_error=raise_error,
        )

    def check_left_invertible(
        self,
        p_i: ArrayLike,
        *,
        check_dp: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        decode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        r"""
        Check numerically that the decode method is a left inverse of the encode method
        at the given input parameters :math:`\vec{p_i}`.

        .. math::

            P^{-1}(P(\vec{p_i})) = \vec{p_i}

        If ``check_dp`` is True, the consistency of the Jacobians is also checked
        using the chain rule:

        .. math::

            \frac{\partial P^{-1}}{\partial \vec{p_o}}(P(\vec{p_i})) \cdot \frac{\partial P}{\partial \vec{p_i}}(\vec{p_i}) = \mathbf{I}_{n_i}

        .. note::

            This check is only valid at the given point. It is recommended to perform
            it at several points of the parametric space.

        Parameters
        ----------
        p_i : ArrayLike
            Input parameters :math:`\vec{p_i}` with shape :math:`(n_i,)` at which to perform the check.

        check_dp : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians of encode and decode.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        encode_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_encode`` method.

        decode_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_decode`` method.


        Returns
        -------
        bool
            True if the decode method behaves as a left inverse at the given point, False otherwise.

        """
        if not (self.has_encode and self.has_decode):
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' must implement both "
                f"encode and decode to check the left inverse."
            )
        check_dp = _validate_bool(check_dp, "check_dp")
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        if check_dp and not (self.has_encode_dp and self.has_decode_dp):
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' must support Jacobian "
                f"computation in both encode and decode to check the Jacobians (check_dp=True)."
            )
        encode_kwargs = _validate_kwargs(encode_kwargs, "encode_kwargs")
        decode_kwargs = _validate_kwargs(decode_kwargs, "decode_kwargs")
        p_i = self._validate_parameters(p_i, self.input_dim, "p_i")

        encoded = self.encode(p_i, dp=check_dp, **encode_kwargs)
        decoded = self.decode(encoded.p, dp=check_dp, **decode_kwargs)

        passed = self._compare_arrays(
            decoded.p,
            p_i,
            rtol=rtol,
            atol=atol,
            context="Left inverse (decode(encode(p_i)) = p_i)",
            raise_error=raise_error,
        )

        if check_dp:
            product = decoded.jacobian_dp @ encoded.jacobian_dp  # (n_i, n_o) @ (n_o, n_i)
            passed_dp = self._compare_arrays(
                product,
                numpy.eye(self.input_dim),
                rtol=rtol,
                atol=atol,
                context="Left inverse Jacobians (J_decode @ J_encode = I)",
                raise_error=raise_error,
            )
            passed = passed and passed_dp

        return passed

    def check_right_invertible(
        self,
        p_o: ArrayLike,
        *,
        check_dp: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        raise_error: bool = False,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        decode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        r"""
        Check numerically that the decode method is a right inverse of the encode method
        at the given output parameters :math:`\vec{p_o}`.

        .. math::

            P(P^{-1}(\vec{p_o})) = \vec{p_o}

        If ``check_dp`` is True, the consistency of the Jacobians is also checked
        using the chain rule:

        .. math::

            \frac{\partial P}{\partial \vec{p_i}}(P^{-1}(\vec{p_o})) \cdot \frac{\partial P^{-1}}{\partial \vec{p_o}}(\vec{p_o}) = \mathbf{I}_{n_o}

        .. note::

            This check is only valid at the given point. It is recommended to perform
            it at several points of the parametric space.

        Parameters
        ----------
        p_o : ArrayLike
            Output parameters :math:`\vec{p_o}` with shape :math:`(n_o,)` at which to perform the check.

        check_dp : bool, optional (default: ``False``)
            If True, also check the consistency of the Jacobians of encode and decode.

        rtol : float, optional (default: ``1e-5``)
            The relative tolerance of the comparison (see ``numpy.allclose``).

        atol : float, optional (default: ``1e-8``)
            The absolute tolerance of the comparison (see ``numpy.allclose``).

        raise_error : bool, optional (default: ``False``)
            If True, raise an ``AssertionError`` with the maximum error if the check fails.

        encode_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_encode`` method.

        decode_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional arguments passed to the underlying ``_decode`` method.


        Returns
        -------
        bool
            True if the decode method behaves as a right inverse at the given point, False otherwise.

        """
        if not (self.has_encode and self.has_decode):
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' must implement both "
                f"encode and decode to check the right inverse."
            )
        check_dp = _validate_bool(check_dp, "check_dp")
        rtol, atol = _validate_tolerances(rtol, atol)
        raise_error = _validate_bool(raise_error, "raise_error")
        if check_dp and not (self.has_encode_dp and self.has_decode_dp):
            raise ValueError(
                f"Parametrization of type '{self.__class__.__name__}' must support Jacobian "
                f"computation in both encode and decode to check the Jacobians (check_dp=True)."
            )
        encode_kwargs = _validate_kwargs(encode_kwargs, "encode_kwargs")
        decode_kwargs = _validate_kwargs(decode_kwargs, "decode_kwargs")
        p_o = self._validate_parameters(p_o, self.output_dim, "p_o")

        decoded = self.decode(p_o, dp=check_dp, **decode_kwargs)
        encoded = self.encode(decoded.p, dp=check_dp, **encode_kwargs)

        passed = self._compare_arrays(
            encoded.p,
            p_o,
            rtol=rtol,
            atol=atol,
            context="Right inverse (encode(decode(p_o)) = p_o)",
            raise_error=raise_error,
        )

        if check_dp:
            product = encoded.jacobian_dp @ decoded.jacobian_dp  # (n_o, n_i) @ (n_i, n_o)
            passed_dp = self._compare_arrays(
                product,
                numpy.eye(self.output_dim),
                rtol=rtol,
                atol=atol,
                context="Right inverse Jacobians (J_encode @ J_decode = I)",
                raise_error=raise_error,
            )
            passed = passed and passed_dp

        return passed

    def __str__(self) -> str:
        r"""
        Return a string representation of the parametrization.

        .. code-block:: text

            [<ParametrizationClassName>] (Parametrization)
                Input Dimension: <input_dim>
                Output Dimension: <output_dim>
                Kwargs:
                    key_0: <value of constant 0>
                    key_1: <value of constant 1>
                    ...

        Returns
        -------
        str
            A string representation of the parametrization.
        """
        repr_string = f"[{self.__class__.__name__}] (Parametrization)\n"
        repr_string += f"\tInput Dimension: {self.input_dim}\n"
        repr_string += f"\tOutput Dimension: {self.output_dim}\n"
        data = self._to_dict()
        if len(data) > 0:
            repr_string += f"\tKwargs:\n"
            for key, value in data.items():
                repr_string += f"\t\t{key}: {value}\n"

        return repr_string
    

class ComposedParametrization(Parametrization):
    r"""
    A composition of multiple :class:`pysdic.Parametrization` instances.

    This class allows you to compose multiple parametrizations together, where the output of one
    parametrization is fed as the input to the next. The resulting composition behaves like a single
    parametrization that encodes parameters through the sequence of parametrizations.

    For example, if you have two parametrizations :math:`P_1` and :math:`P_2`, you can create a composition
    :math:`P = P_2 \circ P_1` such that:

    .. math::

        P(\vec{p}) = P_2(P_1(\vec{p}))

    where :math:`\vec{p}` are the parameters to encode. The decoding map is applied
    in the reverse order:

    .. math::

        P^{-1}(\vec{p}) = P_1^{-1}(P_2^{-1}(\vec{p}))

    .. seealso::

        - Class :class:`pysdic.Parametrization` for the base class of parametrizations.
        - The syntax ``P2 @ P1`` which is equivalent to ``ComposedParametrization([P1, P2])``.

    .. note::

        The number of parametrizations in the composition must be at least 1 and cannot
        be changed after initialization. In addition, the parametrizations
        stored in the composition cannot be changed after initialization.

    .. note::

        Locking the composition (by an owner of the package) also locks all its
        parametrizations, recursively for nested compositions.

    .. note::

        The capability flags of the composition are derived from the flags of the
        parametrizations: a flag is True only if it is True for all the parametrizations.
        For example, the composition is left invertible if all the parametrizations are
        left invertible, since:

        .. math::

            P_1^{-1} \circ P_2^{-1} \circ P_2 \circ P_1 = P_1^{-1} \circ P_1 = \text{Id}

    Parameters
    ----------
    parametrizations : List[Parametrization]
        A list of parametrization instances to compose together. The output dimension of each
        parametrization must match the input dimension of the next. The order
        of the list is :math:`[P_1, P_2, ..., P_n]` where :math:`P_1` is the first
        parametrization applied to the parameters.

    """

    __slots__ = ["_parametrizations"]

    def __init__(self, parametrizations: List[Parametrization]):
        if not isinstance(parametrizations, (list, tuple)):
            raise TypeError(
                f"parametrizations must be a list or tuple of Parametrization instances, got {type(parametrizations)}"
            )
        parametrizations = list(parametrizations)  # Convert to list if it's a tuple
        if len(parametrizations) == 0:
            raise ValueError("parametrizations list cannot be empty.")
        for i, p in enumerate(parametrizations):
            if not isinstance(p, Parametrization):
                raise TypeError(
                    f"All elements in parametrizations must be instances of Parametrization, got {type(p)} at index {i}"
                )
            if i > 0 and p.input_dim != parametrizations[i - 1].output_dim:
                raise ValueError(
                    f"Output dimension of parametrization at index {i-1} ({parametrizations[i - 1].output_dim}) "
                    f"must match input dimension of parametrization at index {i} ({p.input_dim})."
                )

        self._parametrizations = parametrizations

        # Capability flags derived from the parametrizations
        self._has_encode = all(p.has_encode for p in parametrizations)
        self._has_encode_dp = all(p.has_encode_dp for p in parametrizations)
        self._has_decode = all(p.has_decode for p in parametrizations)
        self._has_decode_dp = all(p.has_decode_dp for p in parametrizations)
        self._is_left_invertible = all(p.is_left_invertible for p in parametrizations)
        self._is_right_invertible = all(p.is_right_invertible for p in parametrizations)

        super().__init__()

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the composition, i.e. the input
        dimension of the first parametrization :math:`P_1`.

        Returns
        -------
        int
            The input dimension of the composition.

        """
        return self._parametrizations[0].input_dim

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the composition, i.e. the output
        dimension of the last parametrization :math:`P_n`.

        Returns
        -------
        int
            The output dimension of the composition.

        """
        return self._parametrizations[-1].output_dim

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "parametrizations": [p.to_dict() for p in self._parametrizations],
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> ComposedParametrization:
        if not "parametrizations" in data:
            raise ValueError("Input dictionary must contain a 'parametrizations' key.")
        parametrizations_data = data["parametrizations"]
        if not isinstance(parametrizations_data, (list, tuple)):
            raise TypeError(
                f"'parametrizations' key in input dictionary must be a list or tuple, got {type(parametrizations_data)}"
            )
        parametrizations = [
            Parametrization.from_dict(p_data) for p_data in parametrizations_data
        ]
        return cls(parametrizations=parametrizations)

    def _copy(self) -> ComposedParametrization:
        return self.__class__(
            parametrizations=[p.copy() for p in self._parametrizations]
        )

    @property
    def parametrizations(self) -> List[Parametrization]:
        r"""
        [Get] the list of parametrizations that are composed together.

        Returns
        -------
        List[Parametrization]
            The list of parametrization instances that make up the composition.
        """
        return list(
            self._parametrizations
        )  # Return a copy to prevent external modification

    @property
    def n_parametrizations(self) -> int:
        r"""
        [Get] the number of parametrizations in the composition.

        Returns
        -------
        int
            The number of parametrizations in the composition.
        """
        return len(self._parametrizations)

    # ------
    # Lock
    # ------
    @property
    def is_locked(self) -> bool:
        r"""
        [Get] Whether the composition or at least one of its parametrizations is locked.

        .. note::

            Use :meth:`is_locked_composition` to get the lock state of the composition
            alone.

        Returns
        -------
        bool
            True if the composition or one of its parametrizations is locked.

        """
        return self.is_locked_composition or any(
            p.is_locked for p in self._parametrizations
        )

    @property
    def is_locked_composition(self) -> bool:
        r"""
        [Get] Whether the composition itself is locked (without looking at its
        parametrizations).

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
        message of the first locked parametrization.

        Returns
        -------
        Optional[str]
            The message of the lock, or None if nothing is locked or the lock has no
            message.

        """
        if self.is_locked_composition:
            return LockableMixin.locked_message.fget(self)
        for p in self._parametrizations:
            if p.is_locked:
                return p.locked_message
        return None

    def _set_locked(self, locked: bool = True, *, message: Optional[str] = None) -> None:
        r"""
        Lock or unlock the composition and all its parametrizations.

        The lock is propagated to the parametrizations (recursively for nested
        compositions) with the same ``message``.

        .. important::

            Internal use only (see :class:`pysdic.core.lockable.LockableMixin`).

        .. warning::

            The parametrizations are shared, not copied. Unlocking the composition also
            unlocks its parametrizations, even if they were locked by another owner: only
            unlock new objects (as ``copy`` does).

        Parameters
        ----------
        locked : bool, optional (default: ``True``)
            True to lock the composition and its parametrizations, False to unlock them.

        message : Optional[str], optional (default: ``None``)
            The reason of the lock, added to the errors raised by the locked objects.
            Ignored when ``locked`` is False.

        Raises
        ------
        TypeError
            If ``locked`` is not a boolean or ``message`` is not a string.

        ValueError
            If ``message`` is an empty string.

        """
        super()._set_locked(locked, message=message)  # validates the inputs first
        for p in self._parametrizations:
            p._set_locked(locked, message=message)

    def _check_unlocked(self, action: str, *, override_lock: bool = False) -> None:
        # Check the composition, then each parametrization (recursively): the error names
        # the locked object and gives its own message
        super()._check_unlocked(action, override_lock=override_lock)
        for p in self._parametrizations:
            p._check_unlocked(action, override_lock=override_lock)

    def _encode(
        self,
        p_i: numpy.ndarray,
        *,
        dp: bool = False,
        kwargs: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Encode the parameters through the sequence of parametrizations.

        .. math::

            \vec{p_o} = P_n(P_{n-1}(...P_1(\vec{p_i})...))

        The jacobian with respect to the parameters is computed using
        the chain rule across the sequence of parametrizations:

        .. math::

            \mathbf{J}_p = \frac{\partial P_n}{\partial \vec{q}_{n-1}} \cdot ... \cdot \frac{\partial P_2}{\partial \vec{q}_1} \cdot \frac{\partial P_1}{\partial \vec{p_i}}

        where :math:`\vec{q}_k = P_k(...P_1(\vec{p_i})...)` are the intermediate parameters.

        Parameters
        ----------
        p_i : numpy.ndarray
            The parameters to encode, with shape :math:`(n_i,)` where :math:`n_i`
            is the input dimension of the first parametrization in the composition.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters during encoding.
            (ie., :math:`\mathbf{J}_p = \frac{\partial P_n \circ ... \circ P_1}{\partial \vec{p_i}}`).

        kwargs : Optional[List[Optional[Dict[str, Any]]]], optional (default: ``None``)
            A list of dictionaries containing additional keyword arguments to pass to each
            parametrization's encode method. The list must have the same length as the number
            of parametrizations in the composition, in the order :math:`[P_1, P_2, ..., P_n]`.
            ``None`` elements are replaced by empty dictionaries.


        Returns
        -------
        numpy.ndarray
            The encoded parameters after applying the sequence of parametrizations, with shape :math:`(n_o,)` where :math:`n_o`
            is the output dimension of the last parametrization in the composition.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the input parameters if ``dp=True``, with shape
            :math:`(n_o, n_i)`, or None if ``dp=False``.

        """
        kwargs = _validate_kwargs_sequence(kwargs, self.n_parametrizations, "kwargs")

        p = p_i
        jacobian_dp = None

        for parametrization, kwarg in zip(self._parametrizations, kwargs):
            result = parametrization.encode(p, dp=dp, **kwarg)
            p = result.p
            if dp:
                jacobian_dp = (
                    result.jacobian_dp
                    if jacobian_dp is None
                    else result.jacobian_dp @ jacobian_dp
                )  # (n_out_k, n_in_k) @ (n_in_k, n_i) -> (n_out_k, n_i)

        return p, jacobian_dp

    def _decode(
        self,
        p_o: numpy.ndarray,
        *,
        dp: bool = False,
        kwargs: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Decode the encoded parameters through the sequence of parametrizations in reverse.

        .. math::

            \vec{p_i} = P_1^{-1}(P_2^{-1}(...P_n^{-1}(\vec{p_o})...))

        The jacobian with respect to the encoded parameters is computed using
        the chain rule across the sequence of parametrizations in reverse:

        .. math::

            \mathbf{J}_p = \frac{\partial P_1^{-1}}{\partial \vec{r}_1} \cdot ... \cdot \frac{\partial P_{n-1}^{-1}}{\partial \vec{r}_{n-1}} \cdot \frac{\partial P_n^{-1}}{\partial \vec{p_o}}

        where :math:`\vec{r}_k = P_{k+1}^{-1}(...P_n^{-1}(\vec{p_o})...)` are the intermediate parameters.

        Parameters
        ----------
        p_o : numpy.ndarray
            The encoded parameters to decode, with shape :math:`(n_o,)` where :math:`n_o`
            is the output dimension of the last parametrization in the composition.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters during decoding.
            (ie., :math:`\mathbf{J}_p = \frac{\partial P_1^{-1} \circ ... \circ P_n^{-1}}{\partial \vec{p_o}}`).

        kwargs : Optional[List[Optional[Dict[str, Any]]]], optional (default: ``None``)
            A list of dictionaries containing additional keyword arguments to pass to each
            parametrization's decode method. The list must have the same length as the number
            of parametrizations in the composition, in the order :math:`[P_1, P_2, ..., P_n]`
            (the same order as for encoding, even if the decoding is applied in reverse).
            ``None`` elements are replaced by empty dictionaries.


        Returns
        -------
        numpy.ndarray
            The decoded parameters after applying the sequence of decoding maps, with shape :math:`(n_i,)`
            where :math:`n_i` is the input dimension of the first parametrization in the composition.

        Optional[numpy.ndarray]
            The Jacobian matrix with respect to the parameters if ``dp=True``, with shape
            :math:`(n_i, n_o)`, or None if ``dp=False``.

        """
        kwargs = _validate_kwargs_sequence(kwargs, self.n_parametrizations, "kwargs")

        p = p_o
        jacobian_dp = None

        for parametrization, kwarg in zip(
            reversed(self._parametrizations), reversed(kwargs)
        ):
            result = parametrization.decode(p, dp=dp, **kwarg)
            p = result.p
            if dp:
                jacobian_dp = (
                    result.jacobian_dp
                    if jacobian_dp is None
                    else result.jacobian_dp @ jacobian_dp
                )  # (n_in_k, n_out_k) @ (n_out_k, n_o) -> (n_in_k, n_o)

        return p, jacobian_dp