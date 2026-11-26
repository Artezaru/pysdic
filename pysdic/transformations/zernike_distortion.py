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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from numbers import Integral, Real
from numpy.typing import ArrayLike

import hashlib
import warnings
import numpy

from .transformation import Transformation
from .._optionals import import_optional
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _validate_choice,
    _validate_kwargs,
    _as_array,
)

if TYPE_CHECKING:
    import pyzernike


_SET_HINT = "or use 'set_zero' to initialize them with zeros"


def _n_polynomials(order: int) -> int:
    # Number of Zernike polynomials up to the order N_zer (OSA/ANSI indices j < (N+1)(N+2)/2)
    return (order + 1) * (order + 2) // 2


class _CacheEntry(object):
    # One entry of a ZernikePolynomialsCache: the polynomials (and derivatives) at a set of points
    __slots__ = ["ellipse", "digest", "points", "order", "derivatives", "polynomials"]

    def __init__(self, ellipse, digest, points, order, derivatives, polynomials) -> None:
        self.ellipse = ellipse  # (Rx, Ry, x0, y0)
        self.digest = digest  # Hash of the points
        self.points = points  # Read-only copy of the points (n_points, 2), for the exact comparison
        self.order = order  # Maximum order N_zer of the polynomials
        self.derivatives = derivatives  # Whether d/dx and d/dy are stored
        self.polynomials = polynomials  # Read-only (D, P, n_points), D = 3 with derivatives (value, d/dx, d/dy), 1 otherwise

    @property
    def nbytes(self) -> int:
        return self.points.nbytes + self.polynomials.nbytes


class ZernikePolynomialsCache(object):
    r"""
    A cache of the values (and derivatives) of the Zernike polynomials at sets of points, to
    share between :class:`pysdic.ZernikeDistortion` objects (see their ``polynomials_cache``).

    The Zernike polynomials :math:`Z_{n,m}(\rho, \theta)` only depend on the points and on the
    ellipse of definition :math:`(R_x, R_y, x_0, y_0)`, not on the coefficients of the
    distortion. When a distortion is applied several times to the same points (e.g. the
    iterations of a calibration where only the distortion is optimized), the polynomials are
    computed once and reused. Several distortions can share the same cache, for example the
    distortions of different orders evaluated at the same points.

    An entry is identified by the points (hash and exact comparison with a copy of the points)
    and the ellipse (exact comparison of :math:`R_x, R_y, x_0, y_0`):

    - The polynomials are stored up to the highest order computed: an entry of order
      :math:`N` also serves the orders :math:`N' \leq N` (the first polynomials in the
      OSA/ANSI order).
    - The derivatives :math:`\partial Z / \partial x` and :math:`\partial Z / \partial y` are
      stored only when they are requested (Jacobian ``dx`` of the distortion): an entry
      without derivatives is computed again (and replaced) when they are needed.
    - The entries are kept in the order of their last use: when the cache is full, the least
      recently used entry is removed.

    .. note::

        The cache is only used by :meth:`pysdic.ZernikeDistortion.apply`. The undistortion
        (``invert``) evaluates the polynomials at points that change at each Newton iteration:
        it never uses the cache.

    .. important::

        The memory of an entry is about :math:`8 N_p (2 + D P)` bytes for :math:`N_p` points and
        :math:`P` polynomials, with :math:`D = 1` (values) or :math:`D = 3` (with derivatives),
        e.g. about 115 MB for 627 264 points and :math:`P = 21` (order 5) without derivatives.
        Call :meth:`clear` (or remove the cache from the distortions) to free it.

    Parameters
    ----------
    max_entries : Integral, optional (default: ``2``)
        The maximum number of entries (sets of points) kept, at least 1. For a stereo pair whose
        two views share the same distortion, use at least 2 (one entry per view).

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import ZernikeDistortion, ZernikePolynomialsCache

        cache = ZernikePolynomialsCache(max_entries=2)
        distortion = ZernikeDistortion(n_zernike=5, radius_x=1.5, polynomials_cache=cache)
        distortion.set_zero()

        distortion.apply(points)  # The polynomials are computed and stored
        distortion.parameters = new_parameters
        distortion.apply(points)  # The polynomials are reused

        distortion.polynomials_cache = None  # No cache anymore
        cache.clear()  # Free the memory

    """

    __slots__ = ["_max_entries", "_entries", "_hits", "_misses"]

    def __init__(self, max_entries: Integral = 2) -> None:
        self._max_entries = _validate_integer(max_entries, "max_entries", min_value=1)
        self._entries: List[_CacheEntry] = []  # Most recently used first
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ Properties
    @property
    def max_entries(self) -> int:
        r"""
        [Get/Set] The maximum number of entries (at least 1). Reducing it removes the least
        recently used entries.

        Returns
        -------
        int
            The maximum number of entries.
        """
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: Integral) -> None:
        self._max_entries = _validate_integer(value, "max_entries", min_value=1)
        del self._entries[self._max_entries :]

    @property
    def n_entries(self) -> int:
        r"""
        [Get] The number of entries in the cache.

        Returns
        -------
        int
            The number of stored sets of points.
        """
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        r"""
        [Get] The memory used by the entries, in bytes.

        Returns
        -------
        int
            The number of bytes of the stored points and polynomials.
        """
        return sum(entry.nbytes for entry in self._entries)

    @property
    def hits(self) -> int:
        r"""
        [Get] The number of evaluations served by the cache since its creation (or the last
        :meth:`clear`).

        Returns
        -------
        int
            The number of hits.
        """
        return self._hits

    @property
    def misses(self) -> int:
        r"""
        [Get] The number of evaluations that computed the polynomials since its creation (or
        the last :meth:`clear`).

        Returns
        -------
        int
            The number of misses.
        """
        return self._misses

    def clear(self) -> None:
        r"""
        Remove all the entries (and reset the numbers of hits and misses).
        """
        self._entries = []
        self._hits = 0
        self._misses = 0

    def __repr__(self) -> str:
        return (
            f"ZernikePolynomialsCache(max_entries={self._max_entries}, n_entries={self.n_entries}, "
            f"nbytes={self.nbytes}, hits={self._hits}, misses={self._misses})"
        )

    # ------------------------------------------------------------------ Evaluation
    def evaluate(
        self,
        points: numpy.ndarray,
        order: int,
        radius_x: float,
        radius_y: float,
        center: numpy.ndarray,
        derivatives: bool,
    ) -> numpy.ndarray:
        r"""
        Get the Zernike polynomials up to ``order`` at the points, from the cache or computed
        with ``pyzernike`` (and stored).

        .. important::

            Internal use (called by :class:`pysdic.ZernikeDistortion`).

        Parameters
        ----------
        points : numpy.ndarray
            The points :math:`(x, y)`, shape :math:`(N_p, 2)`.

        order : int
            The maximum order :math:`N_{zer}` of the polynomials.

        radius_x, radius_y : float
            The radii of the ellipse of definition.

        center : numpy.ndarray
            The center :math:`(x_0, y_0)` of the ellipse of definition.

        derivatives : bool
            Whether the derivatives :math:`\partial Z / \partial x` and
            :math:`\partial Z / \partial y` are needed.

        Returns
        -------
        numpy.ndarray
            A read-only array of shape :math:`(D, P, N_p)` with :math:`P` the number of
            polynomials up to ``order``: :math:`D = 3` (values, d/dx, d/dy) if ``derivatives``,
            :math:`D = 1` (values) otherwise. NaN outside the ellipse.
        """
        points = numpy.ascontiguousarray(points, dtype=numpy.float64)
        ellipse = (float(radius_x), float(radius_y), float(center[0]), float(center[1]))
        digest = hashlib.blake2b(points, digest_size=16).digest()
        P = _n_polynomials(order)
        D = 3 if derivatives else 1

        for k, entry in enumerate(self._entries):
            if entry.ellipse != ellipse or entry.digest != digest or not numpy.array_equal(entry.points, points):
                continue
            if entry.order >= order and (entry.derivatives or not derivatives):
                self._entries.insert(0, self._entries.pop(k))  # Most recently used
                self._hits += 1
                return entry.polynomials[:D, :P]
            # Same points: computed again with the highest order and the derivatives needed
            order = max(order, entry.order)
            derivatives = derivatives or entry.derivatives
            del self._entries[k]
            break

        self._misses += 1
        polynomials = _compute_polynomials(points, order, ellipse, derivatives)
        stored_points = points.copy()
        for array in (stored_points, polynomials):
            array.flags.writeable = False
        self._entries.insert(0, _CacheEntry(ellipse, digest, stored_points, order, derivatives, polynomials))
        del self._entries[self._max_entries :]
        return polynomials[:D, :P]


def _compute_polynomials(
    points: numpy.ndarray, order: int, ellipse: Tuple[float, float, float, float], derivatives: bool
) -> numpy.ndarray:
    # Zernike polynomials up to the order (and d/dx, d/dy) at the points, shape (D, P, n_points)
    pyzernike = import_optional("pyzernike", "ZernikeDistortion")  # Cached (required by the class)
    radius_x, radius_y, x0, y0 = ellipse
    return numpy.asarray(
        pyzernike.xy_zernike_polynomial_up_to_order(
            points[:, 0],
            points[:, 1],
            order=order,
            Rx=radius_x,
            Ry=radius_y,
            x0=x0,
            y0=y0,
            x_derivative=[0, 1, 0] if derivatives else [0],
            y_derivative=[0, 0, 1] if derivatives else [0],
            default=numpy.nan,
            to_list=False,
        ),
        dtype=numpy.float64,
    )


class ZernikeDistortion(Transformation):
    r"""
    Subclass of :class:`pysdic.Transformation` that represents a distortion transformation
    based on Zernike polynomials.

    This class represents a **distortion transformation**, which is typically the
    middle step in the camera projection pipeline:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Let :math:`\vec{x}_n = (x_n, y_n)` be a **normalized point**.
    The corresponding **distorted point** :math:`\vec{x}_d = (x_d, y_d)` is obtained
    by adding a displacement field expanded on the Zernike polynomials :math:`Z_{n,m}`:

    .. math::

        x_{d} = x_{n} + \sum_{n=0}^{N_{zer}} \sum_{m} C^{x}_{n,m} Z_{n,m}(\rho, \theta)

    .. math::

        y_{d} = y_{n} + \sum_{n=0}^{N_{zer}} \sum_{m} C^{y}_{n,m} Z_{n,m}(\rho, \theta)

    where :math:`m \in [-n, n]` with :math:`n - m` even, :math:`C^{x}_{n,m}` and :math:`C^{y}_{n,m}`
    are the Zernike coefficients for the x and y coordinates, and :math:`\rho` and :math:`\theta`
    are the polar coordinates of the normalized points in the ellipse of definition with
    radii :math:`(R_x, R_y)` and center :math:`(x_0, y_0)`:

    .. math::

        \rho = \sqrt{\left(\frac{x_{n} - x_{0}}{R_x}\right)^2 + \left(\frac{y_{n} - y_{0}}{R_y}\right)^2},
        \qquad
        \theta = \arctan2\left(\frac{y_{n} - y_{0}}{R_y}, \frac{x_{n} - x_{0}}{R_x}\right)

    .. note::

        The Zernike polynomials are only defined inside the ellipse (:math:`\rho \leq 1`).
        The points outside the ellipse are distorted to NaN, so the ellipse must contain the
        whole field of view of the camera.

    .. seealso::

        Package ``pyzernike`` (https://github.com/Artezaru/pyzernike) for the
        implementation of the Zernike polynomials and their derivatives.

    .. note::

        This class requires ``pyzernike``: an ``ImportError`` is raised at the instantiation
        if it is missing (``ZernikeDistortion.is_available()`` returns False).

    The transformation is parameterized by the Zernike coefficients :math:`C^{x}_{n,m}` and
    :math:`C^{y}_{n,m}` for :math:`n \leq N_{zer}`, and depends on 4 **constants**: the radii
    :math:`(R_x, R_y)` and the center :math:`(x_0, y_0)` of the ellipse.

    The coefficients are stored in the ``parameters`` with the OSA/ANSI standard indices
    :math:`j = \frac{n(n+2) + m}{2}`, alternating the x and y coordinates:

    .. code-block:: text

        parameters[2j]     = C^x_{n,m}
        parameters[2j + 1] = C^y_{n,m}

    i.e. :math:`(C^x_{0,0}, C^y_{0,0}, C^x_{1,-1}, C^y_{1,-1}, C^x_{1,1}, C^y_{1,1}, C^x_{2,-2}, \dots)`.
    The number of parameters is:

    .. math::

        N_{params} = (N_{zer}+1)(N_{zer}+2)

    +--------------------------------+---------------------------------+-----------------------------------------+
    | Order ``n_zernike``            | Parameters for X or Y           | Parameters of the model ``n_parameters``|
    +================================+=================================+=========================================+
    | None                           | 0                               | 0                                       |
    +--------------------------------+---------------------------------+-----------------------------------------+
    | 0                              | 1                               | 2                                       |
    +--------------------------------+---------------------------------+-----------------------------------------+
    | 1                              | 3                               | 6                                       |
    +--------------------------------+---------------------------------+-----------------------------------------+
    | 2                              | 6                               | 12                                      |
    +--------------------------------+---------------------------------+-----------------------------------------+
    | 3                              | 10                              | 20                                      |
    +--------------------------------+---------------------------------+-----------------------------------------+
    | 4                              | 15                              | 30                                      |
    +--------------------------------+---------------------------------+-----------------------------------------+

    The inverse transformation (undistortion) has no closed form and is computed with
    Newton iterations (see :meth:`_invert`). It is available with its Jacobians, and it is
    both a left and a right inverse of the distortion in the domain where the distortion is
    bijective (i.e. for reasonable distortions and points inside the ellipse).

    .. important::

        During instantiation:

        - If both ``parameters`` and (``n_parameters`` or ``n_zernike``) are provided, the length of
          ``parameters`` must match ``n_parameters`` (see table above).
        - If only ``n_parameters`` or ``n_zernike`` is provided, the parameters are not set and must
          be set before use (see :meth:`set_zero`).
        - If only ``parameters`` is provided, the number of parameters is
          inferred from the length of the array.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        Distortion parameters of shape ``(n_parameters,)`` ordered as described above.
        If None, the parameters can be set later.

    n_parameters : Optional[Integral], optional (default: ``None``)
        Number of parameters of the distortion model, in ``[0, 2, 6, 12, 20, 30, ...]``.
        If not specified, it is inferred from ``n_zernike`` or from the length of ``parameters``.

    n_zernike : Optional[Integral], optional (default: ``None``)
        Maximum order :math:`N_{zer}` of the Zernike polynomials. If not specified, it is inferred
        from ``n_parameters`` or from the length of ``parameters``.

    radius_x : Real, optional (default: ``1.0``)
        The radius :math:`R_x` of the ellipse of definition along x (strictly positive).

    radius_y : Optional[Real], optional (default: ``None``)
        The radius :math:`R_y` of the ellipse of definition along y (strictly positive).
        If None, it is equal to ``radius_x`` (circle).

    center : ArrayLike, optional (default: ``(0.0, 0.0)``)
        The center :math:`(x_0, y_0)` of the ellipse of definition.

    polynomials_cache : Optional[ZernikePolynomialsCache], optional (default: ``None``)
        A cache of the Zernike polynomials used by :meth:`apply`, possibly shared with other
        distortions (see :class:`ZernikePolynomialsCache`). By default, no cache: the
        polynomials are computed at each call. It can be set or removed (``None``) at any time
        with the property ``polynomials_cache``. It is not serialized and not copied, and it
        never changes the results.


    Examples
    --------
    A Zernike distortion up to the order 2 defined on the disk of radius 1.5:

    .. code-block:: python

        distortion = ZernikeDistortion(n_zernike=2, radius_x=1.5)
        distortion.set_zero()
        distortion.set_Cx(2, 0, 0.01)   # Defocus-like term along x

        distorted = distortion.apply([[0.1, 0.2]]).x
        distortion.invert(distorted).x    # array([[0.1, 0.2]])

    """

    _required_optionals = ("pyzernike",)  # ImportError at the instantiation if missing

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
        *,
        n_parameters: Optional[Integral] = None,
        n_zernike: Optional[Integral] = None,
        radius_x: Real = 1.0,
        radius_y: Optional[Real] = None,
        center: ArrayLike = (0.0, 0.0),
        polynomials_cache: Optional[ZernikePolynomialsCache] = None,
    ) -> None:
        self.polynomials_cache = polynomials_cache  # Validated by the setter

        # Number of parameters
        n_zernike = _validate_integer(n_zernike, "n_zernike", min_value=0, allow_none=True)
        n_parameters = _validate_integer(
            n_parameters, "n_parameters", min_value=0, allow_none=True
        )
        if n_zernike is not None:
            n_from_order = (n_zernike + 1) * (n_zernike + 2)
            if n_parameters is not None and n_parameters != n_from_order:
                raise ValueError(
                    f"n_parameters ({n_parameters}) and n_zernike ({n_zernike}) are not consistent: "
                    f"expected n_parameters = {n_from_order}."
                )
            n_parameters = n_from_order
        if n_parameters is None:
            if parameters is None:
                raise ValueError(
                    "At least one of parameters, n_parameters or n_zernike must be provided."
                )
            # Only to infer the number of parameters: they are validated by set_parameters
            n_parameters = _as_array(
                parameters, "parameters", numpy.float64,
                shape=(None,), shape_hint="(n_parameters,)", finite=True,
            ).shape[0]
        self._n_zernike = self._order_from_n_parameters(n_parameters)
        self._n_distortion_parameters = n_parameters

        # Constants of the ellipse of definition
        radius_x = _validate_real(radius_x, "radius_x", positive=True, finite=True)
        if radius_y is None:
            radius_y = radius_x
        radius_y = _validate_real(radius_y, "radius_y", positive=True, finite=True)
        center = _as_array(center, "center", numpy.float64, shape=(2,), finite=True, copy=True)
        center.flags.writeable = False
        self._radius_x = radius_x
        self._radius_y = radius_y
        self._center = center

        super().__init__(parameters=parameters)

    @staticmethod
    def _order_from_n_parameters(n_parameters: Integral) -> Optional[int]:
        # Return the order N_zer such that n_parameters = (N_zer + 1)(N_zer + 2), or None for 0
        n_parameters = _validate_integer(n_parameters, "n_parameters", min_value=0)
        if n_parameters == 0:
            return None
        order = 0
        while (order + 1) * (order + 2) < n_parameters:
            order += 1
        if (order + 1) * (order + 2) != n_parameters:
            raise ValueError(
                f"n_parameters must be in [0, 2, 6, 12, 20, 30, ...], got {n_parameters}."
            )
        return order

    # =============================================
    # Properties inherited from Transformation Class
    # =============================================
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters of the distortion model :math:`N_{params} = (N_{zer}+1)(N_{zer}+2)`.

        Returns
        -------
        int
            The number of parameters of the distortion model.

        """
        return self._n_distortion_parameters

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the distortion (2D normalized points).

        Returns
        -------
        int
            The input dimension of the distortion.

        """
        return 2

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the distortion (2D distorted points).

        Returns
        -------
        int
            The output dimension of the distortion.

        """
        return 2

    # =============================================
    # I/O Methods for Transformation Class
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "n_parameters": self._n_distortion_parameters,
            "radius_x": self._radius_x,
            "radius_y": self._radius_y,
            "center": self._center.tolist(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> ZernikeDistortion:
        return cls(
            parameters=data.get("parameters", None),
            n_parameters=data.get("n_parameters", None),
            radius_x=data.get("radius_x", 1.0),
            radius_y=data.get("radius_y", None),
            center=data.get("center", (0.0, 0.0)),
        )

    def _copy(self) -> ZernikeDistortion:
        return self.__class__(
            parameters=self.parameters,
            n_parameters=self._n_distortion_parameters,
            radius_x=self._radius_x,
            radius_y=self._radius_y,
            center=self._center,
        )

    # =============================================
    # Constants
    # =============================================
    @property
    def n_zernike(self) -> Optional[int]:
        r"""
        [Get] the maximum order :math:`N_{zer}` of the Zernike polynomials, such that
        :math:`N_{params} = (N_{zer}+1)(N_{zer}+2)`.

        Returns
        -------
        Optional[int]
            The maximum order of the Zernike polynomials, or None if the model has no parameters.
        """
        return self._n_zernike

    @property
    def radius_x(self) -> float:
        r"""
        [Get] the radius :math:`R_x` of the ellipse of definition along x.

        .. note::

            An alias for ``radius_x`` is ``rx``.

        Returns
        -------
        float
            The radius of the ellipse along x.
        """
        return self._radius_x

    @property
    def rx(self) -> float:
        return self.radius_x

    @property
    def radius_y(self) -> float:
        r"""
        [Get] the radius :math:`R_y` of the ellipse of definition along y.

        .. note::

            An alias for ``radius_y`` is ``ry``.

        Returns
        -------
        float
            The radius of the ellipse along y.
        """
        return self._radius_y

    @property
    def ry(self) -> float:
        return self.radius_y

    @property
    def radius(self) -> float:
        r"""
        [Get] the radius :math:`R = R_x = R_y` of the disk of definition.

        Returns
        -------
        float
            The radius of the disk.

        Raises
        ------
        ValueError
            If the domain of definition is an ellipse (:math:`R_x \neq R_y`).
        """
        if self._radius_x != self._radius_y:
            raise ValueError(
                "The domain of definition is an ellipse (radius_x != radius_y), "
                "use radius_x and radius_y instead."
            )
        return self._radius_x

    @property
    def center(self) -> numpy.ndarray:
        r"""
        [Get] the center :math:`(x_0, y_0)` of the ellipse of definition.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The center of the ellipse with shape (2,).
        """
        return self._center

    @property
    def center_x(self) -> float:
        r"""
        [Get] the x coordinate :math:`x_0` of the center of the ellipse. An alias is ``x0``.

        Returns
        -------
        float
            The x coordinate of the center of the ellipse.
        """
        return float(self._center[0])

    @property
    def x0(self) -> float:
        return self.center_x

    @property
    def center_y(self) -> float:
        r"""
        [Get] the y coordinate :math:`y_0` of the center of the ellipse. An alias is ``y0``.

        Returns
        -------
        float
            The y coordinate of the center of the ellipse.
        """
        return float(self._center[1])

    @property
    def y0(self) -> float:
        return self.center_y

    # =============================================
    # Cache of the polynomials
    # =============================================
    @property
    def polynomials_cache(self) -> Optional[ZernikePolynomialsCache]:
        r"""
        [Get/Set] The cache of the Zernike polynomials used by :meth:`apply`, or None (default)
        for no cache (see :class:`ZernikePolynomialsCache`).

        The cache can be shared with other distortions, and set or removed (``None``) at any
        time, also when the parameters are locked: it never changes the results. It is not
        serialized and not copied (a copy has no cache).

        Returns
        -------
        Optional[ZernikePolynomialsCache]
            The cache, or None.
        """
        return self._polynomials_cache

    @polynomials_cache.setter
    def polynomials_cache(self, value: Optional[ZernikePolynomialsCache]) -> None:
        if value is not None and not isinstance(value, ZernikePolynomialsCache):
            raise TypeError(
                f"Input 'polynomials_cache' must be a ZernikePolynomialsCache or None. Got type {type(value)}."
            )
        self._polynomials_cache = value

    # =============================================
    # Parameters
    # =============================================
    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the distortion model :
        ``["Cx(0, 0)", "Cy(0, 0)", "Cx(1, -1)", "Cy(1, -1)", "Cx(1, 1)", "Cy(1, 1)", ...]``.

        Returns
        -------
        List[str]
            The names of the parameters of the distortion model.
        """
        return [
            name
            for pair in zip(self.parameter_x_names, self.parameter_y_names)
            for name in pair
        ]

    @property
    def parameter_x_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters along the x-axis : ``["Cx(0, 0)", "Cx(1, -1)", "Cx(1, 1)", ...]``.

        Returns
        -------
        List[str]
            The names of the parameters along the x-axis.
        """
        return [f"Cx({n}, {m})" for n, m in self._orders()]

    @property
    def parameter_y_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters along the y-axis : ``["Cy(0, 0)", "Cy(1, -1)", "Cy(1, 1)", ...]``.

        Returns
        -------
        List[str]
            The names of the parameters along the y-axis.
        """
        return [f"Cy({n}, {m})" for n, m in self._orders()]

    def _orders(self) -> List[Tuple[int, int]]:
        # Orders (n, m) of the Zernike polynomials of the model in the OSA/ANSI order
        if self._n_zernike is None:
            return []
        return [(n, m) for n in range(self._n_zernike + 1) for m in range(-n, n + 1, 2)]

    def get_index(self, n: Integral, m: Integral, coord: str) -> int:
        r"""
        Get the index in the ``parameters`` of the Zernike coefficient :math:`C^{x}_{n,m}` or
        :math:`C^{y}_{n,m}`.

        .. math::

            \text{index} = n(n+2) + m + (0 \text{ if } coord = 'x' \text{ else } 1)

        Parameters
        ----------
        n : Integral
            The order of the Zernike polynomial.

        m : Integral
            The azimuthal frequency of the Zernike polynomial (:math:`|m| \leq n`, :math:`n - m` even).

        coord : str
            The coordinate ('x' or 'y').


        Returns
        -------
        int
            The index of the Zernike coefficient in the parameters.

        Raises
        ------
        ValueError
            If :math:`(n, m)` is not a valid Zernike polynomial or is not in the model.
        """
        coord = _validate_choice(coord, "coord", ("x", "y"))
        n = _validate_integer(n, "n")
        m = _validate_integer(m, "m")
        if n < 0 or abs(m) > n or (n - m) % 2 != 0:
            raise ValueError(f"(n={n}, m={m}) is not a valid Zernike polynomial.")
        index = n * (n + 2) + m + (1 if coord == "y" else 0)
        if index >= self.n_parameters:
            raise ValueError(
                f"(n={n}, m={m}) is not in the model with n_zernike={self._n_zernike}."
            )
        return index

    def get_Cx(self, n: Integral, m: Integral) -> Optional[float]:
        r"""
        Get the Zernike coefficient :math:`C^{x}_{n,m}` for the x coordinate.

        Parameters
        ----------
        n : Integral
            The order of the Zernike polynomial.

        m : Integral
            The azimuthal frequency of the Zernike polynomial.


        Returns
        -------
        Optional[float]
            The value of the Zernike coefficient, or None if the parameters are not set.
        """
        index = self.get_index(n, m, "x")
        parameters = self.parameters
        return None if parameters is None else float(parameters[index])

    def set_Cx(self, n: Integral, m: Integral, value: Real) -> None:
        r"""
        Set the Zernike coefficient :math:`C^{x}_{n,m}` for the x coordinate.

        The parameters must already be set (see :meth:`set_zero`).

        Parameters
        ----------
        n : Integral
            The order of the Zernike polynomial.

        m : Integral
            The azimuthal frequency of the Zernike polynomial.

        value : Real
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, "x")
        self._set_parameters_at(index, value, f"Cx({n}, {m})", hint=_SET_HINT)

    def get_Cy(self, n: Integral, m: Integral) -> Optional[float]:
        r"""
        Get the Zernike coefficient :math:`C^{y}_{n,m}` for the y coordinate.

        Parameters
        ----------
        n : Integral
            The order of the Zernike polynomial.

        m : Integral
            The azimuthal frequency of the Zernike polynomial.


        Returns
        -------
        Optional[float]
            The value of the Zernike coefficient, or None if the parameters are not set.
        """
        index = self.get_index(n, m, "y")
        parameters = self.parameters
        return None if parameters is None else float(parameters[index])

    def set_Cy(self, n: Integral, m: Integral, value: Real) -> None:
        r"""
        Set the Zernike coefficient :math:`C^{y}_{n,m}` for the y coordinate.

        The parameters must already be set (see :meth:`set_zero`).

        Parameters
        ----------
        n : Integral
            The order of the Zernike polynomial.

        m : Integral
            The azimuthal frequency of the Zernike polynomial.

        value : Real
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, "y")
        self._set_parameters_at(index, value, f"Cy({n}, {m})", hint=_SET_HINT)

    @property
    def parameters_x(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the Zernike coefficients :math:`C^{x}_{n,m}` for the x coordinate in the OSA/ANSI
        order, with shape ``(n_parameters // 2,)``.

        The coefficients can only be set if the parameters are already set (see :meth:`set_zero`).

        Parameters
        ----------
        value : ArrayLike
            The Zernike coefficients for the x coordinate with shape ``(n_parameters // 2,)``.

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the x coordinate, or None if the parameters are not set.
        """
        parameters = self.parameters
        return None if parameters is None else parameters[0::2].copy()

    @parameters_x.setter
    def parameters_x(self, value: ArrayLike) -> None:
        self._set_coordinate_parameters(0, value, "parameters_x")

    @property
    def parameters_y(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the Zernike coefficients :math:`C^{y}_{n,m}` for the y coordinate in the OSA/ANSI
        order, with shape ``(n_parameters // 2,)``.

        The coefficients can only be set if the parameters are already set (see :meth:`set_zero`).

        Parameters
        ----------
        value : ArrayLike
            The Zernike coefficients for the y coordinate with shape ``(n_parameters // 2,)``.

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the y coordinate, or None if the parameters are not set.
        """
        parameters = self.parameters
        return None if parameters is None else parameters[1::2].copy()

    @parameters_y.setter
    def parameters_y(self, value: ArrayLike) -> None:
        self._set_coordinate_parameters(1, value, "parameters_y")

    def _set_coordinate_parameters(self, start: int, value: ArrayLike, name: str) -> None:
        # Set the coefficients of one coordinate (start=0 for x, 1 for y), the lock is respected
        self._set_parameters_at(slice(start, None, 2), value, name, hint=_SET_HINT)

    def set_zero(self) -> None:
        r"""
        Set all the Zernike coefficients to zero (no distortion).
        """
        self.parameters = numpy.zeros(self.n_parameters, dtype=float)

    # =============================================
    # Transform methods
    # =============================================
    def _distort(
        self, points: numpy.ndarray, dx: bool, dp: bool, use_cache: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        # Internal implementation of the distortion (see _apply for the equations). With
        # use_cache=True (only from _apply), the polynomials come from the polynomials_cache if set.
        n_points = points.shape[0]
        distorted = points.copy()
        jacobian_dx = numpy.tile(numpy.eye(2), (n_points, 1, 1)) if dx else None
        jacobian_dp = numpy.zeros((n_points, 2, self.n_parameters)) if dp else None

        if self._n_zernike is None:
            return distorted, jacobian_dx, jacobian_dp

        # Zernike polynomials and their derivatives with shape (D, P, n_points)
        ellipse = (self._radius_x, self._radius_y, float(self._center[0]), float(self._center[1]))
        if use_cache and self._polynomials_cache is not None:
            zernike = self._polynomials_cache.evaluate(
                points, self._n_zernike, self._radius_x, self._radius_y, self._center, dx
            )
        else:
            zernike = _compute_polynomials(points, self._n_zernike, ellipse, dx)
        coefficients = self._parameters.reshape(-1, 2)  # (P, 2) with columns (Cx, Cy)

        values = zernike[0].T  # (n_points, P)
        distorted += values @ coefficients  # (n_points, 2)

        if dx:
            jacobian_dx[:, :, 0] += zernike[1].T @ coefficients  # d/dx_n
            jacobian_dx[:, :, 1] += zernike[2].T @ coefficients  # d/dy_n

        if dp:
            jacobian_dp[:, 0, 0::2] = values  # d x_d / d Cx
            jacobian_dp[:, 1, 1::2] = values  # d y_d / d Cy

        return distorted, jacobian_dx, jacobian_dp

    def _apply(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the **normalized points** to the **distorted points**.

        .. math::

            \vec{x}_d = \vec{x}_n + \sum_{n, m} \vec{C}_{n,m} Z_{n,m}(\rho, \theta)
            \qquad \text{with} \qquad
            \vec{C}_{n,m} = \begin{bmatrix} C^{x}_{n,m} \\ C^{y}_{n,m} \end{bmatrix}

        The Jacobian with respect to the normalized points is:

        .. math::

            \frac{\partial \vec{x}_d}{\partial \vec{x}_n} = I_2 + \sum_{n, m} \vec{C}_{n,m}
            \begin{bmatrix}
            \frac{\partial Z_{n,m}}{\partial x_n} & \frac{\partial Z_{n,m}}{\partial y_n}
            \end{bmatrix}

        where the derivatives of the Zernike polynomials with respect to the cartesian coordinates
        are given by ``pyzernike``. The Jacobian with respect to the coefficients is:

        .. math::

            \frac{\partial x_d}{\partial C^{x}_{n,m}} = \frac{\partial y_d}{\partial C^{y}_{n,m}} = Z_{n,m}(\rho, \theta),
            \qquad
            \frac{\partial x_d}{\partial C^{y}_{n,m}} = \frac{\partial y_d}{\partial C^{x}_{n,m}} = 0

        .. note::

            The points outside the ellipse of definition are distorted to NaN.

        .. note::

            With a ``polynomials_cache``, the polynomials at the same points (and the same
            ellipse) are computed once and reused (see :class:`ZernikePolynomialsCache`).


        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the normalized points is computed.

        dp : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distortion parameters is computed.


        Returns
        -------
        distorted_points : numpy.ndarray
            The distorted points. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the normalized points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the distortion parameters.
            Shape ``(n_points, 2, n_parameters)`` if ``dp=True``, otherwise ``None``.
        """
        return self._distort(normalized_points, dx, dp, use_cache=True)

    def _invert(
        self,
        distorted_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        max_iteration: int = 50,
        tolerance: float = 1e-14,
        pysolvegn: bool = False,
        solve_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the **distorted points** to the **normalized points**.

        The normalized points :math:`\vec{x}_n` are the solutions of :math:`D(\vec{x}_n) = \vec{x}_d`,
        where :math:`D` is the distortion. They are computed with Newton iterations, starting from
        :math:`\vec{x}_n^{(0)} = 2 \vec{x}_d - D(\vec{x}_d)`:

        .. math::

            \vec{x}_n^{(k+1)} = \vec{x}_n^{(k)} - \left(\frac{\partial D}{\partial \vec{x}_n}(\vec{x}_n^{(k)})\right)^{-1}
            \left( D(\vec{x}_n^{(k)}) - \vec{x}_d \right)

        until the update of every point is below ``tolerance`` (relatively to the magnitude of the point).

        The Jacobians are obtained by differentiating the identity :math:`D(D^{-1}(\vec{x}_d; \vec{p}); \vec{p}) = \vec{x}_d`:

        .. math::

            \frac{\partial D^{-1}}{\partial \vec{x}_d} = \left( \frac{\partial D}{\partial \vec{x}_n}(\vec{x}_n) \right)^{-1},
            \qquad
            \frac{\partial D^{-1}}{\partial \vec{p}} = - \frac{\partial D^{-1}}{\partial \vec{x}_d} \cdot \frac{\partial D}{\partial \vec{p}}(\vec{x}_n)

        To solve the undistortion with ``pysolvegn`` (for example with a Levenberg-Marquardt damping),
        set ``pysolvegn=True`` and give the options of the solver with ``solve_kwargs``.
        In this case, the Jacobians are not available.

        .. note::

            A ``RuntimeWarning`` is emitted if some points do not converge within ``max_iteration``
            iterations, which usually means that the points are outside the ellipse of definition or
            outside the domain where the distortion is bijective.


        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distorted points is computed.

        dp : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distortion parameters is computed.

        max_iteration : int, optional (default: ``50``)
            The maximum number of Newton iterations.

        tolerance : float, optional (default: ``1e-14``)
            The relative tolerance on the update of the points to stop the iterations.

        pysolvegn : bool, optional (default: ``False``)
            If True, the undistortion is computed with :func:`pysdic.optimize_transformation_input_points`
            (``pysolvegn.solve_batch``) instead of the Newton iterations, starting from the same initial guess.
            In this case, the Jacobians are not available.

        solve_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Only with ``pysolvegn=True``: additional keyword arguments passed to ``pysolvegn.solve_batch``
            (``damping``, ``verbosity``, ...) or to :func:`pysdic.optimize_transformation_input_points`
            (``mode``, ``loss``, ``loss_scale``). By default, ``max_iteration=max_iteration`` and ``xtol=tolerance``.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the distorted points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the distortion parameters.
            Shape ``(n_points, 2, n_parameters)`` if ``dp=True``, otherwise ``None``.
        """
        pysolvegn = _validate_bool(pysolvegn, "pysolvegn")
        max_iteration = _validate_integer(max_iteration, "max_iteration", min_value=0)
        tolerance = _validate_real(tolerance, "tolerance", non_negative=True, finite=True)

        if pysolvegn:
            if dx or dp:
                raise ValueError(
                    "The Jacobians of the undistortion cannot be computed with pysolvegn."
                )
            guess = 2.0 * distorted_points - self._distort(distorted_points, False, False)[0]
            return (
                self._invert_pysolvegn(distorted_points, guess, max_iteration, tolerance, solve_kwargs),
                None,
                None,
            )

        normalized_points = 2.0 * distorted_points - self._distort(
            distorted_points, False, False
        )[0]  # Initial guess (n_points, 2)
        active = numpy.flatnonzero(numpy.all(numpy.isfinite(normalized_points), axis=1))
        for _ in range(max_iteration):
            if active.size == 0:
                break
            values, jacobian, _ = self._distort(normalized_points[active], True, False)
            residuals = values - distorted_points[active]  # (n_active, 2)
            valid = numpy.all(numpy.isfinite(residuals), axis=1)
            steps = numpy.zeros_like(residuals)
            steps[valid] = numpy.linalg.solve(jacobian[valid], residuals[valid][:, :, None])[:, :, 0]
            normalized_points[active] -= steps
            converged = valid & numpy.all(
                numpy.abs(steps) <= tolerance * (1.0 + numpy.abs(normalized_points[active])),
                axis=1,
            )
            normalized_points[active[~valid]] = numpy.nan  # Left the ellipse of definition
            active = active[~converged & valid]

        n_failed = active.size + int(numpy.sum(~numpy.all(numpy.isfinite(normalized_points), axis=1)))
        if n_failed > 0:
            warnings.warn(
                f"The undistortion did not converge for {n_failed} point(s). The points may be "
                f"outside the ellipse of definition or outside the domain where the distortion "
                f"is bijective.",
                RuntimeWarning,
            )

        jacobian_dx = None
        jacobian_dp = None
        if dx or dp:
            _, jacobian_apply_dx, jacobian_apply_dp = self._distort(normalized_points, True, dp)
            jacobian_inverse_dx = numpy.full((normalized_points.shape[0], 2, 2), numpy.nan)
            finite = numpy.all(numpy.isfinite(jacobian_apply_dx), axis=(1, 2))
            jacobian_inverse_dx[finite] = numpy.linalg.inv(jacobian_apply_dx[finite])
            if dx:
                jacobian_dx = jacobian_inverse_dx
            if dp:
                jacobian_dp = -jacobian_inverse_dx @ jacobian_apply_dp  # (n_points, 2, n_parameters)

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert_pysolvegn(
        self,
        distorted_points: numpy.ndarray,
        guess: numpy.ndarray,
        max_iteration: int,
        tolerance: float,
        solve_kwargs: Optional[Dict[str, Any]],
    ) -> numpy.ndarray:
        # Undistortion with pysolvegn: D(x_n) = x_d solved as an optimization of the input points of D
        from ..transformation_optimizations.optimize_transformation_input_points import (
            optimize_transformation_input_points,
        )

        solve_kwargs = {
            "max_iteration": max_iteration,
            "xtol": tolerance,
            **_validate_kwargs(solve_kwargs, "solve_kwargs"),
        }

        normalized_points, result = optimize_transformation_input_points(
            self, distorted_points, guess, **solve_kwargs
        )
        # The points outside the ellipse of definition are set to NaN, as for the Newton iterations
        outside = ~numpy.all(numpy.isfinite(self._distort(normalized_points, False, False)[0]), axis=1)
        normalized_points[outside] = numpy.nan
        n_failed = int(numpy.sum(~result.success))
        if n_failed > 0:
            warnings.warn(
                f"The undistortion with pysolvegn did not converge for {n_failed} point(s). The points may "
                f"be outside the ellipse of definition or outside the domain where the distortion is bijective.",
                RuntimeWarning,
            )
        return normalized_points