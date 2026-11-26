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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy
from numpy.typing import ArrayLike

from ..transformations.transformation import Transformation, TransformationResult
from ..transformations.no_transformation import NoExtrinsic
from .._validation import _validate_bool, _validate_kwargs, _as_batch_array


# Names of the 4 steps (intrinsic, distortion, R, P), used in the error messages
_STEP_NAMES = ("intrinsic", "distortion", "R", "P")


@dataclass(slots=True, frozen=True)
class DistortionResult(TransformationResult):
    r"""
    A class to represent the result of the distortion or undistortion of 2D image points.

    It extends :class:`pysdic.TransformationResult` with the number of parameters of each step,
    to give access to the blocks of the Jacobian with respect to the parameters of the intrinsic,
    distortion, rectification :math:`R` and projection :math:`P` transformations.

    The parameters are ordered as the steps are applied:

    .. code-block:: text

        jacobian_dp[..., :N_i]                           # intrinsic parameters
        jacobian_dp[..., N_i:N_i + N_d]                  # distortion parameters
        jacobian_dp[..., N_i + N_d:N_i + N_d + N_R]      # rectification parameters
        jacobian_dp[..., N_i + N_d + N_R:]               # projection parameters

    .. seealso::

        - Functions :func:`pysdic.distort_points` and :func:`pysdic.undistort_points` that produce
          a :class:`pysdic.DistortionResult` object.

    Parameters
    ----------
    x : numpy.ndarray
        The output points. Shape :math:`(\dots, 2)`.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the output points with respect to the image points, or None if not computed.
        Shape :math:`(\dots, 2, 2)`.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the output points with respect to the parameters of all the steps,
        or None if not computed. Shape :math:`(\dots, 2, N_i + N_d + N_R + N_P)`.

    n_intrinsic : int
        The number of parameters :math:`N_i` of the intrinsic transformation.

    n_distortion : int
        The number of parameters :math:`N_d` of the distortion transformation.

    n_rectification : int
        The number of parameters :math:`N_R` of the rectification transformation :math:`R`.

    n_projection : int
        The number of parameters :math:`N_P` of the projection transformation :math:`P`.

    """

    n_intrinsic: int = 0
    n_distortion: int = 0
    n_rectification: int = 0
    n_projection: int = 0

    def _jacobian_block(self, start: int, stop: int) -> Optional[numpy.ndarray]:
        # View of a block of columns of the Jacobian with respect to the parameters
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., start:stop]

    @property
    def jacobian_dintrinsic(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the output points with respect to the intrinsic parameters,
        as a view of ``jacobian_dp[..., :N_i]``. Shape :math:`(\dots, 2, N_i)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the intrinsic parameters, or None if not computed.
        """
        return self._jacobian_block(0, self.n_intrinsic)

    @property
    def jacobian_ddistortion(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the output points with respect to the distortion parameters,
        as a view of ``jacobian_dp[..., N_i:N_i + N_d]``. Shape :math:`(\dots, 2, N_d)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the distortion parameters, or None if not computed.
        """
        start = self.n_intrinsic
        return self._jacobian_block(start, start + self.n_distortion)

    @property
    def jacobian_drectification(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the output points with respect to the rectification parameters,
        as a view of ``jacobian_dp[..., N_i + N_d:N_i + N_d + N_R]``. Shape :math:`(\dots, 2, N_R)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the rectification parameters, or None if not computed.
        """
        start = self.n_intrinsic + self.n_distortion
        return self._jacobian_block(start, start + self.n_rectification)

    @property
    def jacobian_dprojection(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the output points with respect to the projection parameters,
        as a view of ``jacobian_dp[..., N_i + N_d + N_R:]``. Shape :math:`(\dots, 2, N_P)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the projection parameters, or None if not computed.
        """
        start = self.n_intrinsic + self.n_distortion + self.n_rectification
        return self._jacobian_block(start, start + self.n_projection)


# ======================================================================
# Internal helpers
# ======================================================================
def _validate_steps(
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    R: Optional[Transformation],
    P: Optional[Transformation],
) -> None:
    # Check the types, the dimensions and the setup of the given transformations
    for name, transformation, input_dims in (
        ("intrinsic", intrinsic, (2,)),
        ("distortion", distortion, (2,)),
        ("R", R, (2, 3)),
        ("P", P, (2,)),
    ):
        if transformation is None:
            continue
        if not isinstance(transformation, Transformation):
            raise TypeError(
                f"{name} must be an instance of Transformation or None, got {type(transformation)}."
            )
        if transformation.input_dim not in input_dims or transformation.output_dim != 2:
            raise ValueError(
                f"The {name} transformation must map points of dimension {' or '.join(map(str, input_dims))} "
                f"to 2D points, got {transformation.input_dim}D to {transformation.output_dim}D."
            )
        if not transformation.is_setup:
            raise ValueError(
                f"The parameters of the {name} transformation ({transformation.__class__.__name__}) "
                f"are not set."
            )


def _run_steps(
    points: ArrayLike,
    steps: List[Tuple[Optional[Transformation], bool, Optional[Dict[str, Any]]]],
    *,
    dx: bool,
    dp: bool,
    bypass_dp: Optional[Sequence[bool]],
) -> DistortionResult:
    # Run the 4 steps (intrinsic, distortion, R, P). Each step is (transformation or None, inverse, kwargs).
    # The Jacobians are propagated with the chain rule as in ComposedTransformation, but each step can
    # be applied or inverted. If R has 3D inputs, the points are lifted to (x, y, 1) before R.
    dx = _validate_bool(dx, "dx")
    dp = _validate_bool(dp, "dp")
    if bypass_dp is None:
        bypass_dp = (False,) * 4
    if not isinstance(bypass_dp, (tuple, list, numpy.ndarray)):
        raise TypeError(
            f"Input 'bypass_dp' must be a sequence of 4 booleans (intrinsic, distortion, R, P) or None. "
            f"Got type {type(bypass_dp)}."
        )
    if len(bypass_dp) != 4:
        raise ValueError(
            f"Input 'bypass_dp' must contain 4 booleans (intrinsic, distortion, R, P). Got {len(bypass_dp)}."
        )
    bypass_dp = tuple(_validate_bool(b, f"bypass_dp[{i}]") for i, b in enumerate(bypass_dp))

    # Copy-free conversion: the transformations never modify their input points
    points = _as_batch_array(points, "image_points", 2, numpy.float64, finite=False, min_ndim=1)
    batch_shape = points.shape[:-1]
    x = points.reshape(-1, 2)
    n_points = x.shape[0]

    # The flattened sequence of operations: (transformation, inverse, kwargs, step index)
    operations = []
    for k, (transformation, inverse, kwargs) in enumerate(steps):
        kwargs = _validate_kwargs(kwargs, f"{_STEP_NAMES[k]}_kwargs")
        if transformation is None:
            continue
        if k == 2 and transformation.input_dim == 3:
            operations.append((NoExtrinsic(input_dim=3), True, None, None))  # lift to (x, y, 1)
        operations.append((transformation, inverse, kwargs, k))

    n_parameters = [0 if t is None else t.n_parameters for t, _, _ in steps]

    jacobian_dx = None
    blocks: List[Optional[numpy.ndarray]] = [None] * 4  # Jacobian of the current points w.r.t. p_k

    for transformation, inverse, kwargs, k in operations:
        compute_dp = dp and k is not None and not bypass_dp[k] and n_parameters[k] > 0
        # The Jacobian dx of the step is needed to propagate the previous Jacobians dp
        propagate = any(block is not None for block in blocks)
        method = transformation.invert if inverse else transformation.apply
        result = method(x, dx=dx or propagate, dp=compute_dp, **(kwargs or {}))  # None for the lift
        x = result.x

        blocks = [None if block is None else result.jacobian_dx @ block for block in blocks]
        if compute_dp:
            blocks[k] = result.jacobian_dp  # (n_points, n_out, n_p_k)

        if dx:
            jacobian_dx = (
                result.jacobian_dx if jacobian_dx is None else result.jacobian_dx @ jacobian_dx
            )  # (n_points, n_out, 2)

    if dx and jacobian_dx is None:  # No step at all: identity
        jacobian_dx = numpy.broadcast_to(numpy.eye(2), (n_points, 2, 2)).copy()

    jacobian_dp = None
    if dp:
        # Bypassed steps are filled with NaN, steps without parameters have no column
        columns = [
            numpy.full((n_points, 2, n_p), numpy.nan) if block is None else block
            for block, n_p in zip(blocks, n_parameters)
        ]
        jacobian_dp = numpy.concatenate(columns, axis=-1).reshape(*batch_shape, 2, -1)

    return DistortionResult(
        x=x.reshape(*batch_shape, 2),
        jacobian_dx=None if jacobian_dx is None else jacobian_dx.reshape(*batch_shape, 2, 2),
        jacobian_dp=jacobian_dp,
        n_intrinsic=n_parameters[0],
        n_distortion=n_parameters[1],
        n_rectification=n_parameters[2],
        n_projection=n_parameters[3],
    )


# ======================================================================
# Public functions
# ======================================================================
def undistort_points(
    image_points: ArrayLike,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    R: Optional[Transformation] = None,
    P: Optional[Transformation] = None,
    *,
    dx: bool = False,
    dp: bool = False,
    bypass_dp: Optional[Sequence[bool]] = None,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
    R_kwargs: Optional[Dict[str, Any]] = None,
    P_kwargs: Optional[Dict[str, Any]] = None,
) -> DistortionResult:
    r"""
    Undistort 2D **image points** :math:`\vec{x}_i` with the inverses of the camera intrinsic and
    distortion transformations, to obtain the **normalized points** :math:`\vec{x}_n`, or the
    **undistorted points** :math:`\vec{x}_u` if ``R`` or ``P`` are given.

    .. math::

        \vec{x}_d &= \text{Intrinsic}^{-1}(\vec{x}_i) \\
        \vec{x}_n &= \text{Distortion}^{-1}(\vec{x}_d) \\
        \vec{x}_u &= P(R(\vec{x}_n))

    Where:

    - :math:`\vec{x}_i` are the 2D **image points** in the image coordinate system.
    - :math:`\vec{x}_d` are the 2D **distorted points** in the normalized coordinate system.
    - :math:`\vec{x}_n` are the 2D **normalized points** in the normalized coordinate system.
    - :math:`\vec{x}_u` are the 2D **undistorted points** in the space required by the user.

    If :math:`R` takes 3D points (for example a :class:`pysdic.PerspectiveExtrinsic` with a rotation
    and no translation, as the rectification of ``cv2.undistortPoints``), the normalized points are
    first lifted to :math:`(x_n, y_n, 1)`.

    .. note::

        The behavior of the function can be adapted by giving the appropriate transformations:

        - Use ``intrinsic=None`` to give directly the distorted points in the normalized coordinate system.
        - Use ``P=intrinsic`` to return the undistorted points in the image coordinate system.
        - Use ``intrinsic=None`` and ``P=None`` if the distortion model is defined in the image coordinate system.

    .. warning::

        The inverse of most distortion models is computed with an iterative method (see the
        ``invert`` method of the distortion). Its options can be given with ``distortion_kwargs``.

    The Jacobians are obtained with the chain rule (as for :func:`pysdic.project_points`) and can
    be accessed with the attributes of the result:

    - ``jacobian_dx``: The Jacobian of the output points with respect to the **image points**. Shape :math:`(\dots, 2, 2)`.
    - ``jacobian_dp``: The Jacobian of the output points with respect to all the parameters. Shape :math:`(\dots, 2, N_i + N_d + N_R + N_P)`.
    - ``jacobian_dintrinsic``, ``jacobian_ddistortion``, ``jacobian_drectification``, ``jacobian_dprojection``:
      Views of ``jacobian_dp`` for the parameters of each step.

    .. note::

        If the same transformation is given twice (for example ``P=intrinsic``), its parameters appear
        in two blocks of ``jacobian_dp``. The total derivative is the sum of the two blocks.

    .. seealso::

        - Class :class:`pysdic.DistortionResult` for the result.
        - Function :func:`pysdic.distort_points` for the opposite direction.
        - Function :func:`pysdic.project_points` to project world points to image points.


    Parameters
    ----------
    image_points : ArrayLike
        The 2D image points in the image coordinate system. Shape :math:`(\dots, 2)`.

    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D), inverted on the image points.
        If None, no intrinsic is applied (identity).

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D), inverted on the distorted points.
        If None, no distortion is applied (identity).

    R : Optional[Transformation], optional (default: ``None``)
        The rectification transformation applied to the normalized points, from 2D or 3D points
        to 2D points. If None, no rectification is applied.

    P : Optional[Transformation], optional (default: ``None``)
        The projection transformation (2D to 2D) applied after the rectification.
        If None, no projection is applied.

    dx : bool, optional (default: ``False``)
        If True, compute the Jacobian of the output points with respect to the **image points**.

    dp : bool, optional (default: ``False``)
        If True, compute the Jacobian of the output points with respect to the parameters
        of the intrinsic, distortion, :math:`R` and :math:`P` transformations.

    bypass_dp : Optional[Sequence[bool]], optional (default: ``None``)
        Four booleans for the intrinsic, distortion, :math:`R` and :math:`P` transformations (in this order).
        If ``dp=True``, the Jacobian with respect to the parameters of the transformations with
        ``True`` is not computed and its columns in ``jacobian_dp`` are filled with NaN.
        If None, all the Jacobians are computed.

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the distortion transformation.

    R_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of :math:`R`.

    P_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of :math:`P`.


    Returns
    -------
    :class:`pysdic.DistortionResult`
        The normalized (or undistorted) points ``x`` with shape :math:`(\dots, 2)` and the Jacobians
        if requested.


    Examples
    --------
    Undistort image points with a pinhole intrinsic and a Brown distortion:

    .. code-block:: python

        import numpy
        from pysdic import undistort_points, PinholeIntrinsic, BrownDistortion

        image_points = numpy.array([[320.0, 240.0], [420.0, 440.0], [520.0, 540.0]])

        intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                  [0.0, 1000.0, 240.0],
                                                  [0.0, 0.0, 1.0]])
        distortion = BrownDistortion([0.1, -0.05, 0.001, 0.002, 0.01])

        normalized_points = undistort_points(image_points, intrinsic, distortion).x

    To return the undistorted points in the image coordinate system, give ``P=intrinsic``:

    .. code-block:: python

        undistorted_points = undistort_points(image_points, intrinsic, distortion, P=intrinsic).x

    """
    _validate_steps(intrinsic, distortion, R, P)
    return _run_steps(
        image_points,
        [
            (intrinsic, True, intrinsic_kwargs),
            (distortion, True, distortion_kwargs),
            (R, False, R_kwargs),
            (P, False, P_kwargs),
        ],
        dx=dx,
        dp=dp,
        bypass_dp=bypass_dp,
    )


def distort_points(
    image_points: ArrayLike,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    R: Optional[Transformation] = None,
    P: Optional[Transformation] = None,
    *,
    dx: bool = False,
    dp: bool = False,
    bypass_dp: Optional[Sequence[bool]] = None,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
    R_kwargs: Optional[Dict[str, Any]] = None,
    P_kwargs: Optional[Dict[str, Any]] = None,
) -> DistortionResult:
    r"""
    Distort 2D **image points** :math:`\vec{x}_i` with the inverse of the camera intrinsic and the
    distortion transformation, to obtain the **distorted points** :math:`\vec{x}_d`, or the
    **transformed points** :math:`\vec{x}_u` if ``R`` or ``P`` are given.

    .. math::

        \vec{x}_n &= \text{Intrinsic}^{-1}(\vec{x}_i) \\
        \vec{x}_d &= \text{Distortion}(\vec{x}_n) \\
        \vec{x}_u &= P(R(\vec{x}_d))

    Where:

    - :math:`\vec{x}_i` are the 2D **image points** in the image coordinate system.
    - :math:`\vec{x}_n` are the 2D **normalized points** in the normalized coordinate system.
    - :math:`\vec{x}_d` are the 2D **distorted points** in the normalized coordinate system.
    - :math:`\vec{x}_u` are the 2D **transformed points** in the space required by the user.

    If :math:`R` takes 3D points, the distorted points are first lifted to :math:`(x_d, y_d, 1)`.

    .. note::

        The behavior of the function can be adapted by giving the appropriate transformations:

        - Use ``intrinsic=None`` to give directly the normalized points in the normalized coordinate system.
        - Use ``P=intrinsic`` to return the distorted points in the image coordinate system.
        - Use ``intrinsic=None`` and ``P=None`` if the distortion model is defined in the image coordinate system.

    The Jacobians and the arguments are the same as for :func:`pysdic.undistort_points`, except that
    the distortion is applied (``distortion_kwargs`` are passed to its ``apply`` method).

    .. seealso::

        - Class :class:`pysdic.DistortionResult` for the result.
        - Function :func:`pysdic.undistort_points` for the opposite direction.


    Parameters
    ----------
    image_points : ArrayLike
        The 2D image points in the image coordinate system. Shape :math:`(\dots, 2)`.

    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D), inverted on the image points.
        If None, no intrinsic is applied (identity).

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D), applied to the normalized points.
        If None, no distortion is applied (identity).

    R, P, dx, dp, bypass_dp, intrinsic_kwargs, R_kwargs, P_kwargs
        See :func:`pysdic.undistort_points`.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the distortion transformation.


    Returns
    -------
    :class:`pysdic.DistortionResult`
        The distorted (or transformed) points ``x`` with shape :math:`(\dots, 2)` and the Jacobians
        if requested.


    Examples
    --------
    Distort the image points of a perfect pinhole camera and return them in the image coordinate system:

    .. code-block:: python

        distorted_points = distort_points(image_points, intrinsic, distortion, P=intrinsic).x

    """
    _validate_steps(intrinsic, distortion, R, P)
    return _run_steps(
        image_points,
        [
            (intrinsic, True, intrinsic_kwargs),
            (distortion, False, distortion_kwargs),
            (R, False, R_kwargs),
            (P, False, P_kwargs),
        ],
        dx=dx,
        dp=dp,
        bypass_dp=bypass_dp,
    )