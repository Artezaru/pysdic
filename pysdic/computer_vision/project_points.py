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
from typing import Any, Dict, Optional, Sequence

import numpy
from numpy.typing import ArrayLike

from ..transformations.transformation import (
    ComposedTransformation,
    Transformation,
    TransformationResult,
)
from ..transformations.no_transformation import NoDistortion, NoExtrinsic, NoIntrinsic
from .._validation import _as_batch_array


@dataclass(slots=True, frozen=True)
class ProjectionResult(TransformationResult):
    r"""
    A class to represent the result of the projection of world points to image points.

    It extends :class:`pysdic.TransformationResult` with the number of parameters of each step
    of the projection, to give access to the blocks of the Jacobian with respect to the
    parameters of the extrinsic, distortion and intrinsic transformations.

    The parameters of the projection are ordered as in the composition
    :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}`:

    .. code-block:: text

        jacobian_dp[..., :N_e]               # extrinsic parameters
        jacobian_dp[..., N_e:N_e + N_d]      # distortion parameters
        jacobian_dp[..., N_e + N_d:]         # intrinsic parameters

    .. seealso::

        - Function :func:`pysdic.project_points` that produces a :class:`pysdic.ProjectionResult` object.

    Parameters
    ----------
    x : numpy.ndarray
        The image points :math:`\vec{x}_i`. Shape :math:`(\dots, 2)`.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the image points with respect to the world points, or None if not computed.
        Shape :math:`(\dots, 2, n_w)` with :math:`n_w` the dimension of the world points.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the image points with respect to the parameters of the projection,
        or None if not computed. Shape :math:`(\dots, 2, N_e + N_d + N_i)`.

    n_extrinsic : int
        The number of parameters :math:`N_e` of the extrinsic transformation.

    n_distortion : int
        The number of parameters :math:`N_d` of the distortion transformation.

    n_intrinsic : int
        The number of parameters :math:`N_i` of the intrinsic transformation.

    """

    n_extrinsic: int = 0
    n_distortion: int = 0
    n_intrinsic: int = 0

    @property
    def image_points(self) -> numpy.ndarray:
        r"""
        [Get] the image points :math:`\vec{x}_i` (alias of ``x``). Shape :math:`(\dots, 2)`.

        Returns
        -------
        numpy.ndarray
            The image points.
        """
        return self.x

    def _jacobian_block(self, start: int, stop: int) -> Optional[numpy.ndarray]:
        # View of a block of columns of the Jacobian with respect to the parameters
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., start:stop]

    @property
    def jacobian_dextrinsic(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the image points with respect to the extrinsic parameters,
        as a view of ``jacobian_dp[..., :N_e]``. Shape :math:`(\dots, 2, N_e)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the extrinsic parameters, or None if not computed.
        """
        return self._jacobian_block(0, self.n_extrinsic)

    @property
    def jacobian_ddistortion(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the image points with respect to the distortion parameters,
        as a view of ``jacobian_dp[..., N_e:N_e + N_d]``. Shape :math:`(\dots, 2, N_d)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the distortion parameters, or None if not computed.
        """
        start = self.n_extrinsic
        return self._jacobian_block(start, start + self.n_distortion)

    @property
    def jacobian_dintrinsic(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the image points with respect to the intrinsic parameters,
        as a view of ``jacobian_dp[..., N_e + N_d:]``. Shape :math:`(\dots, 2, N_i)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the intrinsic parameters, or None if not computed.
        """
        start = self.n_extrinsic + self.n_distortion
        return self._jacobian_block(start, start + self.n_intrinsic)


def compose_projection(
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    extrinsic: Optional[Transformation],
    *,
    world_dim: int = 3,
) -> ComposedTransformation:
    r"""
    Build the projection :math:`T` from the **world points** to the **image points** as the composition
    of the extrinsic, distortion and intrinsic transformations:

    .. math::

        T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}

    The parameters of the projection are the concatenation of the parameters of the extrinsic,
    distortion and intrinsic transformations (in this order).

    .. note::

        The transformations are not copied: modifying the parameters of the projection modifies
        the parameters of the transformations, and conversely.

    Parameters
    ----------
    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D). If None, :class:`pysdic.NoIntrinsic` is used.

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D). If None, :class:`pysdic.NoDistortion` is used.

    extrinsic : Optional[Transformation]
        The extrinsic transformation (world points to 2D). If None, :class:`pysdic.NoExtrinsic`
        is used with the input dimension ``world_dim``.

    world_dim : int, optional (default: ``3``)
        The dimension of the world points (2 or 3), only used if ``extrinsic`` is None.


    Returns
    -------
    :class:`pysdic.ComposedTransformation`
        The projection ``ComposedTransformation([extrinsic, distortion, intrinsic])``.

    """
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if extrinsic is None:
        extrinsic = NoExtrinsic(input_dim=world_dim)

    for name, transformation, input_dims in (
        ("extrinsic", extrinsic, (2, 3)),
        ("distortion", distortion, (2,)),
        ("intrinsic", intrinsic, (2,)),
    ):
        if not isinstance(transformation, Transformation):
            raise TypeError(
                f"{name} must be an instance of Transformation, got {type(transformation)}."
            )
        if transformation.input_dim not in input_dims or transformation.output_dim != 2:
            raise ValueError(
                f"The {name} transformation must map points of dimension {' or '.join(map(str, input_dims))} "
                f"to 2D points, got {transformation.input_dim}D to {transformation.output_dim}D."
            )

    return ComposedTransformation([extrinsic, distortion, intrinsic])


def project_points(
    world_points: ArrayLike,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    extrinsic: Optional[Transformation],
    *,
    dx: bool = False,
    dp: bool = False,
    bypass_dp: Optional[Sequence[bool]] = None,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
    extrinsic_kwargs: Optional[Dict[str, Any]] = None,
) -> ProjectionResult:
    r"""
    Project **world points** :math:`\vec{X}_w` to 2D **image points** :math:`\vec{x}_i`
    using the camera extrinsic, distortion and intrinsic transformations.

    .. math::

        \vec{x}_n &= \text{Extrinsic}(\vec{X}_w) \\
        \vec{x}_d &= \text{Distortion}(\vec{x}_n) \\
        \vec{x}_i &= \text{Intrinsic}(\vec{x}_d)

    Where:

    - :math:`\vec{X}_w` are the **world points** (3D, or 2D for planar problems) in the world coordinate system.
    - :math:`\vec{x}_n` are the 2D **normalized points** in the normalized coordinate system.
    - :math:`\vec{x}_d` are the 2D **distorted points** in the normalized coordinate system.
    - :math:`\vec{x}_i` are the 2D **image points** in the image coordinate system.

    .. note::

        The **image points** can be then converted to pixel coordinates
        :math:`(u, v)` by applying a swap of the axes.

    The projection is computed with the composition :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}`
    (see :func:`pysdic.compose_projection`), and the Jacobians are obtained with the chain rule:

    .. math::

        \frac{\partial \vec{x}_i}{\partial \vec{X}_w} = J_i \, J_d \, \frac{\partial \vec{x}_n}{\partial \vec{X}_w},
        \qquad
        \frac{\partial \vec{x}_i}{\partial \vec{p}} =
        \begin{bmatrix}
        J_i \, J_d \, \frac{\partial \vec{x}_n}{\partial \vec{p}_e} &
        J_i \, \frac{\partial \vec{x}_d}{\partial \vec{p}_d} &
        \frac{\partial \vec{x}_i}{\partial \vec{p}_i}
        \end{bmatrix}

    with :math:`J_d = \frac{\partial \vec{x}_d}{\partial \vec{x}_n}` and :math:`J_i = \frac{\partial \vec{x}_i}{\partial \vec{x}_d}`.

    The Jacobians can be accessed with the following attributes of the result:

    - ``jacobian_dx``: The Jacobian of the **image points** with respect to the **world points**. Shape :math:`(\dots, 2, n_w)`.
    - ``jacobian_dp``: The Jacobian of the **image points** with respect to all the parameters. Shape :math:`(\dots, 2, N_e + N_d + N_i)`.
    - ``jacobian_dextrinsic``: View of ``jacobian_dp[..., :N_e]`` for the extrinsic parameters.
    - ``jacobian_ddistortion``: View of ``jacobian_dp[..., N_e:N_e + N_d]`` for the distortion parameters.
    - ``jacobian_dintrinsic``: View of ``jacobian_dp[..., N_e + N_d:]`` for the intrinsic parameters.

    .. seealso::

        - Class :class:`pysdic.ProjectionResult` for the result of the projection.
        - Function :func:`pysdic.compose_projection` to build the projection as a transformation
          (for example to optimize some of its parameters with a :class:`pysdic.ReparametrizedTransformation`).


    Parameters
    ----------
    world_points : ArrayLike
        The world points with shape :math:`(\dots, n_w)`, with :math:`n_w = 3` (or :math:`n_w = 2`
        for planar problems).

    intrinsic : Optional[Transformation]
        The intrinsic transformation to be applied to the distorted points.
        If None, :class:`pysdic.NoIntrinsic` is used (identity).

    distortion : Optional[Transformation]
        The distortion transformation to be applied to the normalized points.
        If None, :class:`pysdic.NoDistortion` is used (identity).

    extrinsic : Optional[Transformation]
        The extrinsic transformation to be applied to the world points.
        If None, :class:`pysdic.NoExtrinsic` is used with the dimension of the world points.

    dx : bool, optional (default: ``False``)
        If True, compute the Jacobian of the **image points** with respect to the **world points**.

    dp : bool, optional (default: ``False``)
        If True, compute the Jacobian of the **image points** with respect to the parameters
        of the extrinsic, distortion and intrinsic transformations.

    bypass_dp : Optional[Sequence[bool]], optional (default: ``None``)
        Three booleans for the extrinsic, distortion and intrinsic transformations (in this order).
        If ``dp=True``, the Jacobian with respect to the parameters of the transformations with
        ``True`` is not computed and its columns in ``jacobian_dp`` are filled with NaN, which saves
        computations when only some parameters are needed. If None, all the Jacobians are computed.
        For example, ``bypass_dp=(False, True, True)`` only computes ``jacobian_dextrinsic``.

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the distortion transformation.

    extrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the extrinsic transformation.


    Returns
    -------
    :class:`pysdic.ProjectionResult`
        The image points (``x`` or ``image_points``) with shape :math:`(\dots, 2)` and the
        Jacobians if requested.

    """
    # 3D world points, or 2D world points for planar problems
    world_points = _as_batch_array(
        world_points, "world_points", (3, 2), numpy.float64, finite=False, min_ndim=2
    )

    projection = compose_projection(
        intrinsic, distortion, extrinsic, world_dim=world_points.shape[-1]
    )
    extrinsic, distortion, intrinsic = projection.transformations
    for name, transformation in (
        ("extrinsic", extrinsic),
        ("distortion", distortion),
        ("intrinsic", intrinsic),
    ):
        if not transformation.is_setup:
            raise ValueError(
                f"The parameters of the {name} transformation ({transformation.__class__.__name__}) "
                f"are not set."
            )

    result = projection.apply(
        world_points,
        dx=dx,
        dp=dp,
        kwargs=[extrinsic_kwargs, distortion_kwargs, intrinsic_kwargs],
        bypass_dp=bypass_dp,
    )

    return ProjectionResult(
        x=result.x,
        jacobian_dx=result.jacobian_dx,
        jacobian_dp=result.jacobian_dp,
        n_extrinsic=extrinsic.n_parameters,
        n_distortion=distortion.n_parameters,
        n_intrinsic=intrinsic.n_parameters,
    )