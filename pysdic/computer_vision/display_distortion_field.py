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
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy
from numpy.typing import ArrayLike

from ..transformations.transformation import Transformation
from .distorting_points import distort_points
from .._validation import (
    _as_bool_mask,
    _check_shape,
    _validate_bool,
    _validate_choice,
    _validate_integer,
    _validate_real,
)


_MAX_CONTOUR_LEVELS = 200

# Displayed quantity: (title, colormap) for the axis 0 (x), 1 (y) and "n" (magnitude)
_AXIS_DISPLAY = {
    0: ("Horizontal distortion", "RdBu_r"),
    1: ("Vertical distortion", "RdBu_r"),
    "n": ("Distortion magnitude", "viridis"),
}

AxisLike = Union[int, str]
BoundsLike = Optional[Tuple[Real, Real]]


# =============================================================================
# Internal helpers
# =============================================================================
def _parse_axis(axis: AxisLike, name: str) -> Union[int, str]:
    # 0 / "x" -> 0, 1 / "y" -> 1, "n" -> "n"
    if isinstance(axis, str):
        return {"x": 0, "y": 1, "n": "n"}[_validate_choice(axis, name, ("x", "y", "n"))]
    axis = _validate_integer(axis, name)  # booleans are rejected
    if axis not in (0, 1):
        raise ValueError(f"Input '{name}' must be 0, 1, 'x', 'y' or 'n'. Got {axis!r}.")
    return axis


def _is_bounds(value: Any) -> bool:
    # A single pair (min, max) of real numbers
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, Real) for v in value)
    )


def _per_axis(value: Any, n_axes: int, name: str, is_single) -> List[Any]:
    # A single value repeated for each axis, or one value per axis (length n_axes)
    if is_single(value):
        return [value] * n_axes
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"Input '{name}' must be a single value or a list/tuple of {n_axes} values. Got type {type(value)}."
        )
    if len(value) != n_axes:
        raise ValueError(
            f"Input '{name}' must contain one value per displayed axis ({n_axes}). Got {len(value)} values."
        )
    return list(value)


def _validate_bounds(bounds: BoundsLike, name: str) -> Optional[Tuple[float, float]]:
    if bounds is None:
        return None
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise TypeError(f"Input '{name}' must be a tuple (min, max) of two real numbers or None. Got {bounds!r}.")
    bounds = (
        _validate_real(bounds[0], f"{name}[0]", finite=True),
        _validate_real(bounds[1], f"{name}[1]", finite=True),
    )
    if not bounds[0] < bounds[1]:
        raise ValueError(f"Input '{name}' must be (min, max) with min < max. Got {bounds}.")
    return bounds


def _draw_field(
    ax: matplotlib.axes.Axes,
    values: numpy.ndarray,
    X: numpy.ndarray,
    Y: numpy.ndarray,
    axis: Union[int, str],
    contour_step: Optional[float],
    contour_bounds: Optional[Tuple[float, float]],
    width: int,
    height: int,
) -> None:
    # Draw one component of the field (ny, nx) on the axes (cleared), NaN values are not drawn
    title, cmap_name = _AXIS_DISPLAY[axis]
    if not numpy.any(numpy.isfinite(values)):
        raise ValueError(f"The field of the '{title}' contains only NaN or infinite values.")

    # ------------- bounds and levels -------------
    if contour_bounds is None:
        vmin, vmax = float(numpy.nanmin(values)), float(numpy.nanmax(values))
        if axis != "n":  # Symmetric bounds so that 0 is white
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
    else:
        vmin, vmax = contour_bounds

    levels = None
    if contour_step is not None:
        k_min = int(numpy.floor(vmin / contour_step))
        k_max = max(int(numpy.ceil(vmax / contour_step)), k_min + 1)  # At least 2 levels
        if k_max - k_min > _MAX_CONTOUR_LEVELS:
            raise ValueError(
                f"The contour step {contour_step} gives {k_max - k_min + 1} levels for the range "
                f"[{vmin:.4g}, {vmax:.4g}] of the '{title}'. Increase contour_step or give contour_bounds."
            )
        levels = numpy.arange(k_min, k_max + 1) * contour_step
        vmin, vmax = levels[0], levels[-1]
        cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    else:
        cmap = plt.get_cmap(cmap_name)

    # ------------- display -------------
    ax.clear()
    nx, ny = X.shape[1], X.shape[0]
    hx = 0.5 * (width - 1) / (nx - 1)  # Half step of the reduced grid
    hy = 0.5 * (height - 1) / (ny - 1)
    image = ax.imshow(
        values,
        extent=(-hx, width - 1 + hx, height - 1 + hy, -hy),  # Cells centered on the samples
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if levels is not None:
        contours = ax.contour(X, Y, values, levels=levels, colors="black", linewidths=0.8)
        ax.clabel(contours, fmt="%g", fontsize=8, inline=True)

    ax.figure.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Image x")
    ax.set_ylabel("Image y")
    ax.set_xlim(-hx, width - 1 + hx)
    ax.set_ylim(height - 1 + hy, -hy)
    ax.set_aspect("equal")


# =============================================================================
# Display
# =============================================================================
def display_distortion_field(
    distortion: Transformation,
    image_size: Tuple[Integral, Integral],
    intrinsic: Optional[Transformation] = None,
    *,
    axis: Union[AxisLike, Sequence[AxisLike]] = "n",
    reduction: Integral = 4,
    contour_step: Union[Optional[Real], Sequence[Optional[Real]]] = 1.0,
    contour_bounds: Union[BoundsLike, Sequence[BoundsLike]] = None,
    mask: Optional[ArrayLike] = None,
    ax: Optional[Union[matplotlib.axes.Axes, Sequence[matplotlib.axes.Axes]]] = None,
    show: bool = True,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[matplotlib.axes.Axes, Tuple[matplotlib.axes.Axes, ...]]:
    r"""
    Display the displacement field generated by a distortion model on the image.

    A regular grid of image points :math:`\vec{x}` is built and the displacement of each point
    is computed once with :func:`pysdic.distort_points` (with ``P=intrinsic``):

    .. math::

        \Delta(\vec{x}) = \text{Intrinsic}(\text{Distortion}(\text{Intrinsic}^{-1}(\vec{x}))) - \vec{x}

    One or several quantities are then displayed side by side: the displacement along one axis
    (:math:`\Delta_x` or :math:`\Delta_y`) and/or its magnitude :math:`\|\Delta\|`, with optional
    contour lines.

    .. note::

        If ``intrinsic`` is None, the distortion is directly applied to the image points, so the
        displacements are expressed in the coordinate system of the distortion (for example normalized
        coordinates). Adapt ``contour_step`` accordingly (for example ``0.001``).

    .. seealso::

        - Function :func:`pysdic.distort_points` to compute the distorted points.


    Parameters
    ----------
    distortion : Transformation
        The distortion transformation (2D to 2D) to display.

    image_size : Tuple[Integral, Integral]
        The size of the image ``(height, width)`` in pixels.

    intrinsic : Optional[Transformation], optional (default: ``None``)
        The intrinsic transformation (2D to 2D) of the camera. If None, no intrinsic is applied.

    axis : Union[int, str, Sequence[Union[int, str]]], optional (default: ``"n"``)
        The displayed quantity: ``0`` or ``"x"`` for the horizontal displacement, ``1`` or ``"y"`` for
        the vertical displacement and ``"n"`` for the magnitude of the displacement. A list or tuple of
        these values (e.g. ``("x", "y", "n")``) displays several quantities side by side, in the given
        order. The same quantity cannot be given twice (``"x"`` and ``0`` are the same).

    reduction : Integral, optional (default: ``4``)
        The reduction factor of the grid: the field is evaluated approximately every ``reduction`` pixels.

    contour_step : Union[Optional[Real], Sequence[Optional[Real]]], optional (default: ``1.0``)
        The distance between two contour levels, the levels are :math:`k \cdot \text{contour\_step}`
        with :math:`k` an integer. The colormap is discretized with the same levels.
        If None, no contours are drawn and the colormap is continuous. A single value is used for
        all the displayed quantities, a list/tuple gives one value per quantity of ``axis``.

    contour_bounds : Union[Optional[Tuple[Real, Real]], Sequence[Optional[Tuple[Real, Real]]]], optional (default: ``None``)
        The range ``(min, max)`` of the contour levels and of the colormap. If None, the range of the field
        is used (symmetric around 0 for the displacements along one axis). Useful to keep the same colors
        on several figures (e.g. the frames of a video). A single pair is used for all the displayed
        quantities, a list/tuple gives one pair (or None) per quantity of ``axis``.

    mask : Optional[ArrayLike], optional (default: ``None``)
        A boolean mask of shape ``(height, width)`` of the pixels of the image where the field is
        computed. The points of the reduced grid are associated with their nearest pixel: outside the
        mask, the field is not computed and set to NaN (not drawn).

    ax : Optional[Union[matplotlib.axes.Axes, Sequence[matplotlib.axes.Axes]]], optional (default: ``None``)
        The axes to draw on (they are cleared): one axes per quantity of ``axis``, in the same order
        (a single axes is accepted for a single quantity). If None, a new figure with the quantities
        side by side is created.

    show : bool, optional (default: ``True``)
        Whether to show the figure (with ``matplotlib.pyplot.show``).

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the distortion transformation.


    Returns
    -------
    Union[matplotlib.axes.Axes, Tuple[matplotlib.axes.Axes, ...]]
        The axes containing the distortion fields, in the order of ``axis``: a single axes if
        ``axis`` is a single quantity, a tuple of axes otherwise.


    Raises
    ------
    TypeError
        If an input has a wrong type (e.g. a non-boolean mask).

    ValueError
        If an input has a wrong value or length, a quantity is given twice, the mask has not the
        shape of the image or selects no point of the grid, or a field contains only NaN.


    Examples
    --------
    .. code-block:: python

        from pysdic import display_distortion_field, PinholeIntrinsic, BrownDistortion

        intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                  [0.0, 1000.0, 240.0],
                                                  [0.0, 0.0, 1.0]])
        distortion = BrownDistortion([0.1, -0.05, 0.001, 0.002, 0.01])

        # The three quantities side by side, with fixed bounds
        ax_x, ax_y, ax_n = display_distortion_field(
            distortion, (480, 640), intrinsic,
            axis=("x", "y", "n"),
            contour_step=(2.0, 2.0, 1.0),
            contour_bounds=[(-20, 20), (-20, 20), (0, 20)],
        )

    """
    # ------------- arguments -------------
    single = isinstance(axis, (str, Integral))
    if single:
        axes_ids = [_parse_axis(axis, "axis")]
    else:
        if not isinstance(axis, (tuple, list)) or len(axis) == 0:
            raise TypeError(
                f"Input 'axis' must be 0, 1, 'x', 'y', 'n' or a non-empty list/tuple of them. Got {axis!r}."
            )
        axes_ids = [_parse_axis(a, f"axis[{k}]") for k, a in enumerate(axis)]
        if len(set(axes_ids)) != len(axes_ids):
            raise ValueError(f"Input 'axis' must not contain the same quantity twice. Got {axis!r}.")
    n_axes = len(axes_ids)

    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise TypeError(
            f"Input 'image_size' must be a tuple (height, width) of two integers. Got {image_size!r}."
        )
    height = _validate_integer(image_size[0], "image_size[0]", min_value=1)
    width = _validate_integer(image_size[1], "image_size[1]", min_value=1)
    reduction = _validate_integer(reduction, "reduction", min_value=1)
    show = _validate_bool(show, "show")

    steps = [
        _validate_real(step, f"contour_step[{k}]", positive=True, finite=True, allow_none=True)
        for k, step in enumerate(
            _per_axis(contour_step, n_axes, "contour_step", lambda v: v is None or isinstance(v, Real))
        )
    ]
    bounds = [
        _validate_bounds(b, f"contour_bounds[{k}]")
        for k, b in enumerate(
            _per_axis(contour_bounds, n_axes, "contour_bounds", lambda v: v is None or _is_bounds(v))
        )
    ]

    if ax is None:
        axes_list = None
    elif isinstance(ax, matplotlib.axes.Axes):
        if n_axes != 1:
            raise ValueError(f"Input 'ax' must contain one axes per displayed quantity ({n_axes}). Got a single axes.")
        axes_list = [ax]
    else:
        if not isinstance(ax, (tuple, list, numpy.ndarray)):
            raise TypeError(f"Input 'ax' must be a matplotlib Axes, a list/tuple of Axes or None. Got type {type(ax)}.")
        axes_list = list(numpy.ravel(numpy.asarray(ax, dtype=object))) if isinstance(ax, numpy.ndarray) else list(ax)
        if len(axes_list) != n_axes:
            raise ValueError(f"Input 'ax' must contain one axes per displayed quantity ({n_axes}). Got {len(axes_list)}.")
        for k, a in enumerate(axes_list):
            if not isinstance(a, matplotlib.axes.Axes):
                raise TypeError(f"Input 'ax[{k}]' must be a matplotlib Axes. Got type {type(a)}.")
        if len({id(a) for a in axes_list}) != n_axes:
            raise ValueError("Input 'ax' must not contain the same axes twice.")

    if mask is not None:
        mask = numpy.asarray(mask)
        _check_shape(mask, "mask", (height, width), shape_hint=f"({height}, {width}) (height, width)")
        mask = _as_bool_mask(mask.ravel(), height * width, "mask").reshape(height, width)

    # ------------- field (computed once) -------------
    nx = max(2, width // reduction)
    ny = max(2, height // reduction)
    X, Y = numpy.meshgrid(numpy.linspace(0, width - 1, nx), numpy.linspace(0, height - 1, ny))
    points = numpy.stack((X, Y), axis=-1)  # (ny, nx, 2)

    def _displacements(image_points: numpy.ndarray) -> numpy.ndarray:
        return (
            distort_points(
                image_points,
                intrinsic,
                distortion,
                P=intrinsic,
                intrinsic_kwargs=intrinsic_kwargs,
                distortion_kwargs=distortion_kwargs,
            ).x
            - image_points
        )

    if mask is None:
        field = _displacements(points)  # (ny, nx, 2)
    else:
        # Nearest pixel of each point of the grid (X along the columns, Y along the rows)
        grid_mask = mask[numpy.rint(Y).astype(int), numpy.rint(X).astype(int)]  # (ny, nx)
        if not numpy.any(grid_mask):
            raise ValueError(
                f"Input 'mask' selects no point of the reduced grid (reduction={reduction}). "
                f"Reduce 'reduction' or enlarge the mask."
            )
        field = numpy.full(points.shape, numpy.nan, dtype=numpy.float64)
        field[grid_mask] = _displacements(points[grid_mask])  # (N, 2) only for the selected points

    # ------------- display -------------
    if axes_list is None:
        _, created = plt.subplots(1, n_axes, figsize=(8 * n_axes, 7), squeeze=False)
        axes_list = list(created[0])

    for current_ax, axis_id, step, bound in zip(axes_list, axes_ids, steps, bounds):
        values = numpy.linalg.norm(field, axis=-1) if axis_id == "n" else field[..., axis_id]
        _draw_field(current_ax, values, X, Y, axis_id, step, bound, width, height)

    for figure in {id(a.figure): a.figure for a in axes_list}.values():
        if figure.canvas is not None:
            figure.canvas.draw_idle()
    if show:
        plt.show()

    return axes_list[0] if single else tuple(axes_list)