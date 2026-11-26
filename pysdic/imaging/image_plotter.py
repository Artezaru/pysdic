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

from typing import Any, Dict, List, Optional, Tuple, Union
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle

from .. import fem
from ..geometry.mesh import Mesh
from ..geometry.point_cloud import PointCloud
from ..geometry.integration_points import IntegrationPoints
from ..geometry.plotter3d import _resolve_values, _scalar_values, _warped_coordinates
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_kwargs,
    _validate_real,
    _as_array,
    _as_bool_mask,
    _as_scalar_or_1d_array,
)

from .camera import Camera
from .image import Image
from .view import View


# Order of the nodes along the contour of each element type (corner and mid-edge nodes)
_CONTOUR_NODES = {
    "segment_2": (0, 1),
    "segment_3": (0, 2, 1),
    "triangle_3": (0, 1, 2),
    "triangle_6": (0, 3, 1, 4, 2, 5),
    "quadrangle_4": (0, 1, 2, 3),
    "quadrangle_8": (0, 4, 1, 5, 2, 6, 3, 7),
}

Color = Union[str, Tuple[Real, ...]]


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_opacity(value: Real, name: str) -> float:
    value = _validate_real(value, name, non_negative=True, finite=True)
    if value > 1.0:
        raise ValueError(f"Input '{name}' must be between 0.0 and 1.0. Got {value}.")
    return value


class _OpacityFilter(object):
    # matplotlib 'agg_filter' multiplying the alpha channel of the rendered artist
    def __init__(self, opacity: float) -> None:
        self.opacity = opacity

    def __call__(self, image: numpy.ndarray, dpi: float) -> Tuple[numpy.ndarray, int, int]:
        image = image.copy()
        image[..., 3] *= self.opacity
        return image, 0, 0


def _validate_clim(clim: Optional[Tuple[Real, Real]], data: numpy.ndarray) -> Tuple[float, float]:
    # Color limits (vmin, vmax) given by the user, or the extrema of the finite values of 'data'
    if clim is None:
        finite = data[numpy.isfinite(data)] if numpy.issubdtype(data.dtype, numpy.floating) else data
        if finite.size == 0:
            raise ValueError("The image has no finite value: give 'clim' explicitly.")
        vmin, vmax = float(finite.min()), float(finite.max())
        return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)
    vmin, vmax = _as_array(clim, "clim", numpy.float64, shape=(2,), finite=True)
    if not vmax > vmin:
        raise ValueError(f"Input 'clim' must satisfy min < max. Got ({vmin}, {vmax}).")
    return float(vmin), float(vmax)


def _color_kwargs(
    color: Optional[Color],
    values: Optional[numpy.ndarray],
    mask: Optional[numpy.ndarray],
    clim: Optional[Tuple[Real, Real]],
    cmap: Optional[Union[str, matplotlib.colors.Colormap]],
    below_color: Optional[Color],
    above_color: Optional[Color],
    values_key: Optional[str],
) -> Dict[str, Any]:
    # Keyword arguments of a matplotlib artist: a uniform color, or values mapped with a colormap
    if values is None:
        return {"color": color}
    if cmap is None:
        cmap = matplotlib.rcParams["image.cmap"]
    colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    colormap = colormap.with_extremes(under=below_color, over=above_color)
    kwargs: Dict[str, Any] = {"cmap": colormap}
    if clim is not None:
        vmin, vmax = _as_array(clim, "property_clim", numpy.float64, shape=(2,), finite=True)
        if not vmax > vmin:
            raise ValueError(f"Input 'property_clim' must satisfy min < max. Got ({vmin}, {vmax}).")
        kwargs["norm"] = matplotlib.colors.Normalize(vmin=float(vmin), vmax=float(vmax))
    if values_key is not None:
        kwargs[values_key] = values if mask is None else values[mask]
    return kwargs


def _resolve_image_vectors(
    container: Any, vectors: Union[str, ArrayLike], n_rows: int, name: str, location: str
) -> numpy.ndarray:
    # Vectors (n_rows, 2) in the image plane, given by the name of a property or an array
    if isinstance(vectors, str):
        if vectors not in container:
            raise ValueError(f"Property '{vectors}' not found in the {location}.")
        vectors = container[vectors]
    return _as_array(vectors, name, numpy.float64, shape=(n_rows, 2), finite=False)


class ImagePlotter(object):
    r"""
    A plotter for the geometrical objects of pysdic (:class:`pysdic.PointCloud`,
    :class:`pysdic.Mesh` and :class:`pysdic.IntegrationPoints`) and the images on the
    sensor of a camera, based on ``matplotlib``.

    The API mirrors :class:`pysdic.Plotter3D`: ``add_image``, ``add_sensor``,
    ``add_point_cloud``, ``add_mesh``, ``add_integration_points``, ``add_rectangle``,
    ``set_title`` and ``show``. It also draws the regions of interest of the images with
    ``add_roi``.

    The plot is in the **image coordinate system** :math:`(x, y)` in pixels: :math:`x`
    is the column index and :math:`y` the row index (pointing downwards). The centers of
    the pixels are at integer coordinates, so the sensor covers
    :math:`[-0.5, W - 0.5] \times [-0.5, H - 0.5]`.

    The objects are drawn as follows, according to their dimension :math:`E`:

    - With ``projected=None`` (default): if a camera is set and :math:`E` is the input
      dimension of the camera, the points are projected with
      :meth:`pysdic.Camera.project`. Otherwise :math:`E` must be 2 and the points are
      drawn as image points :math:`(x, y)`.
    - With ``projected=True``: the points are always projected (a camera is required).
    - With ``projected=False``: the points are always drawn as image points (:math:`E = 2`).

    The non-finite points (e.g. after a projection) are not drawn, nor the faces with a
    non-finite vertex.

    Parameters
    ----------
    source: Union[:class:`pysdic.Camera`, :class:`pysdic.View`, Tuple[Integral, Integral]]
        The camera (used to project the objects), the view (its camera and its image, the
        default of :meth:`add_image`) or the sensor size ``(height, width)`` in pixels
        (only image points can then be drawn). The sensor size cannot be changed.

    figsize: Optional[Tuple[Real, Real]], optional
        The size of the new figure in inches (width, height). Default: ``(8, 6)``.
        Cannot be given with ``ax``.

    ax: Optional[matplotlib.axes.Axes], optional
        Existing matplotlib axes to draw in (e.g. a subplot of a figure composed of
        several plotters). By default, a new figure is created.

    Raises
    ------
    TypeError
        If ``source`` or ``ax`` has a wrong type.

    ValueError
        If the sensor size or ``figsize`` are not strictly positive, or if both
        ``figsize`` and ``ax`` are given.

    """

    __slots__ = ["_height", "_width", "_camera", "_view_image", "_figure", "_ax", "_colorbars"]

    def __init__(
        self,
        source: Union[Camera, View, Tuple[Integral, Integral]],
        *,
        figsize: Optional[Tuple[Real, Real]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> None:
        self._camera = None
        self._view_image = None
        if isinstance(source, View):
            self._camera, self._view_image = source.camera, source.image
        elif isinstance(source, Camera):
            self._camera = source
        elif isinstance(source, (tuple, list)) and len(source) == 2:
            pass
        else:
            raise TypeError(
                "Input 'source' must be a Camera, a View or a tuple (height, width). "
                f"Got type {type(source)}."
            )
        if self._camera is not None:
            self._height, self._width = self._camera.sensor_height, self._camera.sensor_width
        else:
            self._height = _validate_integer(source[0], "height", min_value=1)
            self._width = _validate_integer(source[1], "width", min_value=1)

        if ax is not None:
            if figsize is not None:
                raise ValueError("Only one of 'figsize' and 'ax' can be given.")
            if not isinstance(ax, matplotlib.axes.Axes):
                raise TypeError(f"Input 'ax' must be a matplotlib Axes. Got type {type(ax)}.")
            self._figure, self._ax = ax.figure, ax
        else:
            figsize = _as_array((8, 6) if figsize is None else figsize, "figsize", numpy.float64, shape=(2,), finite=True)
            if numpy.any(figsize <= 0):
                raise ValueError(f"Input 'figsize' must be strictly positive. Got {tuple(figsize)}.")
            self._figure = plt.figure(figsize=tuple(figsize))
            self._ax = self._figure.add_subplot(1, 1, 1)
        self._colorbars: List[Any] = []
        self._setup_axes()

    def _setup_axes(self) -> None:
        # Image coordinates: pixel centers at integers, y pointing downwards
        self._ax.set_xlim(-0.5, self._width - 0.5)
        self._ax.set_ylim(self._height - 0.5, -0.5)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x (pixels)")
        self._ax.set_ylabel("y (pixels)")

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def figure(self) -> matplotlib.figure.Figure:
        r"""
        [Get] the internal matplotlib Figure object used for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The internal Figure object.
        """
        return self._figure

    @property
    def ax(self) -> matplotlib.axes.Axes:
        r"""
        [Get] the internal matplotlib Axes object used for plotting.

        Returns
        -------
        matplotlib.axes.Axes
            The internal Axes object.
        """
        return self._ax

    @property
    def camera(self) -> Optional[Camera]:
        r"""
        [Get] the camera used to project the objects, or None.

        Returns
        -------
        Optional[:class:`pysdic.Camera`]
            The camera of the plotter.
        """
        return self._camera

    @property
    def sensor_height(self) -> int:
        r"""
        [Get] the height of the sensor in pixels.

        Returns
        -------
        int
            The height of the sensor.
        """
        return self._height

    @property
    def sensor_width(self) -> int:
        r"""
        [Get] the width of the sensor in pixels.

        Returns
        -------
        int
            The width of the sensor.
        """
        return self._width

    # ===================================================================
    # Figure methods
    # ===================================================================
    def clear(self) -> None:
        r"""
        Remove all the objects, color bars and the title from the plotter.
        """
        for colorbar in self._colorbars:
            colorbar.remove()
        self._colorbars = []
        self._ax.clear()
        self._setup_axes()

    def set_title(
        self,
        title: Optional[str],
        *,
        font_size: Optional[Real] = None,
        color: Optional[Color] = None,
    ) -> None:
        r"""
        Add a title to the plotter.

        Parameters
        ----------
        title: Optional[str]
            The text of the title. If ``None``, no title is added.

        font_size: Optional[Real], optional
            The font size of the title text (strictly positive). Default is set by
            matplotlib.

        color: Optional[Union[str, Tuple[Real, ...]]], optional
            The color of the title text. Default is set by matplotlib.

        Raises
        ------
        TypeError
            If ``title`` is not a string.

        """
        if title is None:
            return
        if not isinstance(title, str):
            raise TypeError(f"Input 'title' must be a string. Got type {type(title)}.")
        font_size = _validate_real(font_size, "font_size", positive=True, finite=True, allow_none=True)
        kwargs = {key: value for key, value in (("fontsize", font_size), ("color", color)) if value is not None}
        self._ax.set_title(title, **kwargs)

    def show(self) -> None:
        r"""
        Show the figure of the plotter (with ``matplotlib.pyplot.show``).
        """
        plt.figure(self._figure.number)
        plt.show()

    def close(self) -> None:
        r"""
        Close the figure of the plotter to free its memory (the whole figure, also when
        the plotter draws in given axes).
        """
        plt.close(self._figure)

    def _add_colorbar(self, mappable: Any, bar: Dict[str, Any]) -> None:
        self._colorbars.append(self._figure.colorbar(mappable, ax=self._ax, **bar))

    # ===================================================================
    # Sensor and image
    # ===================================================================
    def add_sensor(
        self,
        *,
        color: Optional[Color] = "lightgray",
        show_pixel_grid: bool = False,
        pixel_grid_color: Color = "gray",
        pixel_grid_width: Real = 0.5,
        pixel_grid_opacity: Real = 0.5,
    ) -> None:
        r"""
        Draw the sensor of the camera and optionally the grid of the pixels.

        Parameters
        ----------
        color: Optional[Union[str, Tuple[Real, ...]]], optional (default: ``"lightgray"``)
            The color of the sensor. If None, only the pixel grid is drawn (e.g. over an
            image).

        show_pixel_grid: bool, optional (default: ``False``)
            Whether to draw the borders of the pixels.

        pixel_grid_color: Union[str, Tuple[Real, ...]], optional (default: ``"gray"``)
            The color of the pixel grid.

        pixel_grid_width: Real, optional (default: ``0.5``)
            The width of the lines of the pixel grid (strictly positive).

        pixel_grid_opacity: Real, optional (default: ``0.5``)
            The opacity of the pixel grid, between ``0.0`` and ``1.0``.

        """
        show_pixel_grid = _validate_bool(show_pixel_grid, "show_pixel_grid")
        H, W = self._height, self._width
        if color is not None:
            self._ax.add_patch(Rectangle((-0.5, -0.5), W, H, facecolor=color, edgecolor="none", zorder=0))
        if show_pixel_grid:
            pixel_grid_width = _validate_real(pixel_grid_width, "pixel_grid_width", positive=True, finite=True)
            pixel_grid_opacity = _validate_opacity(pixel_grid_opacity, "pixel_grid_opacity")
            xs = numpy.arange(W + 1) - 0.5
            ys = numpy.arange(H + 1) - 0.5
            segments = [((x, -0.5), (x, H - 0.5)) for x in xs] + [((-0.5, y), (W - 0.5, y)) for y in ys]
            self._ax.add_collection(
                LineCollection(
                    segments,
                    colors=pixel_grid_color,
                    linewidths=pixel_grid_width,
                    alpha=pixel_grid_opacity,
                    zorder=1,
                )
            )

    def add_image(
        self,
        image: Optional[Union[Image, ArrayLike]] = None,
        *,
        channel: Optional[Integral] = None,
        bgr: bool = True,
        cmap: Optional[Union[str, matplotlib.colors.Colormap]] = "gray",
        clim: Optional[Tuple[Real, Real]] = None,
        opacity: Real = 1.0,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""
        Draw an image on the sensor, with its pixels centered at integer coordinates.

        - A grayscale image :math:`(H, W)` or :math:`(H, W, 1)` (or one channel of an
          image, selected with ``channel``) is drawn with the colormap ``cmap`` and the
          color limits ``clim``.
        - A color image :math:`(H, W, 3)` or :math:`(H, W, 4)` is drawn in RGB(A). Its
          values are rescaled from ``clim`` to :math:`[0, 1]`, except for
          :obj:`numpy.uint8` images displayed as they are when ``clim`` is None.

        Parameters
        ----------
        image: Optional[Union[:class:`pysdic.Image`, ArrayLike]], optional
            The image to draw, of height and width equal to the sensor size: an
            :class:`pysdic.Image` or an array (e.g. a map of residuals). If None, the
            image of the view given to the plotter is used.

        channel: Optional[Integral], optional
            The index of a channel to draw as a grayscale image. By default, all the
            channels are drawn.

        bgr: bool, optional (default: ``True``)
            Whether the channels of a color image are in BGR(A) order (as read by
            :meth:`pysdic.Image.from_file` with OpenCV). They are then converted to RGB(A).

        cmap: Optional[Union[str, :class:`matplotlib.colors.Colormap`]], optional (default: ``"gray"``)
            The colormap of a grayscale image.

        clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, the minimum and the maximum of the
            finite values of the image (except for :obj:`numpy.uint8` color images).

        opacity: Real, optional (default: ``1.0``)
            The opacity of the image, between ``0.0`` and ``1.0``.

        show_bar: bool, optional (default: ``False``)
            Whether to show the color bar of a grayscale image.

        bar: Optional[Dict[str, Any]], optional
            The keyword arguments of ``matplotlib.figure.Figure.colorbar`` when
            ``show_bar`` is True.

        Raises
        ------
        TypeError
            If ``image`` has a wrong type.

        ValueError
            If no image is given without view, if the image does not match the sensor
            size, has a wrong number of channels, or if ``clim`` is invalid.

        IndexError
            If ``channel`` is out of range.

        """
        if image is None:
            if self._view_image is None:
                raise ValueError("Input 'image' is required when the plotter is not built from a View.")
            image = self._view_image
        if isinstance(image, Image):
            if image.image is None:
                raise ValueError("The given Image has no data.")
            data = image.image
        else:
            data = numpy.asarray(image)
            if not (numpy.issubdtype(data.dtype, numpy.integer) or numpy.issubdtype(data.dtype, numpy.floating)):
                raise TypeError(f"Input 'image' must have a numeric dtype. Got {data.dtype}.")
        if data.ndim not in (2, 3) or data.shape[:2] != (self._height, self._width):
            raise ValueError(
                f"Input 'image' must have shape ({self._height}, {self._width}) or "
                f"({self._height}, {self._width}, C). Got shape {data.shape}."
            )
        bgr = _validate_bool(bgr, "bgr")
        show_bar = _validate_bool(show_bar, "show_bar")
        bar = _validate_kwargs(bar, "bar")
        opacity = _validate_opacity(opacity, "opacity")

        if data.ndim == 3:
            channel = _validate_integer(channel, "channel", min_value=0, allow_none=True)
            if channel is not None:
                if channel >= data.shape[2]:
                    raise IndexError(f"Input 'channel' must be in [0, {data.shape[2] - 1}]. Got {channel}.")
                data = data[:, :, channel]
            elif data.shape[2] == 1:
                data = data[:, :, 0]
            elif data.shape[2] not in (3, 4):
                raise ValueError(
                    f"A color image must have 3 or 4 channels. Got {data.shape[2]}: use "
                    "'channel' to draw one channel."
                )
        elif channel is not None:
            raise ValueError("Input 'channel' can only be given for an image with channels.")

        extent = (-0.5, self._width - 0.5, self._height - 0.5, -0.5)
        if data.ndim == 2:
            vmin, vmax = _validate_clim(clim, data)
            mappable = self._ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=opacity,
                extent=extent, origin="upper", interpolation="nearest", zorder=0,
            )
            if show_bar:
                self._add_colorbar(mappable, bar)
            return

        if bgr:
            data = data[:, :, [2, 1, 0, 3][: data.shape[2]]]
        if not (clim is None and data.dtype == numpy.uint8):
            vmin, vmax = _validate_clim(clim, data)
            data = numpy.clip((data.astype(numpy.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
        self._ax.imshow(
            data, alpha=opacity, extent=extent, origin="upper", interpolation="nearest", zorder=0
        )

    # ===================================================================
    # Geometrical objects
    # ===================================================================
    def _to_image_points(self, coordinates: numpy.ndarray, projected: Optional[bool]) -> numpy.ndarray:
        # Image points (N, 2) of points (N, E): projected with the camera or used directly
        projected = _validate_bool(projected, "projected") if projected is not None else None
        E = coordinates.shape[1]
        if projected is None:
            projected = self._camera is not None and E == self._camera.input_dim
        if projected:
            if self._camera is None:
                raise ValueError("A camera is required to project the points: build the plotter from a Camera or a View.")
            if E != self._camera.input_dim:
                raise ValueError(
                    f"The points must have the input dimension of the camera ({self._camera.input_dim}) "
                    f"to be projected. Got E={E}."
                )
            return self._camera.project(coordinates).image_points
        if E != 2:
            raise ValueError(
                "The points must be 2D image points (x, y) when they are not projected "
                f"(camera input dimension: {None if self._camera is None else self._camera.input_dim}). Got E={E}."
            )
        return coordinates

    def _add_arrows(
        self,
        origins: numpy.ndarray,
        vectors: numpy.ndarray,
        mask: numpy.ndarray,
        scale: Union[Real, ArrayLike],
        color: Color,
    ) -> None:
        # Draw the image vectors (N, 2) times their scale (scalar or (N,)) from the origins (N, 2)
        scale = _as_scalar_or_1d_array(scale, "vector_scale", vectors.shape[0], numpy.float64, finite=True)
        if numpy.any(scale <= 0):
            raise ValueError(f"Input 'vector_scale' must be strictly positive. Got a minimum of {scale.min()}.")
        keep = mask & numpy.isfinite(origins).all(axis=1) & numpy.isfinite(vectors).all(axis=1)
        if not numpy.any(keep):
            raise ValueError("No finite vectors available for visualization.")
        scaled = vectors[keep] * scale[keep, numpy.newaxis]
        self._ax.quiver(
            origins[keep, 0], origins[keep, 1], scaled[:, 0], scaled[:, 1],
            angles="xy", scale_units="xy", scale=1.0, color=color, zorder=4,
        )

    def _add_index_labels(
        self,
        image_points: numpy.ndarray,
        mask: numpy.ndarray,
        color: Color,
        font_size: Real,
    ) -> None:
        # Write the index of each selected finite image point (N, 2) next to it. The font size and
        # the offset are in typographic points: the labels keep their size when zooming
        font_size = _validate_real(font_size, "index_font_size", positive=True, finite=True)
        keep = mask & numpy.isfinite(image_points).all(axis=1)
        for index in numpy.flatnonzero(keep):
            self._ax.annotate(
                str(index),
                xy=(image_points[index, 0], image_points[index, 1]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=font_size,
                color=color,
                zorder=5,
                annotation_clip=True,  # Only the labels of the points inside the axes are drawn
            )

    def add_point_cloud(
        self,
        point_cloud: Union[PointCloud, ArrayLike],
        *,
        projected: Optional[bool] = None,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        color: Color = "black",
        size: Real = 5.0,
        opacity: Real = 1.0,
        mask: Optional[ArrayLike] = None,
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Color] = None,
        property_above_color: Optional[Color] = None,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Color = "red",
        show_points_index: bool = False,
        index_color: Color = "black",
        index_font_size: Real = 8,
    ) -> None:
        r"""
        Add a point cloud to the plotter (projected or drawn as image points, see
        :class:`pysdic.ImagePlotter`).

        Only the finite image points selected by ``mask`` are displayed.

        Parameters
        ----------
        point_cloud: Union[:class:`pysdic.PointCloud`, ArrayLike]
            The point cloud to display, or an array of shape :math:`(N_p, E)`.

        projected: Optional[bool], optional
            Whether to project the points with the camera (see :class:`pysdic.ImagePlotter`).
            By default, chosen with the dimension of the points.

        warp_by: Optional[Union[str, ArrayLike]], optional
            The displacement added to the points before the projection: the name of a
            property of the point cloud or an array, of shape :math:`(N_p, E)`.

        color: Union[str, Tuple[Real, ...]], optional (default: ``"black"``)
            The color of the points. Ignored if ``property`` is given.

        size: Real, optional (default: ``5.0``)
            The size of the points (strictly positive, in points squared as
            ``matplotlib.axes.Axes.scatter``).

        opacity: Real, optional (default: ``1.0``)
            The opacity of the points, between ``0.0`` and ``1.0``.

        mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_p,)`, True for the points to display.

        property: Optional[Union[str, ArrayLike]], optional
            The property used to color the points: the name of a property of the point
            cloud, or an array of shape :math:`(N_p,)` or :math:`(N_p, A)`.

        property_axis: Optional[Integral], optional
            The component used to color the points when the property has :math:`A > 1`
            components. By default, the norm of the property vector is used.

        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, they are set automatically.

        property_cmap: Optional[Union[str, :class:`matplotlib.colors.Colormap`]], optional
            The colormap (name or object). Matplotlib default: ``"viridis"``.

        property_below_color: Optional[Union[str, Tuple[Real, ...]]], optional
            The color of the values below ``property_clim``. By default, the first
            color of the colormap.

        property_above_color: Optional[Union[str, Tuple[Real, ...]]], optional
            The color of the values above ``property_clim``. By default, the last
            color of the colormap.

        show_bar: bool, optional (default: ``False``)
            Whether to show the color bar when ``property`` is given.

        bar: Optional[Dict[str, Any]], optional
            The keyword arguments of ``matplotlib.figure.Figure.colorbar``.

        vectors: Optional[Union[str, ArrayLike]], optional
            A vector field **in the image plane** (pixels) drawn as arrows from the
            displayed points: the name of a property of the point cloud or an array, of
            shape :math:`(N_p, 2)`. The points with non-finite vectors have no arrow.

        vector_scale: Union[Real, ArrayLike], optional (default: ``1.0``)
            The scale factor of the arrows (strictly positive and finite): a scalar or a
            1D array of shape :math:`(N_p,)`.

        vector_color: Union[str, Tuple[Real, ...]], optional (default: ``"red"``)
            The color of the arrows.

        show_points_index: bool, optional (default: ``False``)
            Whether to write the index of each displayed point next to it (the index in
            the point cloud, also with a ``mask``).

        index_color: Union[str, Tuple[Real, ...]], optional (default: ``"black"``)
            The color of the indices.

        index_font_size: Real, optional (default: ``8``)
            The font size of the indices in typographic points (strictly positive): the
            labels keep their size when zooming.

        Raises
        ------
        TypeError
            If ``point_cloud`` has a wrong type.

        ValueError
            If the point cloud is empty, if the points cannot be projected or drawn, if
            an input has a wrong shape, or if no point can be displayed.

        IndexError
            If ``property_axis`` is out of range.

        """
        if not isinstance(point_cloud, PointCloud):
            point_cloud = PointCloud(_as_array(point_cloud, "point_cloud", numpy.float64, shape=(None, None), finite=False))
        if point_cloud.n_points == 0:
            raise ValueError("Cannot visualize an empty point cloud.")
        size = _validate_real(size, "size", positive=True, finite=True)
        opacity = _validate_opacity(opacity, "opacity")
        show_bar = _validate_bool(show_bar, "show_bar")
        bar = _validate_kwargs(bar, "bar")
        show_points_index = _validate_bool(show_points_index, "show_points_index")
        N = point_cloud.n_points

        image_points = self._to_image_points(_warped_coordinates(point_cloud, warp_by, "point cloud"), projected)
        display_mask = _as_bool_mask(mask, N) if mask is not None else numpy.ones(N, dtype=bool)
        display_mask = display_mask & numpy.isfinite(image_points).all(axis=1)
        property_values = None
        if property is not None:
            property_values = _scalar_values(
                _resolve_values(point_cloud, property, N, "property", "point cloud"), property_axis
            )
        if not numpy.any(display_mask):
            raise ValueError("No finite points available for visualization.")

        mappable = self._ax.scatter(
            image_points[display_mask, 0],
            image_points[display_mask, 1],
            s=size,
            alpha=opacity,
            zorder=3,
            **_color_kwargs(
                color, property_values, display_mask, property_clim, property_cmap,
                property_below_color, property_above_color, "c",
            ),
        )
        if property_values is not None and show_bar:
            self._add_colorbar(mappable, bar)

        if vectors is not None:
            vector_values = _resolve_image_vectors(point_cloud, vectors, N, "vectors", "point cloud")
            self._add_arrows(image_points, vector_values, display_mask, vector_scale, vector_color)

        if show_points_index:
            self._add_index_labels(image_points, display_mask, index_color, index_font_size)

    def add_mesh(
        self,
        mesh: Mesh,
        *,
        projected: Optional[bool] = None,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        vertex_color: Color = "black",
        vertex_size: Real = 5.0,
        vertex_opacity: Real = 1.0,
        vertex_mask: Optional[ArrayLike] = None,
        show_vertices: bool = True,
        edge_color: Color = "black",
        edge_width: Real = 1.0,
        edge_opacity: Real = 1.0,
        show_edges: bool = True,
        face_color: Color = "lightblue",
        face_opacity: Real = 0.5,
        face_mask: Optional[ArrayLike] = None,
        show_faces: bool = True,
        vertex_property: Optional[Union[str, ArrayLike]] = None,
        element_property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Color] = None,
        property_above_color: Optional[Color] = None,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vertex_vectors: Optional[Union[str, ArrayLike]] = None,
        element_vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Color = "red",
        show_vertices_index: bool = False,
        show_elements_index: bool = False,
        vertex_index_color: Color = "black",
        element_index_color: Color = "blue",
        index_font_size: Real = 8,
    ) -> None:
        r"""
        Add a mesh to the plotter (projected or drawn in image points, see
        :class:`pysdic.ImagePlotter`).

        The element type of the mesh must be set. The faces are the polygons through the
        nodes of the contour of the elements (corner and mid-edge nodes). The segment
        elements are drawn as edges only (no faces). The faces are displayed if they are
        selected by ``face_mask`` and all their vertices are finite (after projection).
        The vertices are displayed if they are selected by ``vertex_mask`` and finite.

        Parameters
        ----------
        mesh: :class:`pysdic.Mesh`
            The mesh to display.

        projected: Optional[bool], optional
            Whether to project the vertices with the camera (see
            :class:`pysdic.ImagePlotter`). By default, chosen with the dimension of the
            vertices.

        warp_by: Optional[Union[str, ArrayLike]], optional
            The displacement added to the vertices before the projection: the name of a
            vertex property or an array, of shape :math:`(N_v, E)`.

        vertex_color, vertex_size, vertex_opacity:
            The color, size (strictly positive) and opacity (in :math:`[0, 1]`) of the
            vertices.

        vertex_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_v,)`, True for the vertices to display.
            The faces are not affected.

        show_vertices: bool, optional (default: ``True``)
            Whether to show the vertices.

        edge_color, edge_width, edge_opacity:
            The color, width (strictly positive) and opacity (in :math:`[0, 1]`) of the
            edges.

        show_edges: bool, optional (default: ``True``)
            Whether to show the edges of the displayed faces.

        face_color: Union[str, Tuple[Real, ...]], optional (default: ``"lightblue"``)
            The color of the faces. Ignored if a property is given.

        face_opacity: Real, optional (default: ``0.5``)
            The opacity of the faces, in :math:`[0, 1]`.

        face_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_e,)`, True for the faces to display.

        show_faces: bool, optional (default: ``True``)
            Whether to show the faces.

        vertex_property: Optional[Union[str, ArrayLike]], optional
            The property used to color the faces (interpolated between the nodes of the
            contour): the name of a vertex property, or an array of shape :math:`(N_v,)`
            or :math:`(N_v, A)`.

        element_property: Optional[Union[str, ArrayLike]], optional
            The property used to color the faces (one color per face): the name of an
            element property, or an array of shape :math:`(N_e,)` or :math:`(N_e, A)`.
            Only one of ``vertex_property`` and ``element_property`` can be given.

        property_axis, property_clim, property_cmap, property_below_color, property_above_color, show_bar, bar:
            See :meth:`add_point_cloud`.

        vertex_vectors: Optional[Union[str, ArrayLike]], optional
            A vector field **in the image plane** drawn as arrows from the vertices
            selected by ``vertex_mask``: the name of a vertex property or an array of
            shape :math:`(N_v, 2)`.

        element_vectors: Optional[Union[str, ArrayLike]], optional
            A vector field **in the image plane** drawn as arrows from the (projected)
            centers of the elements selected by ``face_mask`` (see
            :meth:`pysdic.Mesh.compute_elements_centers`): the name of an element
            property or an array of shape :math:`(N_e, 2)`. Only one of
            ``vertex_vectors`` and ``element_vectors`` can be given.

        vector_scale: Union[Real, ArrayLike], optional (default: ``1.0``)
            The scale factor of the arrows: a scalar or a 1D array with one value per row
            of the vector field (:math:`N_v` or :math:`N_e`).

        vector_color: Union[str, Tuple[Real, ...]], optional (default: ``"red"``)
            The color of the arrows.

        show_vertices_index: bool, optional (default: ``False``)
            Whether to write the index of each vertex selected by ``vertex_mask`` next to
            it (even if ``show_vertices`` is False).

        show_elements_index: bool, optional (default: ``False``)
            Whether to write the index of each element selected by ``face_mask`` at its
            (projected) center (see :meth:`pysdic.Mesh.compute_elements_centers`), even if
            ``show_faces`` is False.

        vertex_index_color: Union[str, Tuple[Real, ...]], optional (default: ``"black"``)
            The color of the indices of the vertices.

        element_index_color: Union[str, Tuple[Real, ...]], optional (default: ``"blue"``)
            The color of the indices of the elements.

        index_font_size: Real, optional (default: ``8``)
            The font size of the indices in typographic points (strictly positive): the
            labels keep their size when zooming.

        Raises
        ------
        TypeError
            If ``mesh`` is not a :class:`pysdic.Mesh`.

        ValueError
            If the mesh is empty or has no element type, if the vertices cannot be
            projected or drawn, if an input has a wrong shape, or if nothing can be
            displayed.

        IndexError
            If ``property_axis`` is out of range.

        """
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        if mesh.n_vertices == 0 or mesh.n_elements == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.element_type not in _CONTOUR_NODES:
            raise ValueError(
                f"Cannot visualize a mesh with element type {mesh.element_type!r}. "
                "Set 'mesh.element_type' first."
            )
        show_vertices = _validate_bool(show_vertices, "show_vertices")
        show_edges = _validate_bool(show_edges, "show_edges")
        show_faces = _validate_bool(show_faces, "show_faces")
        show_bar = _validate_bool(show_bar, "show_bar")
        bar = _validate_kwargs(bar, "bar")
        show_vertices_index = _validate_bool(show_vertices_index, "show_vertices_index")
        show_elements_index = _validate_bool(show_elements_index, "show_elements_index")
        if vertex_property is not None and element_property is not None:
            raise ValueError("Only one of 'vertex_property' and 'element_property' can be given.")
        if vertex_vectors is not None and element_vectors is not None:
            raise ValueError("Only one of 'vertex_vectors' and 'element_vectors' can be given.")
        N_v, N_e = mesh.n_vertices, mesh.n_elements
        is_segment = mesh.n_topological_dimensions == 1
        contours = mesh.elements[:, _CONTOUR_NODES[mesh.element_type]]  # (N_e, N_c)

        coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")  # (N_v, E)
        image_points = self._to_image_points(coordinates, projected)  # (N_v, 2)
        finite_vertices = numpy.isfinite(image_points).all(axis=1)

        vertex_display = _as_bool_mask(vertex_mask, N_v, "vertex_mask") if vertex_mask is not None else numpy.ones(N_v, dtype=bool)
        vertex_display = vertex_display & finite_vertices
        face_display = _as_bool_mask(face_mask, N_e, "face_mask") if face_mask is not None else numpy.ones(N_e, dtype=bool)
        face_display = face_display & finite_vertices[mesh.elements].all(axis=1)

        if (show_faces or show_edges) and not numpy.any(face_display):
            raise ValueError("No face with finite vertices available for visualization.")
        polygons = image_points[contours[face_display]]  # (N_f, N_c, 2)

        if show_faces and not is_segment:
            face_kwargs = dict(alpha=_validate_opacity(face_opacity, "face_opacity"), zorder=2)
            mappable = None
            if vertex_property is not None:
                vertex_values = _scalar_values(
                    _resolve_values(mesh.vertices, vertex_property, N_v, "vertex_property", "mesh vertices"),
                    property_axis,
                )
                # Fan triangulation of the contour of each face, colors interpolated (Gouraud)
                n_c = contours.shape[1]
                fan = contours[face_display][:, [[0, i, i + 1] for i in range(1, n_c - 1)]].reshape(-1, 3)
                used, fan = numpy.unique(fan, return_inverse=True)  # Only the finite vertices of the faces
                mappable = self._ax.tripcolor(
                    image_points[used, 0], image_points[used, 1], fan.reshape(-1, 3), vertex_values[used],
                    shading="gouraud", zorder=2,
                    **_color_kwargs(
                        None, vertex_values, None, property_clim, property_cmap,
                        property_below_color, property_above_color, None,
                    ),
                )
                # The opacity is applied to the rendered faces as a whole: with an alpha per
                # triangle, the borders of the triangles of the fans would be visible
                if face_kwargs["alpha"] < 1.0:
                    mappable.set_agg_filter(_OpacityFilter(face_kwargs["alpha"]))
            elif element_property is not None:
                element_values = _scalar_values(
                    _resolve_values(mesh.connectivity, element_property, N_e, "element_property", "mesh connectivity"),
                    property_axis,
                )
                mappable = PolyCollection(
                    polygons, edgecolors="none", **face_kwargs,
                    **_color_kwargs(
                        None, element_values, face_display, property_clim, property_cmap,
                        property_below_color, property_above_color, "array",
                    ),
                )
                self._ax.add_collection(mappable)
            else:
                self._ax.add_collection(PolyCollection(polygons, facecolors=face_color, edgecolors="none", **face_kwargs))
            if mappable is not None and show_bar:
                self._add_colorbar(mappable, bar)

        if show_edges:
            edge_width = _validate_real(edge_width, "edge_width", positive=True, finite=True)
            edge_opacity = _validate_opacity(edge_opacity, "edge_opacity")
            edges = LineCollection(
                [polygon for polygon in polygons] if is_segment else [numpy.vstack((p, p[:1])) for p in polygons],
                colors=edge_color, linewidths=edge_width, alpha=edge_opacity, zorder=2.5,
            )
            self._ax.add_collection(edges)

        if show_vertices:
            if not numpy.any(vertex_display):
                raise ValueError("No finite vertices available for visualization.")
            vertex_size = _validate_real(vertex_size, "vertex_size", positive=True, finite=True)
            vertex_opacity = _validate_opacity(vertex_opacity, "vertex_opacity")
            self._ax.scatter(
                image_points[vertex_display, 0], image_points[vertex_display, 1],
                s=vertex_size, color=vertex_color, alpha=vertex_opacity, zorder=3,
            )

        if vertex_vectors is not None:
            vector_values = _resolve_image_vectors(mesh.vertices, vertex_vectors, N_v, "vertex_vectors", "mesh vertices")
            self._add_arrows(image_points, vector_values, vertex_display, vector_scale, vector_color)

        if element_vectors is not None:
            vector_values = _resolve_image_vectors(
                mesh.connectivity, element_vectors, N_e, "element_vectors", "mesh connectivity"
            )
            centers = fem.compute_elements_centers(coordinates, mesh.elements, mesh.element_type)  # (N_e, E)
            self._add_arrows(self._to_image_points(centers, projected), vector_values, face_display, vector_scale, vector_color)

        if show_vertices_index:
            self._add_index_labels(image_points, vertex_display, vertex_index_color, index_font_size)

        if show_elements_index:
            centers = fem.compute_elements_centers(coordinates, mesh.elements, mesh.element_type)  # (N_e, E)
            self._add_index_labels(
                self._to_image_points(centers, projected), face_display, element_index_color, index_font_size
            )

    def add_integration_points(
        self,
        integration_points: IntegrationPoints,
        mesh: Mesh,
        *,
        projected: Optional[bool] = None,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        color: Color = "black",
        size: Real = 5.0,
        opacity: Real = 1.0,
        mask: Optional[ArrayLike] = None,
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Color] = None,
        property_above_color: Optional[Color] = None,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Color = "red",
        show_points_index: bool = False,
        index_color: Color = "black",
        index_font_size: Real = 8,
    ) -> None:
        r"""
        Add integration points to the plotter, at their global coordinates in a mesh
        (see :meth:`pysdic.Mesh.compute_property_interpolation`), projected or drawn as
        image points (see :class:`pysdic.ImagePlotter`).

        The unlocated points (element index ``-1``) are not displayed.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points to display, consistent with ``mesh`` (see
            :meth:`pysdic.Mesh.validate_integration_points`).

        mesh: :class:`pysdic.Mesh`
            The mesh in which the points are located.

        projected: Optional[bool], optional
            See :meth:`add_point_cloud`.

        warp_by: Optional[Union[str, ArrayLike]], optional
            The displacement added to the vertices of the mesh before computing the
            coordinates of the points: the name of a vertex property or an array, of
            shape :math:`(N_v, E)`.

        color, size, opacity:
            See :meth:`add_point_cloud`.

        mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_p,)`, True for the points to display.

        property: Optional[Union[str, ArrayLike]], optional
            The property used to color the points: the name of a property of the
            integration points, or an array of shape :math:`(N_p,)` or :math:`(N_p, A)`.

        property_axis, property_clim, property_cmap, property_below_color, property_above_color, show_bar, bar:
            See :meth:`add_point_cloud`.

        vectors: Optional[Union[str, ArrayLike]], optional
            A vector field in the image plane: the name of a property of the integration
            points or an array of shape :math:`(N_p, 2)`.

        vector_scale, vector_color:
            See :meth:`add_point_cloud`.

        show_points_index, index_color, index_font_size:
            See :meth:`add_point_cloud` (the index of each point in ``integration_points``).

        Raises
        ------
        TypeError
            If ``integration_points`` is not a :class:`pysdic.IntegrationPoints` or
            ``mesh`` is not a :class:`pysdic.Mesh`.

        ValueError
            If the points are not consistent with the mesh, or see :meth:`add_point_cloud`.

        """
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        mesh.validate_integration_points(integration_points)
        locations = mesh.compute_property_interpolation(
            _warped_coordinates(mesh.vertices, warp_by, "mesh vertices"),
            integration_points,
            allow_unlocated_points=True,
        )  # (N_p, E), NaN for the unlocated points
        point_cloud = PointCloud(locations, n_dimensions=mesh.n_dimensions, properties=integration_points.properties)
        self.add_point_cloud(
            point_cloud,
            projected=projected,
            color=color,
            size=size,
            opacity=opacity,
            mask=mask,
            property=property,
            property_axis=property_axis,
            property_clim=property_clim,
            property_cmap=property_cmap,
            property_below_color=property_below_color,
            property_above_color=property_above_color,
            show_bar=show_bar,
            bar=bar,
            vectors=vectors,
            vector_scale=vector_scale,
            vector_color=vector_color,
            show_points_index=show_points_index,
            index_color=index_color,
            index_font_size=index_font_size,
        )

    def add_rectangle(
        self,
        center: ArrayLike,
        width: Real,
        height: Real,
        *,
        angle: Real = 0.0,
        face_color: Color = "lightblue",
        face_opacity: Real = 1.0,
        show_face: bool = True,
        edge_color: Color = "black",
        edge_width: Real = 1.0,
        edge_opacity: Real = 1.0,
        show_edges: bool = True,
    ) -> None:
        r"""
        Draw a rectangle in the image plane (e.g. a subset of DIC).

        Parameters
        ----------
        center: ArrayLike
            The image coordinates :math:`(x, y)` of the center of the rectangle.

        width: Real
            The width of the rectangle along :math:`x` before rotation (strictly positive).

        height: Real
            The height of the rectangle along :math:`y` before rotation (strictly positive).

        angle: Real, optional (default: ``0.0``)
            The rotation angle in degrees around the center, from the :math:`x`-axis
            towards the :math:`y`-axis (clockwise on the screen, as :math:`y` points
            downwards).

        face_color, face_opacity, show_face:
            The color, opacity (in :math:`[0, 1]`) and visibility of the face.

        edge_color, edge_width, edge_opacity, show_edges:
            The color, width (strictly positive), opacity (in :math:`[0, 1]`) and
            visibility of the edges.

        """
        center = _as_array(center, "center", numpy.float64, shape=(2,), finite=True)
        width = _validate_real(width, "width", positive=True, finite=True)
        height = _validate_real(height, "height", positive=True, finite=True)
        angle = _validate_real(angle, "angle", finite=True)
        show_face = _validate_bool(show_face, "show_face")
        show_edges = _validate_bool(show_edges, "show_edges")
        face_rgba = matplotlib.colors.to_rgba(face_color, _validate_opacity(face_opacity, "face_opacity"))
        edge_rgba = matplotlib.colors.to_rgba(edge_color, _validate_opacity(edge_opacity, "edge_opacity"))
        edge_width = _validate_real(edge_width, "edge_width", positive=True, finite=True)
        self._ax.add_patch(
            Rectangle(
                (center[0] - width / 2, center[1] - height / 2),
                width,
                height,
                angle=angle,
                rotation_point="center",
                facecolor=face_rgba if show_face else "none",
                edgecolor=edge_rgba if show_edges else "none",
                linewidth=edge_width if show_edges else 0.0,
                zorder=2,
            )
        )

    def add_roi(
        self,
        roi: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
        *,
        face_color: Color = "orange",
        face_opacity: Real = 0.2,
        show_face: bool = False,
        edge_color: Color = "orange",
        edge_width: Real = 2.0,
        edge_opacity: Real = 1.0,
        show_edges: bool = True,
    ) -> None:
        r"""
        Draw a region of interest (ROI) along the borders of its pixels.

        The ROI ``(x, y, width, height)`` contains the columns ``x, ..., x + width - 1`` and
        the rows ``y, ..., y + height - 1`` (see :attr:`pysdic.Image.roi`): it is drawn from
        :math:`x - 0.5` to :math:`x + width - 0.5` and from :math:`y - 0.5` to
        :math:`y + height - 0.5`.

        .. seealso::

            :func:`pysdic.compute_projected_roi` to compute the ROI of a projected mesh.

        Parameters
        ----------
        roi: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
            The ROI ``(x, y, width, height)`` in pixels, inside the sensor. If None, the ROI
            of the image of the view given to the plotter is used (the whole sensor if this
            image has no ROI).

        face_color, face_opacity, show_face:
            The color, opacity (in :math:`[0, 1]`) and visibility of the face. By default,
            the face is not drawn.

        edge_color, edge_width, edge_opacity, show_edges:
            The color, width (strictly positive), opacity (in :math:`[0, 1]`) and
            visibility of the edges.

        Raises
        ------
        TypeError
            If ``roi`` is not a tuple of 4 integers.

        ValueError
            If no ROI is given without view, or if the ROI is out of the sensor.

        """
        if roi is None:
            if self._view_image is None:
                raise ValueError("Input 'roi' is required when the plotter is not built from a View.")
            roi = self._view_image.roi
            if roi is None:
                roi = (0, 0, self._width, self._height)
        try:
            roi = tuple(roi)
        except TypeError:
            raise TypeError(f"Input 'roi' must be a tuple (x, y, width, height). Got type {type(roi)}.") from None
        if len(roi) != 4:
            raise ValueError(f"Input 'roi' must be a tuple (x, y, width, height). Got {len(roi)} values.")
        x = _validate_integer(roi[0], "roi x", min_value=0)
        y = _validate_integer(roi[1], "roi y", min_value=0)
        width = _validate_integer(roi[2], "roi width", min_value=1)
        height = _validate_integer(roi[3], "roi height", min_value=1)
        if x + width > self._width or y + height > self._height:
            raise ValueError(
                f"ROI {(x, y, width, height)} must be within the sensor "
                f"(width={self._width}, height={self._height})."
            )
        self.add_rectangle(
            (x + (width - 1) / 2, y + (height - 1) / 2),
            width,
            height,
            face_color=face_color,
            face_opacity=face_opacity,
            show_face=show_face,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_opacity=edge_opacity,
            show_edges=show_edges,
        )