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
from typing import TYPE_CHECKING, Optional, Union, Tuple, Any, Dict
from numpy.typing import ArrayLike
from numbers import Integral, Real

import numpy

from .mesh import Mesh
from .point_cloud import PointCloud
from .integration_points import IntegrationPoints
from .. import fem
from .._optionals import import_optional, RequiresOptionalMixin
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _validate_kwargs,
    _as_scalar_or_1d_array,
    _as_array,
    _as_bool_mask,
)

if TYPE_CHECKING:
    import matplotlib.colors
    import pyvista


# =============================================================================
# Internal helpers
# =============================================================================
def _as_3d(coordinates: numpy.ndarray) -> numpy.ndarray:
    # Complete coordinates of shape (N, E) with E <= 3 with zeros to get shape (N, 3)
    E = coordinates.shape[1]
    if E == 3:
        return coordinates
    return numpy.hstack((coordinates, numpy.zeros((coordinates.shape[0], 3 - E), dtype=coordinates.dtype)))


def _warped_coordinates(point_cloud: PointCloud, warp_by: Optional[Union[str, ArrayLike]], location: str) -> numpy.ndarray:
    # Coordinates (N, E) of a point cloud displaced by a property name or an array of shape (N, E)
    coordinates = point_cloud.coordinates
    if warp_by is None:
        return coordinates
    if isinstance(warp_by, str):
        if warp_by not in point_cloud:
            raise ValueError(f"Warping property '{warp_by}' not found in the {location}.")
        warp_values = point_cloud[warp_by]
        if warp_values.shape != point_cloud.shape:
            raise ValueError(
                f"Warping property '{warp_by}' must have shape {point_cloud.shape}. "
                f"Got {warp_values.shape}."
            )
    else:
        warp_values = _as_array(
            warp_by, "warp_by", numpy.float64, shape=point_cloud.shape, finite=False
        )
    return coordinates + warp_values


def _resolve_values(container: Any, values: Union[str, ArrayLike], n_rows: int, name: str, location: str) -> numpy.ndarray:
    # Values (n_rows, A) given by the name of a property of 'container' or an array (n_rows,) / (n_rows, A)
    if isinstance(values, str):
        if values not in container:
            raise ValueError(f"Property '{values}' not found in the {location}.")
        values = container[values]
    return _as_array(
        values, name, numpy.float64, shape=(n_rows, None), shape_hint=f"({n_rows},) or ({n_rows}, A)",
        finite=False, accept_1d_as_column=True,
    )


def _resolve_vectors(
    container: Any, vectors: Union[str, ArrayLike], n_rows: int, n_dimensions: int, name: str, location: str
) -> numpy.ndarray:
    # Vectors (n_rows, 3) given by the name of a property of 'container' or an array (n_rows, 3)
    # or (n_rows, E): vectors with E < 3 components are completed with zeros
    if isinstance(vectors, str):
        if vectors not in container:
            raise ValueError(f"Property '{vectors}' not found in the {location}.")
        vectors = container[vectors]
    vectors = _as_array(vectors, name, numpy.float64, shape=(n_rows, None), finite=False)
    if vectors.shape[1] not in (3, n_dimensions):
        raise ValueError(
            f"Input '{name}' must have shape ({n_rows}, 3) or ({n_rows}, {n_dimensions}). "
            f"Got shape {vectors.shape}."
        )
    return _as_3d(vectors)


def _scalar_values(values: numpy.ndarray, property_axis: Optional[Integral]) -> numpy.ndarray:
    # Scalar values (N,) from values (N, A): one component, or the norm if property_axis is None
    property_axis = _validate_integer(property_axis, "property_axis", min_value=0, allow_none=True)
    if property_axis is None:
        # A scalar property keeps its sign, the norm is only used for vectors (A > 1)
        return values[:, 0] if values.shape[1] == 1 else numpy.linalg.norm(values, axis=1)
    if property_axis >= values.shape[1]:
        raise IndexError(
            f"Input 'property_axis' must be in [0, {values.shape[1] - 1}]. Got {property_axis}."
        )
    return values[:, property_axis]


def _as_color_texture(texture: ArrayLike, texture_rgb: bool) -> numpy.ndarray:
    # Texture image for PyVista: gray levels are repeated on 3 channels
    texture = numpy.asarray(texture)
    if texture.ndim == 2:
        return numpy.repeat(texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
    if texture.ndim == 3 and texture.shape[2] == 1:
        return numpy.repeat(texture, 3, axis=2).astype(numpy.uint8)
    if texture.ndim == 3 and texture.shape[2] in (3, 4):
        if texture_rgb:
            return texture
        gray_texture = numpy.round(numpy.dot(texture[..., :3], [0.2989, 0.5870, 0.1140])).astype(numpy.uint8)
        return numpy.repeat(gray_texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
    raise ValueError(
        f"Input 'texture' must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4). "
        f"Got shape {texture.shape}."
    )


class Plotter3D(RequiresOptionalMixin):
    r"""
    A plotter for the geometrical objects of pysdic (:class:`pysdic.PointCloud`,
    :class:`pysdic.Mesh` and :class:`pysdic.IntegrationPoints`), based on ``pyvista``.

    .. important::

        Requires the optional package ``pyvista`` (``pip install pyvista``): an
        ``ImportError`` is raised when the class is instantiated without it. Use
        ``Plotter3D.is_available()`` to check it.

    """

    _required_optionals = ("pyvista",)

    def __init__(self) -> None:
        pyvista = import_optional("pyvista", "Plotter3D")
        self._plotter = pyvista.Plotter()

    @property
    def plotter(self) -> pyvista.Plotter:
        r"""
        [Get] the internal pyvista.Plotter object used for plotting.

        Returns
        -------
        plotter: pyvista.Plotter
            The internal pyvista.Plotter object.
        """
        return self._plotter

    def clear(self) -> None:
        r"""
        Clear the plotter of all geometrical objects and reset the view.

        This method removes all geometrical objects from the plotter and resets the camera view to the default position.

        Returns
        -------
        None
        """
        self._plotter.clear()

    def set_title(
        self,
        title: Optional[str],
        *,
        font_size: Optional[int] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        font_family: Optional[str] = None,
    ) -> None:
        r"""
        Add a title to the plotter.

        Parameters
        ----------
        title: Optional[str]
            The text of the title. If ``None``, no title is added.

        font_size: Optional[int], optional
            The font size of the title text. Default is set by PyVista.

        color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the title text. Can be a string representing a color name (e.g., ``"black"``),
            or an RGB tuple or hex string. Default is set by PyVista.

        font_family: Optional[str], optional
            The font type of the title text. Must be one of ``"arial"``, ``"courier"``, and ``"times"``.
            Default is set by PyVista.


        Returns
        -------
        None
        """
        if title is None:
            return
        self._plotter.add_title(
            title=title,
            font_size=font_size,
            color=color,
            font=font_family,
        )

    def set_bounds(
        self,
        bounds: Optional[Union[str, Tuple[Real, Real, Real, Real, Real, Real]]] = None,
        *,
        grid: Optional[str] = None,
        ticks: Optional[str] = None,
        location: Optional[str] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        font_size: Optional[int] = None,
        font_family: Optional[str] = None,
        bold: bool = False,
        fmt: Optional[str] = None,
        title_x: str = "X Axis",
        title_y: str = "Y Axis",
        title_z: str = "Z Axis",
        n_xlabels: Integral = 5,
        n_ylabels: Integral = 5,
        n_zlabels: Integral = 5,
        show_xaxis: bool = True,
        show_yaxis: bool = True,
        show_zaxis: bool = True,
        show_xlabels: bool = True,
        show_ylabels: bool = True,
        show_zlabels: bool = True,
    ) -> None:
        r"""
        Show bounds on the plotter for visualizing the axes and grid.

        Parameters
        ----------
        bounds: Optional[Union[str, Tuple[Real, Real, Real, Real, Real, Real]]], optional
            The bounds of the plot as a tuple (xmin, xmax, ymin, ymax, zmin, zmax).
            If ``None``, the bounds are determined automatically based on the plotted data.

        grid: Optional[str], optional
            The type of grid to display. Must be one of ``"front"``, ``"back"``, ``"all"``, or ``"none"``.
            Default is ``"none"``.

        ticks: Optional[str], optional
            The type of ticks to display. Must be one of ``"front"``, ``"back"``, ``"all"``, or ``"none"``.
            Default is ``"none"``.

        location: Optional[str], optional
            The location of the axes. Must be one of ``"origin"``, ``"center"``, or ``"default"``.
            Default is ``"default"``.

        color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the bounds and grid lines. Can be a string representing a color name (e.g., ``"black"``),
            or an RGB tuple or hex string. Default is set by PyVista.

        font_size: Optional[int], optional
            The font size of the axis labels and ticks. Default is set by PyVista.

        font_family: Optional[str], optional
            The font type of the axis labels and ticks. Must be one of ``"arial"``, ``"courier"``, and ``"times"``.
            Default is set by PyVista.

        bold: bool, optional
            Whether to display the axis labels and ticks in bold font. Default is ``False``.

        fmt: Optional[str], optional
            A format string for formatting the tick labels (e.g., ``"{:.2f}"``). If ``None``, default formatting is used.

        title_x: str, optional
            The title for the X axis. Default is "X Axis".

        title_y: str, optional
            The title for the Y axis. Default is "Y Axis".

        title_z: str, optional
            The title for the Z axis. Default is "Z Axis".

        n_xlabels: Integral, optional
            The number of labels to display on the X axis. Default is 5.

        n_ylabels: Integral, optional
            The number of labels to display on the Y axis. Default is 5.

        n_zlabels: Integral, optional
            The number of labels to display on the Z axis. Default is 5.

        show_xaxis: bool, optional
            Whether to show the X axis. Default is ``True``.

        show_yaxis: bool, optional
            Whether to show the Y axis. Default is ``True``.

        show_zaxis: bool, optional
            Whether to show the Z axis. Default is ``True``.

        show_xlabels: bool, optional
            Whether to show the labels on the X axis. Default is ``True``.

        show_ylabels: bool, optional
            Whether to show the labels on the Y axis. Default is ``True``.

        show_zlabels: bool, optional
            Whether to show the labels on the Z axis. Default is ``True``.

        """
        self._plotter.show_bounds(
            bounds=bounds,
            grid=grid,
            ticks=ticks,
            location=location,
            color=color,
            font_size=font_size,
            font_family=font_family,
            bold=bold,
            fmt=fmt,
            xtitle=title_x,
            ytitle=title_y,
            ztitle=title_z,
            n_xlabels=n_xlabels,
            n_ylabels=n_ylabels,
            n_zlabels=n_zlabels,
            show_xaxis=show_xaxis,
            show_yaxis=show_yaxis,
            show_zaxis=show_zaxis,
            show_xlabels=show_xlabels,
            show_ylabels=show_ylabels,
            show_zlabels=show_zlabels,
        )

    def set_camera(
        self,
        position: Optional[
            Union[
                str,
                Tuple[Real, Real, Real],
                Tuple[
                    Tuple[Real, Real, Real],
                    Tuple[Real, Real, Real],
                    Tuple[Real, Real, Real],
                ],
            ]
        ] = None,
        focal_point: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        view_up: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
    ) -> None:
        r"""
        Set the camera position for the plotter.

        Parameters
        ----------
        position: Optional[Union[str, Tuple[Real, Real, Real], Tuple[Tuple[Real, Real, Real], Tuple[Real, Real, Real], Tuple[Real, Real, Real]]]], optional
            The position of the camera. Can be a string representing a predefined view (e.g., ``"xy"``),
            a tuple of three real numbers representing the (x, y, z) coordinates of the camera position,
            or a tuple of three tuples representing the (position, focal_point, view_up) for the camera.
            If ``None``, the default camera position is used.

        focal_point: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The focal point of the camera. Can be a string representing a predefined view (e.g., ``"center"``),
            or a tuple of three real numbers representing the (x, y, z) coordinates of the focal point.
            If ``None``, the default focal point is used.

        view_up: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The view up vector for the camera. Can be a string representing a predefined view (e.g., ``"z"``),
            or a tuple of three real numbers representing the (x, y, z) components of the view up vector.
            If ``None``, the default view up vector is used.

        Returns
        -------
        None
        """
        if position is not None:
            self._plotter.camera.position = position
        if focal_point is not None:
            self._plotter.camera.focal_point = focal_point
        if view_up is not None:
            self._plotter.camera.up = view_up

    def add_axes(
        self,
        *,
        size: Real = 0.2,
        line_width: Real = 2.0,
        x_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        y_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        z_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        label_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        x_label: str = "X",
        y_label: str = "Y",
        z_label: str = "Z",
        show_labels: bool = True,
        interactive: bool = False,
    ) -> None:
        r"""
        Add an orientation widget showing the X, Y and Z axes in the bottom left corner
        of the plotter.

        The widget follows the rotations of the camera, which helps to see the
        orientation of the displayed objects. It is not affected by the zoom or the
        translations of the camera.

        Parameters
        ----------
        size: Real, optional (default: ``0.2``)
            The size of the widget as a fraction of the window size, in :math:`(0, 1]`.

        line_width: Real, optional (default: ``2.0``)
            The width of the axis lines (strictly positive).

        x_color, y_color, z_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The colors of the X, Y and Z axes: a color name (e.g. ``"red"``), an RGB
            tuple or a hex string. PyVista defaults: red, green and blue.

        label_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the axis labels. Default is set by PyVista.

        x_label, y_label, z_label: str, optional (default: ``"X"``, ``"Y"``, ``"Z"``)
            The labels of the axes.

        show_labels: bool, optional (default: ``True``)
            Whether to show the labels of the axes.

        interactive: bool, optional (default: ``False``)
            If True, the widget can be moved and resized with the mouse.

        Raises
        ------
        ValueError
            If ``size`` is not in :math:`(0, 1]` or ``line_width`` is not strictly
            positive.

        Returns
        -------
        None
        """
        show_labels = _validate_bool(show_labels, "show_labels")
        interactive = _validate_bool(interactive, "interactive")
        if not isinstance(size, Real) or not numpy.isfinite(size) or not 0 < size <= 1:
            raise ValueError(f"Input 'size' must be a real number in (0, 1]. Got {size}.")
        if not isinstance(line_width, Real) or not numpy.isfinite(line_width) or line_width <= 0:
            raise ValueError(f"Input 'line_width' must be a strictly positive real number. Got {line_width}.")

        viewport = (0.0, 0.0, float(size), float(size))  # Bottom left corner

        # Optional colors are only passed when given, to keep the PyVista defaults
        colors = {"x_color": x_color, "y_color": y_color, "z_color": z_color, "color": label_color}
        colors = {key: value for key, value in colors.items() if value is not None}

        self._plotter.add_axes(
            interactive=interactive,
            line_width=line_width,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
            labels_off=not show_labels,
            viewport=viewport,
            **colors,
        )

    def show(
        self,
        *,
        screenshot: Optional[Any] = None,
    ) -> None:
        r"""
        Show the plotter with the added geometrical objects.

        Parameters
        ----------
        screenshot: Optional[Any], optional
            If provided, saves a screenshot of the plot to the specified file path (e.g., ``"plot.png"``).
            If ``None``, no screenshot is saved.

        Returns
        -------
        None
        """
        self._plotter.show(screenshot=screenshot)

    def add_scalar_bar(
        self,
        *,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        position_x: Optional[int] = None,
        position_y: Optional[int] = None,
        font_family: Optional[str] = None,
        title_font_size: Optional[int] = None,
        label_font_size: Optional[int] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        bold: bool = False,
        vertical: bool = True,
        interactive: bool = False,
    ) -> None:
        r"""
        Add a scalar bar to the plotter for visualizing scalar properties.

        Parameters
        ----------
        title: Optional[str], optional
            The title of the scalar bar. If ``None``, no title is displayed.

        width: Optional[int], optional
            The width of the scalar bar as a fraction of the window width.
            Must be between ``0.0`` and ``1.0`` if provided.

        height: Optional[int], optional
            The height of the scalar bar as a fraction of the window height.
            Must be between ``0.0`` and ``1.0`` if provided.

        position_x: Optional[int], optional
            The x position of the scalar bar as a fraction of the window width.
            Must be between ``0.0`` and ``1.0`` if provided.

        position_y: Optional[int], optional
            The y position of the scalar bar as a fraction of the window height.
            Must be between ``0.0`` and ``1.0`` if provided.

        font_family: Optional[str], optional
            The font type of the scalar bar title. Must be one of ``"arial"``, ``"courier"``, and ``"times"``.

        title_font_size: Optional[int], optional
            The font size of the scalar bar title.

        label_font_size: Optional[int], optional
            The font size of the scalar bar labels.

        color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the scalar bar title text. Can be a string representing a color
            name (e.g., ``"black"``), or an RGB tuple or hex string.

        bold: bool, optional
            Whether to display the scalar bar title in bold font. Default is ``False``.

        vertical: bool, optional
            Whether to display the scalar bar vertically. If ``False``,
            the scalar bar is displayed horizontally. Default is ``True``.

        interactive: bool, optional
            If ``True``, enables interactive mode for the scalar bar in PyVista. Default is ``False``.

        """
        self._plotter.add_scalar_bar(
            title=title,
            width=width,
            height=height,
            position_x=position_x,
            position_y=position_y,
            font_family=font_family,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            color=color,
            bold=bold,
            vertical=vertical,
            interactive=interactive,
        )


    def add_point_cloud(
        self,
        point_cloud: PointCloud,
        *,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        size: Optional[Real] = None,
        opacity: Optional[Real] = None,
        mask: Optional[ArrayLike] = None,
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        show_points_index: bool = False,
        index_color: Optional[Union[str, Tuple[Real, Real, Real]]] = "black",
        index_font_size: Real = 10,
    ) -> None:
        r"""
        Add a point cloud to the plotter (only for :math:`E \leq 3`, the missing
        coordinates are set to 0).

        Only the finite points (after warping) selected by ``mask`` are displayed.

        Parameters
        ----------
        point_cloud: :class:`pysdic.PointCloud`
            The point cloud to display.

        warp_by: Optional[Union[str, ArrayLike]], optional
            The displacement added to the points: the name of a property of the point
            cloud or an array, of shape :math:`(N_p, E)`.

        color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the points: a color name (e.g. ``"red"``), an RGB tuple or a
            hex string. Ignored if ``property`` is given.

        size: Optional[Real], optional
            The size of the points (positive). PyVista default: ``5.0``.

        opacity: Optional[Real], optional
            The opacity of the points, between ``0.0`` (transparent) and ``1.0``
            (opaque). PyVista default: ``1.0``.

        mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_p,)`, True for the points to display.
            By default, all the finite points are displayed.

        property: Optional[Union[str, ArrayLike]], optional
            The property used to color the points: the name of a property of the point
            cloud, or an array of shape :math:`(N_p,)` or :math:`(N_p, A)` (for all the
            points, ``mask`` is applied by the plotter).

        property_axis: Optional[Integral], optional
            The component used to color the points when the property has :math:`A > 1`
            components, in :math:`[0, A)`. By default, the norm of the property vector
            is used.

        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, they are set automatically.

        property_cmap: Optional[Union[str, :class:`matplotlib.colors.Colormap`]], optional
            The colormap (name or object). Matplotlib default: ``"viridis"``.

        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the values below ``property_clim``. By default, the first
            color of the colormap.

        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the values above ``property_clim``. By default, the last
            color of the colormap.

        show_bar: bool, optional (default: ``False``)
            Whether to show the color bar when ``property`` is given.

        bar: Optional[Dict[str, Any]], optional
            The keyword arguments of :meth:`add_scalar_bar` when ``show_bar`` is True.

        vectors: Optional[Union[str, ArrayLike]], optional
            A vector field drawn as arrows from the displayed points: the name of a
            property of the point cloud or an array, of shape :math:`(N_p, 3)` or
            :math:`(N_p, E)`. The points with non-finite vectors have no arrow.

        vector_scale: Union[Real, ArrayLike], optional (default: ``1.0``)
            The scale factor of the arrows (strictly positive and finite): an arrow has
            the length of its vector times its scale. A scalar for all the arrows, or a
            1D array with one value per row of the vector field (:math:`N_p`).

        vector_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the arrows of the vector fields. Default is set by PyVista.

        show_points_index: bool, optional (default: ``False``)
            Whether to write the index of each displayed point next to it (the index in
            the point cloud, also with a ``mask``).

        index_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional (default: ``"black"``)
            The color of the indices.

        index_font_size: Real, optional (default: ``10``)
            The font size of the indices, in pixels of the screen (strictly positive): the
            labels keep their size when zooming.

        Raises
        ------
        TypeError
            If ``point_cloud`` is not a :class:`pysdic.PointCloud`.

        ValueError
            If the point cloud is empty, if :math:`E > 3`, if an input has a wrong
            shape, or if no point can be displayed.

        IndexError
            If ``property_axis`` is out of range.

        """
        pyvista = import_optional("pyvista", "Plotter3D.add_point_cloud")
        if not isinstance(point_cloud, PointCloud):
            raise TypeError(f"Input 'point_cloud' must be a PointCloud. Got type {type(point_cloud)}.")
        if point_cloud.n_points == 0:
            raise ValueError("Cannot visualize an empty point cloud.")
        if point_cloud.n_dimensions > 3:
            raise ValueError(
                f"Visualization is only supported for point clouds with E <= 3. "
                f"Got E={point_cloud.n_dimensions}."
            )
        show_bar = _validate_bool(show_bar, "show_bar")
        bar = _validate_kwargs(bar, "bar")
        show_points_index = _validate_bool(show_points_index, "show_points_index")
        N = point_cloud.n_points

        coordinates = _warped_coordinates(point_cloud, warp_by, "point cloud")
        display_mask = _as_bool_mask(mask, N) if mask is not None else numpy.ones(N, dtype=bool)
        property_values = None
        if property is not None:
            property_values = _scalar_values(
                _resolve_values(point_cloud, property, N, "property", "point cloud"), property_axis
            )

        # Keep only the selected finite points
        display_mask = display_mask & numpy.isfinite(coordinates).all(axis=1)
        if not numpy.any(display_mask):
            raise ValueError("No finite points available for visualization.")

        self._plotter.add_mesh(
            pyvista.PolyData(_as_3d(coordinates[display_mask])),
            color=color,
            point_size=size,
            opacity=opacity,
            scalars=None if property_values is None else property_values[display_mask],
            clim=property_clim,
            cmap=property_cmap,
            below_color=property_below_color,
            above_color=property_above_color,
            render_points_as_spheres=True,
            show_scalar_bar=False,  # The scalar bar is added with add_scalar_bar
        )

        if property_values is not None and show_bar:
            self.add_scalar_bar(**bar)

        if vectors is not None:
            vector_values = _resolve_vectors(point_cloud, vectors, N, point_cloud.n_dimensions, "vectors", "point cloud")
            self._add_arrows(_as_3d(coordinates), vector_values, display_mask, vector_scale, vector_color)

        if show_points_index:
            self._add_index_labels(_as_3d(coordinates), display_mask, index_color, index_font_size)

    def _add_arrows(
        self,
        origins: numpy.ndarray,
        vectors: numpy.ndarray,
        mask: numpy.ndarray,
        scale: Union[Real, ArrayLike],
        color: Optional[Union[str, Tuple[Real, Real, Real]]],
    ) -> None:
        # Draw the vectors (N, 3) times their scale (scalar or (N,)) as arrows from the origins
        # (N, 3), for the selected finite rows
        scale = _as_scalar_or_1d_array(scale, "vector_scale", vectors.shape[0], numpy.float64, finite=True)
        if numpy.any(scale <= 0):
            raise ValueError(f"Input 'vector_scale' must be strictly positive. Got a minimum of {scale.min()}.")
        keep = mask & numpy.isfinite(origins).all(axis=1) & numpy.isfinite(vectors).all(axis=1)
        if not numpy.any(keep):
            raise ValueError("No finite vectors available for visualization.")
        self._plotter.add_arrows(origins[keep], vectors[keep] * scale[keep, numpy.newaxis], mag=1.0, color=color)

    def _add_index_labels(
        self,
        positions: numpy.ndarray,
        mask: numpy.ndarray,
        color: Optional[Union[str, Tuple[Real, Real, Real]]],
        font_size: Real,
    ) -> None:
        # Write the index of each selected finite row of positions (N, 3) next to it. The labels
        # are drawn in the screen plane, with a size in pixels that does not change with the zoom
        font_size = _validate_real(font_size, "index_font_size", positive=True, finite=True)
        keep = mask & numpy.isfinite(positions).all(axis=1)
        if not numpy.any(keep):
            return
        indices = numpy.flatnonzero(keep)
        self._plotter.add_point_labels(
            positions[keep],
            [str(index) for index in indices],
            font_size=max(1, int(round(font_size))),
            text_color=color,
            bold=False,
            show_points=False,
            shape=None,
            always_visible=True,  # Not hidden by the faces of the meshes
        )

    def add_mesh(
        self,
        mesh: Mesh,
        *,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        vertex_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        vertex_size: Optional[Real] = None,
        vertex_opacity: Optional[Real] = None,
        vertex_mask: Optional[ArrayLike] = None,
        show_vertices: bool = True,
        edge_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        edge_width: Optional[Real] = None,
        edge_opacity: Optional[Real] = None,
        show_edges: bool = True,
        face_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        face_opacity: Optional[Real] = None,
        face_mask: Optional[ArrayLike] = None,
        show_faces: bool = True,
        vertex_property: Optional[Union[str, ArrayLike]] = None,
        element_property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        texture: Optional[ArrayLike] = None,
        texture_rgb: bool = True,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vertex_vectors: Optional[Union[str, ArrayLike]] = None,
        element_vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        show_vertices_index: bool = False,
        show_elements_index: bool = False,
        vertex_index_color: Optional[Union[str, Tuple[Real, Real, Real]]] = "black",
        element_index_color: Optional[Union[str, Tuple[Real, Real, Real]]] = "blue",
        index_font_size: Real = 10,
    ) -> None:
        r"""
        Add a mesh to the plotter (only for :math:`E \leq 3`, the missing coordinates
        are set to 0).

        The element type of the mesh must be set. The faces (elements) are displayed if
        they are selected by ``face_mask`` and all their vertices are finite (after
        warping). The vertices are displayed if they are selected by ``vertex_mask`` and
        finite.

        .. note::

            The faces are displayed with duplicated vertices (one copy per element), so
            that each element can have its own texture coordinates.

        Parameters
        ----------
        mesh: :class:`pysdic.Mesh`
            The mesh to display.

        warp_by: Optional[Union[str, ArrayLike]], optional
            The displacement added to the vertices: the name of a vertex property or an
            array, of shape :math:`(N_v, E)`.

        vertex_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the vertices: a color name (e.g. ``"red"``), an RGB tuple or a
            hex string.

        vertex_size: Optional[Real], optional
            The size of the vertices (positive). PyVista default: ``5.0``.

        vertex_opacity: Optional[Real], optional
            The opacity of the vertices, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        vertex_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_v,)`, True for the vertices to display.
            By default, all the finite vertices are displayed. The faces are not
            affected.

        show_vertices: bool, optional (default: ``True``)
            Whether to show the vertices.

        edge_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the edges.

        edge_width: Optional[Real], optional
            The width of the edges (positive). PyVista default: ``1.0``.

        edge_opacity: Optional[Real], optional
            The opacity of the edges, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        show_edges: bool, optional (default: ``True``)
            Whether to show the edges of the displayed faces.

        face_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the faces. Ignored if a property or a texture is given.

        face_opacity: Optional[Real], optional
            The opacity of the faces, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        face_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_e,)`, True for the faces to display. By
            default, all the faces with finite vertices are displayed.

        show_faces: bool, optional (default: ``True``)
            Whether to show the faces.

        vertex_property: Optional[Union[str, ArrayLike]], optional
            The property used to color the faces (interpolated between the vertices):
            the name of a vertex property, or an array of shape :math:`(N_v,)` or
            :math:`(N_v, A)` (for all the vertices).

        element_property: Optional[Union[str, ArrayLike]], optional
            The property used to color the faces (one color per face): the name of an
            element property, or an array of shape :math:`(N_e,)` or :math:`(N_e, A)`
            (for all the elements). Ignored if ``vertex_property`` is given.

        property_axis: Optional[Integral], optional
            The component used for coloring when the property has :math:`A > 1`
            components, in :math:`[0, A)`. By default, the norm of the property vector
            is used.

        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, they are set automatically.

        property_cmap: Optional[Union[str, :class:`matplotlib.colors.Colormap`]], optional
            The colormap (name or object). Matplotlib default: ``"viridis"``.

        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the values below ``property_clim``. By default, the first
            color of the colormap.

        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the values above ``property_clim``. By default, the last
            color of the colormap.

        texture: Optional[ArrayLike], optional
            An image used as texture, of shape :math:`(H, W)`, :math:`(H, W, 1)` (gray
            levels), :math:`(H, W, 3)` or :math:`(H, W, 4)` (RGB or RGBA). Only for
            surface meshes with a UV map (see :attr:`pysdic.Mesh.uvmap`).

        texture_rgb: bool, optional (default: ``True``)
            If False, an RGB or RGBA texture is converted to gray levels.

        show_bar: bool, optional (default: ``False``)
            Whether to show the color bar when a property is given.

        bar: Optional[Dict[str, Any]], optional
            The keyword arguments of :meth:`add_scalar_bar` when ``show_bar`` is True.

        vertex_vectors: Optional[Union[str, ArrayLike]], optional
            A vector field drawn as arrows from the vertices selected by ``vertex_mask``
            (even if ``show_vertices`` is False): the name of a vertex property or an
            array, of shape :math:`(N_v, 3)` or :math:`(N_v, E)`.

        element_vectors: Optional[Union[str, ArrayLike]], optional
            A vector field drawn as arrows from the centers of the elements selected by
            ``face_mask`` (see :meth:`pysdic.Mesh.compute_elements_centers`): the name of
            an element property or an array, of shape :math:`(N_e, 3)` or
            :math:`(N_e, E)`. Only one of ``vertex_vectors`` and ``element_vectors`` can
            be given.

        vector_scale: Union[Real, ArrayLike], optional (default: ``1.0``)
            The scale factor of the arrows (strictly positive and finite): an arrow has
            the length of its vector times its scale. A scalar for all the arrows, or a
            1D array with one value per row of the vector field: :math:`N_v` for
            ``vertex_vectors`` or :math:`N_e` for ``element_vectors``.

        vector_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional
            The color of the arrows of the vector fields. Default is set by PyVista.

        show_vertices_index: bool, optional (default: ``False``)
            Whether to write the index of each vertex selected by ``vertex_mask`` next to
            it (even if ``show_vertices`` is False).

        show_elements_index: bool, optional (default: ``False``)
            Whether to write the index of each element selected by ``face_mask`` at its
            center (see :meth:`pysdic.Mesh.compute_elements_centers`), even if
            ``show_faces`` is False.

        vertex_index_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional (default: ``"black"``)
            The color of the indices of the vertices.

        element_index_color: Optional[Union[str, Tuple[Real, Real, Real]]], optional (default: ``"blue"``)
            The color of the indices of the elements.

        index_font_size: Real, optional (default: ``10``)
            The font size of the indices, in pixels of the screen (strictly positive): the
            labels keep their size when zooming.

        Raises
        ------
        TypeError
            If ``mesh`` is not a :class:`pysdic.Mesh`.

        ValueError
            If the mesh is empty or has no element type, if :math:`E > 3`, if an input
            has a wrong shape, if a texture is given without UV map, or if nothing can
            be displayed.

        IndexError
            If ``property_axis`` is out of range.

        """
        pyvista = import_optional("pyvista", "Plotter3D.add_mesh")
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        if mesh.n_vertices == 0 or mesh.n_elements == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.n_dimensions > 3:
            raise ValueError(
                f"Visualization is only supported for meshes with E <= 3. Got E={mesh.n_dimensions}."
            )
        vtk_cell_type = mesh._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise ValueError(
                "Cannot visualize a mesh without element type. Set 'mesh.element_type' first."
            )
        show_vertices = _validate_bool(show_vertices, "show_vertices")
        show_edges = _validate_bool(show_edges, "show_edges")
        show_faces = _validate_bool(show_faces, "show_faces")
        texture_rgb = _validate_bool(texture_rgb, "texture_rgb")
        show_bar = _validate_bool(show_bar, "show_bar")
        bar = _validate_kwargs(bar, "bar")
        show_vertices_index = _validate_bool(show_vertices_index, "show_vertices_index")
        show_elements_index = _validate_bool(show_elements_index, "show_elements_index")
        if vertex_vectors is not None and element_vectors is not None:
            raise ValueError("Only one of 'vertex_vectors' and 'element_vectors' can be given.")
        N_v, N_e, N_vpe = mesh.n_vertices, mesh.n_elements, mesh.n_vertices_per_element
        elements = mesh.connectivity._elements

        coordinates = _as_3d(_warped_coordinates(mesh.vertices, warp_by, "mesh vertices"))
        finite_vertices = numpy.isfinite(coordinates).all(axis=1)  # (N_v,)

        vertex_display = _as_bool_mask(vertex_mask, N_v, "vertex_mask") if vertex_mask is not None else numpy.ones(N_v, dtype=bool)
        vertex_display = vertex_display & finite_vertices
        face_display = _as_bool_mask(face_mask, N_e, "face_mask") if face_mask is not None else numpy.ones(N_e, dtype=bool)
        face_display = face_display & finite_vertices[elements].all(axis=1)  # Faces with finite vertices

        # Property values at the duplicated vertices of each element, computed for all the elements
        point_values = None
        if vertex_property is not None:
            vertex_values = _scalar_values(
                _resolve_values(mesh.vertices, vertex_property, N_v, "vertex_property", "mesh vertices"),
                property_axis,
            )
            point_values = vertex_values[elements]  # (N_e, N_vpe)
        elif element_property is not None:
            element_values = _scalar_values(
                _resolve_values(mesh.connectivity, element_property, N_e, "element_property", "mesh connectivity"),
                property_axis,
            )
            point_values = numpy.repeat(element_values[:, numpy.newaxis], N_vpe, axis=1)  # (N_e, N_vpe)

        # Texture and texture coordinates (u_i, v_i) of the vertices of each element
        pvtexture = None
        texture_coordinates = None
        if texture is not None:
            uvmap = mesh.uvmap  # (N_e, 2 N_vpe) or None
            if uvmap is None:
                raise ValueError("The UV map of the mesh ('mesh.uvmap') must be set to use a texture.")
            texture_coordinates = uvmap.reshape((N_e, N_vpe, 2))
            pvtexture = pyvista.Texture(_as_color_texture(texture, texture_rgb))

        if (show_faces or show_edges) and numpy.any(face_display):
            # Mesh of the displayed faces, with one copy of the vertices per element
            n_faces = int(numpy.count_nonzero(face_display))
            fictive_vertices = coordinates[elements[face_display]].reshape((-1, 3))
            fictive_cells = numpy.hstack(
                (
                    numpy.full((n_faces, 1), N_vpe, dtype=numpy.int64),
                    numpy.arange(n_faces * N_vpe, dtype=numpy.int64).reshape((n_faces, N_vpe)),
                )
            ).ravel()
            fictive_celltypes = numpy.full(n_faces, vtk_cell_type, dtype=numpy.uint8)
            fictive_mesh = pyvista.UnstructuredGrid(fictive_cells, fictive_celltypes, fictive_vertices)
            if texture_coordinates is not None:
                fictive_mesh.active_texture_coordinates = texture_coordinates[face_display].reshape((-1, 2))

            if show_faces:
                self._plotter.add_mesh(
                    fictive_mesh,
                    color=face_color,
                    opacity=face_opacity,
                    scalars=None if point_values is None else point_values[face_display].ravel(),
                    texture=pvtexture,
                    clim=property_clim,
                    cmap=property_cmap,
                    below_color=property_below_color,
                    above_color=property_above_color,
                    show_scalar_bar=False,  # The scalar bar is added with add_scalar_bar
                    show_vertices=False,
                    show_edges=False,
                )
                if show_bar and point_values is not None:
                    self.add_scalar_bar(**bar)

            if show_edges:
                self._plotter.add_mesh(
                    fictive_mesh.extract_all_edges(),
                    color=edge_color,
                    opacity=edge_opacity,
                    line_width=edge_width,
                    show_scalar_bar=False,
                    show_vertices=False,
                )
        elif show_faces or show_edges:
            raise ValueError("No face with finite vertices available for visualization.")

        if show_vertices:
            if not numpy.any(vertex_display):
                raise ValueError("No finite vertices available for visualization.")
            self._plotter.add_points(
                coordinates[vertex_display],
                color=vertex_color,
                opacity=vertex_opacity,
                point_size=vertex_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )

        if vertex_vectors is not None:
            vector_values = _resolve_vectors(
                mesh.vertices, vertex_vectors, N_v, mesh.n_dimensions, "vertex_vectors", "mesh vertices"
            )
            self._add_arrows(coordinates, vector_values, vertex_display, vector_scale, vector_color)

        if element_vectors is not None:
            vector_values = _resolve_vectors(
                mesh.connectivity, element_vectors, N_e, mesh.n_dimensions, "element_vectors", "mesh connectivity"
            )
            centers = fem.compute_elements_centers(coordinates, elements, mesh.element_type)  # (N_e, 3)
            self._add_arrows(centers, vector_values, face_display, vector_scale, vector_color)

        if show_vertices_index:
            self._add_index_labels(coordinates, vertex_display, vertex_index_color, index_font_size)

        if show_elements_index:
            centers = fem.compute_elements_centers(coordinates, elements, mesh.element_type)  # (N_e, 3)
            self._add_index_labels(centers, face_display, element_index_color, index_font_size)

    def add_integration_points(
        self,
        integration_points: IntegrationPoints,
        mesh: Mesh,
        *,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        size: Optional[Real] = None,
        opacity: Optional[Real] = None,
        mask: Optional[ArrayLike] = None,
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        show_bar: bool = False,
        bar: Optional[Dict[str, Any]] = None,
        vectors: Optional[Union[str, ArrayLike]] = None,
        vector_scale: Union[Real, ArrayLike] = 1.0,
        vector_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        show_points_index: bool = False,
        index_color: Optional[Union[str, Tuple[Real, Real, Real]]] = "black",
        index_font_size: Real = 10,
    ) -> None:
        r"""
        Add integration points to the plotter, at their global coordinates in a mesh
        (see :meth:`pysdic.Mesh.compute_integration_points_coordinates`).

        The unlocated points (element index ``-1``) are not displayed.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points to display, consistent with ``mesh`` (see
            :meth:`pysdic.Mesh.validate_integration_points`).

        mesh: :class:`pysdic.Mesh`
            The mesh in which the points are located.

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
            A vector field drawn as arrows from the displayed points: the name of a
            property of the integration points or an array, of shape :math:`(N_p, 3)`
            or :math:`(N_p, E)`.

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
            If the points are not consistent with the mesh, or if an input has a wrong
            shape (see :meth:`add_point_cloud`).

        """
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        mesh.validate_integration_points(integration_points)

        warped_coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")
        locations = mesh.compute_property_interpolation(
            warped_coordinates, integration_points, allow_unlocated_points=True
        )  # (N_p, E), NaN for the unlocated points

        # The properties of the integration points are shared (read only here)
        point_cloud = PointCloud(
            locations, n_dimensions=mesh.n_dimensions, properties=integration_points.properties
        )
        self.add_point_cloud(
            point_cloud,
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