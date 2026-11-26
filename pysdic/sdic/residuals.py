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
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from ..geometry.integration_points import IntegrationPoints
from ..imaging.image_plotter import ImagePlotter
from .._validation import _validate_bool, _validate_integer, _validate_real
from .gl_correction import GrayLevelGroups, compute_gray_level_groups, matching_correction, znssd_correction
from .pairing import Pairing
from .sdic_analysis import SDICAnalysis


# The values of a gray level correction do not depend on its derivative ("approximate" or "exact"): the
# approximate functions give them without the (possibly dense) exact derivative
_GL_VALUES = {"ZNSSD": znssd_correction, "matching": matching_correction}


# =============================================================================
# Residuals at the integration points
# =============================================================================
@dataclass(frozen=True)
class PairingResiduals:
    r"""
    The gray levels and the residuals of the material points (integration points of the mesh) in the two views
    of a pairing, as compared by :class:`pysdic.SDICCorrelationTerm` (gray level correction included, without
    the weights of the points).

    .. math::

        r_p = g'_2(\vec{X}_p) - g'_1(\vec{X}_p)

    with :math:`g'_k` the gray levels of the view :math:`k` after the gray level correction (the raw gray
    levels without correction).

    Returned by :func:`pysdic.compute_residuals` (current state of the analysis) and
    :meth:`pysdic.SDICCorrelationTerm.compute_residuals` (state given by a vector of unknowns). Used by
    :func:`pysdic.compute_pixel_residual_maps` and :func:`pysdic.display_residual_maps`.

    Attributes
    ----------
    pairing : :class:`pysdic.Pairing`
        The two views.

    gray_levels_1, gray_levels_2 : numpy.ndarray
        The (corrected) gray levels in the two views, shape :math:`(N_p, C)`. NaN for the points projected out
        of the ROI of an image.

    residuals : numpy.ndarray
        ``gray_levels_2 - gray_levels_1``, shape :math:`(N_p, C)`.

    image_points_1, image_points_2 : numpy.ndarray
        The projections :math:`(x, y)` of the points in the two views (pixels), shape :math:`(N_p, 2)`.

    displacement_1, displacement_2 : numpy.ndarray
        The displacements :math:`(N_v, E)` of the frames of the two views used for the evaluation.

    gl_correction : Optional[Dict[str, Any]]
        The gray level correction (complete options), or None for the raw gray levels.

    """

    pairing: Pairing
    gray_levels_1: numpy.ndarray
    gray_levels_2: numpy.ndarray
    residuals: numpy.ndarray
    image_points_1: numpy.ndarray
    image_points_2: numpy.ndarray
    displacement_1: numpy.ndarray
    displacement_2: numpy.ndarray
    gl_correction: Optional[Dict[str, Any]] = None

    @property
    def n_points(self) -> int:
        r"""The number of points :math:`N_p`."""
        return self.residuals.shape[0]

    @property
    def n_channels(self) -> int:
        r"""The number of channels :math:`C`."""
        return self.residuals.shape[1]

    @property
    def label(self) -> str:
        r"""A short description of the correction: ``"raw"``, ``"ZNSSD"`` or ``"matching"``."""
        return "raw" if self.gl_correction is None else self.gl_correction["mode"]


def _pairing_residuals(
    analysis: SDICAnalysis,
    pairing: Pairing,
    positions: Mapping[int, numpy.ndarray],
    displacements: Mapping[int, numpy.ndarray],
    gl_options: Optional[Dict[str, Any]],
    groups: Optional[GrayLevelGroups],
) -> PairingResiduals:
    r"""
    Residuals of a pairing for the positions :math:`(N_p, E)` of the points at the frames of its views.

    .. important::

        Internal use only (shared by :func:`compute_residuals` and
        :meth:`pysdic.SDICCorrelationTerm.compute_residuals`): the inputs are validated.

    """
    result_1 = analysis.get_view(*pairing.view_1).image_project(positions[pairing.frame_index_1])
    result_2 = analysis.get_view(*pairing.view_2).image_project(positions[pairing.frame_index_2])
    corrected_1, corrected_2 = result_1, result_2
    if gl_options is not None:
        corrected_1, corrected_2 = _GL_VALUES[gl_options["mode"]](result_1, result_2, groups, scale=gl_options["scale"])
    g_1, g_2 = numpy.asarray(corrected_1.x), numpy.asarray(corrected_2.x)
    return PairingResiduals(
        pairing=pairing,
        gray_levels_1=g_1,
        gray_levels_2=g_2,
        residuals=g_2 - g_1,
        image_points_1=numpy.asarray(result_1.image_points),
        image_points_2=numpy.asarray(result_2.image_points),
        displacement_1=numpy.asarray(displacements[pairing.frame_index_1]),
        displacement_2=numpy.asarray(displacements[pairing.frame_index_2]),
        gl_correction=None if gl_options is None else dict(gl_options),
    )


def compute_residuals(
    analysis: SDICAnalysis,
    pairing: Pairing,
    integration_points: IntegrationPoints,
    *,
    gl_correction: Optional[Mapping[str, Any]] = None,
    gray_level_groups: Optional[GrayLevelGroups] = None,
) -> PairingResiduals:
    r"""
    Compute the gray levels of material points in the two views of a pairing and their residual, for the current
    state of the analysis, with the gray level correction of :class:`pysdic.SDICCorrelationTerm`.

    The positions of the integration points at the frames of the pairing are interpolated from the deformed
    vertices, projected in the views to interpolate the images, and corrected with ``gl_correction`` (same
    options and same functions as the correlation term):

    .. math::

        r = g'_2 - g'_1, \qquad g_k = [I_k \circ P_k](\vec{X} + \vec{u}_k)

    With the same integration points, ``gl_correction`` and ``gray_level_groups``, the residuals are those of a
    :class:`pysdic.SDICCorrelationTerm` without the weights of the points. This function does not need any
    unknown (e.g. a pairing between two reference frames); to evaluate the residuals of a term for a vector of
    unknowns, use :meth:`pysdic.SDICCorrelationTerm.compute_residuals`.

    .. note::

        The occlusions are not taken into account (as in the correlation term): a point hidden in a view still
        gets the gray level of the image at its projection.

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with the displacements of the frames of the pairing set.

    pairing : :class:`pysdic.Pairing`
        The two views to compare.

    integration_points : :class:`pysdic.IntegrationPoints`
        The material points, all located in the mesh of the analysis.

    gl_correction : Optional[Mapping[str, Any]], optional
        The gray level correction, as in :class:`pysdic.SDICCorrelationTerm`. None (default): raw gray levels.

    gray_level_groups : Optional[:class:`pysdic.GrayLevelGroups`], optional
        The groups of the correction, as in :class:`pysdic.SDICCorrelationTerm`. By default, computed from
        ``gl_correction["neighborhood"]``.

    Returns
    -------
    :class:`pysdic.PairingResiduals`
        The gray levels, the residuals and the projections of the points in the two views.

    Raises
    ------
    ValueError
        If the pairing is not valid in the analysis, the integration points are not consistent with the mesh or
        not all located, a displacement of the pairing is not set, or the gray level options are not valid.

    """
    from .sdic_term import _validate_gl_correction, _validate_gray_level_groups  # Shared validation

    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    analysis.validate_pairing(pairing)
    mesh = analysis.mesh
    mesh.validate_integration_points(integration_points)
    if not numpy.all(integration_points.is_located()):
        raise ValueError("The integration points must all be located (use 'remove_unlocated').")
    gl_options = _validate_gl_correction(gl_correction)
    groups = _validate_gray_level_groups(gray_level_groups, gl_correction, mesh, integration_points)
    if gl_options is not None and groups is None:
        groups = compute_gray_level_groups(mesh, integration_points, gl_options["neighborhood"])

    displacements, positions = {}, {}
    for frame in {pairing.frame_index_1, pairing.frame_index_2}:
        displacements[frame] = analysis.get_displacement(frame)
        positions[frame] = mesh.compute_property_interpolation(analysis.get_deformed_coordinates(frame), integration_points)
    return _pairing_residuals(analysis, pairing, positions, displacements, gl_options, groups)


# =============================================================================
# Pixel residual maps
# =============================================================================
def _splat(
    image_points: numpy.ndarray, values: numpy.ndarray, height: int, width: int, fill_radius: Optional[float]
) -> numpy.ndarray:
    # Map (H, W, C): mean of the values of the points whose projection falls in each pixel (NaN without point),
    # then the pixels without point within fill_radius of a pixel with a point take its value
    C = values.shape[1]
    columns = numpy.rint(image_points[:, 0])
    rows = numpy.rint(image_points[:, 1])
    inside = numpy.isfinite(columns) & numpy.isfinite(rows)
    inside &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    pixel_map = numpy.full((height, width, C), numpy.nan)
    for c in range(C):
        valid = inside & numpy.isfinite(values[:, c])
        index = rows[valid].astype(numpy.int64) * width + columns[valid].astype(numpy.int64)
        counts = numpy.bincount(index, minlength=height * width)
        sums = numpy.bincount(index, weights=values[valid, c], minlength=height * width)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            pixel_map[:, :, c] = (sums / counts).reshape(height, width)

    if fill_radius is not None and fill_radius > 0:
        import scipy.ndimage

        has_value = numpy.isfinite(pixel_map).any(axis=2)
        if has_value.any():
            # Only in the box of the pixels with a value (+ radius): the distance transform is cheaper
            r = int(numpy.ceil(fill_radius))
            rows_any, columns_any = numpy.nonzero(has_value)
            r0, r1 = max(rows_any.min() - r, 0), min(rows_any.max() + r, height - 1)
            c0, c1 = max(columns_any.min() - r, 0), min(columns_any.max() + r, width - 1)
            crop = pixel_map[r0 : r1 + 1, c0 : c1 + 1]
            distances, (nearest_rows, nearest_columns) = scipy.ndimage.distance_transform_edt(
                ~has_value[r0 : r1 + 1, c0 : c1 + 1], return_indices=True
            )
            fill = (distances > 0) & (distances <= fill_radius)
            crop[fill] = crop[nearest_rows[fill], nearest_columns[fill]]
    return pixel_map


def compute_pixel_residual_maps(
    analysis: SDICAnalysis,
    residuals: PairingResiduals,
    *,
    fill_radius: Optional[Real] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Draw the residuals of the material points of a pairing (see :func:`pysdic.compute_residuals` and
    :meth:`pysdic.SDICCorrelationTerm.compute_residuals`) on the pixels of its two views.

    The residual of a point belongs to the material point, not to a view: it is drawn at the projection of the
    point in each view. Each pixel gets the mean residual of the points projected in it (nearest pixel), NaN
    without point. The resolution of the maps is the density of the integration points in the views: with less
    than one point per pixel, some pixels have no point (see ``fill_radius``).

    .. note::

        No ray casting: the residuals are exactly those of the correlation term (corrected gray levels), and the
        occlusions are not taken into account (as in the term).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis (the sizes of the sensors of the views).

    residuals : :class:`pysdic.PairingResiduals`
        The residuals of the pairing.

    fill_radius : Optional[Real], optional
        If given, a pixel without point takes the value of the nearest pixel with a point if it is within
        ``fill_radius`` pixels (e.g. ``1.0`` to close the holes of a density of about one point per pixel).
        By default, no filling: the maps show the actual density of the points.

    Returns
    -------
    map_1, map_2 : numpy.ndarray
        The residual maps on the views 1 and 2 of the pairing, of shape :math:`(H, W)` for one channel or
        :math:`(H, W, C)`.

    """
    if not isinstance(residuals, PairingResiduals):
        raise TypeError(f"Input 'residuals' must be a PairingResiduals. Got type {type(residuals)}.")
    fill_radius = _validate_real(fill_radius, "fill_radius", non_negative=True, finite=True, allow_none=True)
    maps = []
    for (frame, index), image_points in (
        (residuals.pairing.view_1, residuals.image_points_1),
        (residuals.pairing.view_2, residuals.image_points_2),
    ):
        camera = analysis.get_view(frame, index).camera
        pixel_map = _splat(image_points, residuals.residuals, camera.sensor_height, camera.sensor_width, fill_radius)
        maps.append(pixel_map[:, :, 0] if residuals.n_channels == 1 else pixel_map)
    return maps[0], maps[1]


def _frame_label(analysis: SDICAnalysis, frame: Integral) -> str:
    # "frame f", with its acquisition time if the times of the analysis are set
    if analysis.times is None:
        return f"frame {frame}"
    return f"frame {frame} (t = {analysis.times[frame]:g})"


def display_residual_maps(
    analysis: SDICAnalysis,
    residuals: PairingResiduals,
    *,
    fill_radius: Optional[Real] = None,
    channel: Optional[Integral] = None,
    clim: Optional[Tuple[Real, Real]] = None,
    cmap: str = "RdBu",
    image_opacity: Real = 0.3,
    show_mesh: bool = True,
    figsize: Tuple[Real, Real] = (18, 6),
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""
    Display the residuals of a pairing (see :func:`pysdic.compute_pixel_residual_maps`).

    The figure has 3 panels: the residual map on the view 1 and on the view 2 of the pairing (over their
    images, with the deformed mesh), and the histogram of the residuals of the points, with their mean and RMS.

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis.

    residuals : :class:`pysdic.PairingResiduals`
        The residuals of the pairing (:func:`pysdic.compute_residuals` or
        :meth:`pysdic.SDICCorrelationTerm.compute_residuals`).

    fill_radius : Optional[Real], optional
        See :func:`pysdic.compute_pixel_residual_maps`.

    channel : Optional[Integral], optional
        The channel of the residual to display, required for images with several channels.

    clim : Optional[Tuple[Real, Real]], optional
        The color limits of the maps. By default, symmetric limits given by the 99th percentile of :math:`|r|`.

    cmap : str, optional (default: ``"RdBu"``)
        The colormap of the maps.

    image_opacity : Real, optional (default: ``0.3``)
        The opacity of the images under the maps.

    show_mesh : bool, optional (default: ``True``)
        Whether to draw the edges of the mesh deformed by the displacements of the residuals.

    figsize : Tuple[Real, Real], optional (default: ``(18, 6)``)
        The size of the figure in inches (width, height).

    title : Optional[str], optional
        The title of the figure. By default, the pairing and the correction.

    show : bool, optional (default: ``True``)
        Whether to show the figure (with ``matplotlib.pyplot.show``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If ``channel`` is missing for images with several channels, or if no residual is finite.

    """
    show_mesh = _validate_bool(show_mesh, "show_mesh")
    show = _validate_bool(show, "show")
    map_1, map_2 = compute_pixel_residual_maps(analysis, residuals, fill_radius=fill_radius)
    values = residuals.residuals
    if residuals.n_channels > 1:
        if channel is None:
            raise ValueError(
                f"Input 'channel' is required to display the residuals of images with {residuals.n_channels} channels."
            )
        channel = _validate_integer(channel, "channel", min_value=0)
        if channel >= residuals.n_channels:
            raise IndexError(f"Input 'channel' must be in [0, {residuals.n_channels - 1}]. Got {channel}.")
        map_1, map_2, values = map_1[:, :, channel], map_2[:, :, channel], values[:, channel]
    elif channel is not None:
        raise ValueError("Input 'channel' can only be given for images with several channels.")
    values = values[numpy.isfinite(values)].ravel()
    if values.size == 0:
        raise ValueError(f"No finite residual for {residuals.pairing}.")
    if clim is None:
        bound = float(numpy.percentile(numpy.abs(values), 99)) or 1.0
        clim = (-bound, bound)
    rms = float(numpy.sqrt(numpy.mean(values**2)))

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    views = (
        (residuals.pairing.view_1, map_1, residuals.displacement_1),
        (residuals.pairing.view_2, map_2, residuals.displacement_2),
    )
    for ax, ((frame, index), pixel_map, displacement) in zip(axes[:2], views):
        plotter = ImagePlotter(analysis.get_view(frame, index), ax=ax)
        plotter.add_image(opacity=image_opacity)
        plotter.add_image(pixel_map, cmap=cmap, clim=clim, show_bar=True, bar={"label": f"residual ({residuals.label})"})
        if show_mesh:
            plotter.add_mesh(
                analysis.mesh, warp_by=displacement, show_faces=False, show_vertices=False,
                edge_color="black", edge_opacity=0.4,
            )
        plotter.set_title(f"{_frame_label(analysis, frame).capitalize()}, view {index}")

    ax = axes[2]
    ax.hist(values, bins=100, color="gray")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(f"residual ({residuals.label})")
    ax.set_ylabel("number of points")
    ax.set_title(f"{values.size} points: mean = {values.mean():.3g}, RMS = {rms:.3g}")

    figure.suptitle(f"{residuals.pairing}, {residuals.label} gray levels" if title is None else title)
    figure.tight_layout()
    if show:
        plt.show()
    return figure