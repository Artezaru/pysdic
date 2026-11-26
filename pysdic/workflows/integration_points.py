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

# Preparation of the integration points of an analysis: a regular grid of points in each
# element, its size from a target density of points per pixel, and the selection of the points
# visible in all the views.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Optional, Sequence, Tuple

import numpy
from numpy.typing import ArrayLike

from ..geometry.integration_points import IntegrationPoints
from ..geometry.mesh import Mesh
from ..imaging.mesh_projection import compute_projected_elements_areas, compute_visibility
from .._validation import _as_array, _validate_choice, _validate_integer, _validate_real
from ..sdic.sdic_analysis import SDICAnalysis


# =============================================================================
# Internal helpers
# =============================================================================
_QUADRANGLES = ("quadrangle_4", "quadrangle_8")
_TRIANGLES = ("triangle_3", "triangle_6")


def _reference_domain(element_type: str) -> Tuple[numpy.ndarray, float]:
    # Center (2,) and area of the reference element, read from its Gauss points (their weights
    # integrate the constant 1 over the reference element), so that the convention of the
    # natural coordinates is not hard-coded here
    gauss_points = IntegrationPoints.from_gauss_points(element_type, 1)
    return gauss_points.natural_coordinates.mean(axis=0), float(numpy.sum(gauss_points.weights))


def _reference_grid(element_type: str, n: int) -> Tuple[numpy.ndarray, float]:
    # Natural coordinates (n², 2) of the centers of the n² cells of the reference element, and
    # the area of a cell (the weight of each point: midpoint rule)
    center, area = _reference_domain(element_type)
    steps = (numpy.arange(n) + 0.5) / n  # Centers of the cells of [0, 1]
    if element_type in _QUADRANGLES:
        # Square reference element of side sqrt(area) centered on the Gauss points
        side = numpy.sqrt(area)
        a, b = numpy.meshgrid(steps, steps, indexing="ij")
        points = center - 0.5 * side + side * numpy.stack((a.ravel(), b.ravel()), axis=1)
    else:
        # Reference triangle (0, 0), (1, 0), (0, 1): n² sub-triangles of the regular subdivision
        if not (numpy.isclose(area, 0.5) and numpy.allclose(center, 1.0 / 3.0)):
            raise NotImplementedError(
                f"The grid of a {element_type} element expects the reference triangle (0, 0), (1, 0), (0, 1). "
                f"Got a reference element of area {area} centered on {center}."
            )
        i, j = numpy.meshgrid(numpy.arange(n), numpy.arange(n), indexing="ij")
        up = (i + j) <= n - 1  # Sub-triangles pointing up: centroid ((i + 1/3) / n, (j + 1/3) / n)
        down = (i + j) <= n - 2  # Sub-triangles pointing down: centroid ((i + 2/3) / n, (j + 2/3) / n)
        points = numpy.concatenate(
            (
                numpy.stack((i[up] + 1.0 / 3.0, j[up] + 1.0 / 3.0), axis=1),
                numpy.stack((i[down] + 2.0 / 3.0, j[down] + 2.0 / 3.0), axis=1),
            )
        ) / n
    return points, area / n**2


def _subset(integration_points: IntegrationPoints, keep: numpy.ndarray) -> IntegrationPoints:
    # New integration points with the selected points (weights kept)
    return IntegrationPoints(
        integration_points.natural_coordinates[keep],
        integration_points.element_indices[keep],
        weights=integration_points.weights[keep],
    )


def _frame_displacement(analysis: SDICAnalysis, frame: int) -> numpy.ndarray:
    # The displacement of a frame, which must be set (no hidden zero displacement)
    if not analysis.has_displacement(frame):
        raise ValueError(
            f"The displacement of the frame {frame} is not set: set it before (e.g. "
            f"analysis.set_zero_displacement({frame}))."
        )
    return analysis.get_displacement(frame)


# =============================================================================
# Grid of integration points
# =============================================================================
def create_grid_integration_points(
    mesh: Mesh,
    n: Integral,
    *,
    elements: Optional[ArrayLike] = None,
) -> IntegrationPoints:
    r"""
    Create a regular grid of :math:`n^2` integration points in each element of a surface mesh.

    The reference element is divided into :math:`n^2` cells of the same area and a point is
    placed at the center of each cell, with the area of the cell as weight (midpoint rule): the
    weights of an element sum to the area of the reference element, as the Gauss points.

    - **quadrangles** (``quadrangle_4``, ``quadrangle_8``): :math:`n \times n` squares;
    - **triangles** (``triangle_3``, ``triangle_6``): the :math:`n^2` sub-triangles of the
      regular subdivision of each edge in :math:`n` segments, points at their centroids.

    The points are ordered element by element (the :math:`n^2` points of the first element,
    then the next one).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        A surface mesh with one of the element types above.

    n : Integral
        The number of subdivisions of the reference element along an edge (at least 1).

    elements : Optional[ArrayLike], optional
        The indices of the elements receiving points. By default, all the elements.

    Returns
    -------
    :class:`pysdic.IntegrationPoints`
        The :math:`n^2 N_e` points, all located.

    Raises
    ------
    TypeError
        If ``mesh`` is not a :class:`pysdic.Mesh`.

    ValueError
        If the element type is not supported, or an input has a wrong value.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import create_grid_integration_points

        integration_points = create_grid_integration_points(analysis.mesh, 6)  # 36 points per element

    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if mesh.element_type not in _QUADRANGLES + _TRIANGLES:
        raise ValueError(
            f"The element type must be one of {_QUADRANGLES + _TRIANGLES}. Got {mesh.element_type!r}."
        )
    n = _validate_integer(n, "n", min_value=1)
    if elements is None:
        elements = numpy.arange(mesh.n_elements)
    else:
        elements = _as_array(numpy.atleast_1d(elements), "elements", numpy.int64, shape=(None,), finite=True)
        if numpy.any((elements < 0) | (elements >= mesh.n_elements)):
            raise ValueError(f"Input 'elements' must be in [0, {mesh.n_elements - 1}].")

    points, weight = _reference_grid(mesh.element_type, n)
    return IntegrationPoints(
        numpy.tile(points, (elements.size, 1)),
        numpy.repeat(elements, points.shape[0]),
        weights=numpy.full(points.shape[0] * elements.size, weight),
    )


def compute_grid_size(
    analysis: SDICAnalysis,
    density: Real,
    *,
    frame: Integral = 0,
    statistic: str = "max",
) -> int:
    r"""
    Compute the grid size :math:`n` of :func:`create_grid_integration_points` giving a target
    density of points per pixel in the views of a frame.

    The projected area :math:`A_e` (pixels²) of each element is computed in each view of the
    frame (see :func:`pysdic.compute_projected_elements_areas`, with the displacement of the
    frame), and the largest one over the views is kept. The grid size is

    .. math::

        n = \left\lceil \sqrt{\rho \, A} \right\rceil

    with :math:`\rho` the density and :math:`A` a statistic of the areas :math:`A_e` over the
    elements: with ``"max"``, every element gets at least the density in every view.

    .. note::

        The occlusions are not taken into account: hidden elements also have an area (see
        :func:`select_visible_integration_points`).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with the displacement of ``frame`` set.

    density : Real
        The target number of points per pixel (strictly positive), e.g. 1 for about one point
        per pixel.

    frame : Integral, optional (default: ``0``)
        The frame whose views give the projected areas.

    statistic : str, optional (default: ``"max"``)
        ``"max"``, ``"median"`` or ``"mean"`` of the areas over the elements.

    Returns
    -------
    int
        The grid size :math:`n \geq 1`.

    Raises
    ------
    ValueError
        If the frame has no view, its displacement is not set, or no element has a finite area.

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    density = _validate_real(density, "density", positive=True, finite=True)
    frame = _validate_integer(frame, "frame", min_value=0)
    statistic = _validate_choice(statistic, "statistic", ("max", "median", "mean"))
    if analysis.n_views(frame) == 0:
        raise ValueError(f"The frame {frame} has no view.")
    displacement = _frame_displacement(analysis, frame)

    areas = numpy.stack(
        [
            compute_projected_elements_areas(analysis.mesh, view, warp_by=displacement)
            for view in analysis.get_views(frame)
        ]
    )  # (n_views, N_e)
    with numpy.errstate(invalid="ignore"):
        areas = numpy.nanmax(numpy.where(numpy.isfinite(areas), areas, numpy.nan), axis=0)  # (N_e,)
    areas = areas[numpy.isfinite(areas)]
    if areas.size == 0:
        raise ValueError(f"No element has a finite projected area in the views of the frame {frame}.")
    area = {"max": numpy.max, "median": numpy.median, "mean": numpy.mean}[statistic](areas)
    return max(1, int(numpy.ceil(numpy.sqrt(density * area))))


# =============================================================================
# Visible integration points
# =============================================================================
@dataclass(frozen=True)
class VisibleIntegrationPointsResult:
    r"""
    The integration points visible in all the views of the selected frames, returned by
    :func:`select_visible_integration_points`.

    Attributes
    ----------
    integration_points : :class:`pysdic.IntegrationPoints`
        The visible points (weights kept), to give to the terms.

    mask : numpy.ndarray
        The boolean mask :math:`(N_p,)` of the visible points among the input points.

    visibility : Dict[Tuple[int, int], numpy.ndarray]
        For each view ``(frame, view)``, the boolean mask :math:`(N_p,)` of the input points
        visible in this view (to find the view that removes points).

    """

    integration_points: IntegrationPoints
    mask: numpy.ndarray
    visibility: Dict[Tuple[int, int], numpy.ndarray]

    @property
    def n_removed(self) -> int:
        r"""The number of removed points."""
        return int(self.mask.size - numpy.count_nonzero(self.mask))


def select_visible_integration_points(
    analysis: SDICAnalysis,
    integration_points: IntegrationPoints,
    *,
    frames: Optional[Sequence[Integral]] = None,
    tolerance: Optional[Real] = None,
) -> VisibleIntegrationPointsResult:
    r"""
    Select the integration points visible in all the views of some frames of an analysis.

    A point is kept if it is located in the mesh and, in every view ``(f, v)`` of the frames,
    with the mesh deformed by the displacement of the frame :math:`f`:

    - **3D mesh**: it is visible (see :func:`pysdic.compute_visibility`): projected in the
      domain of the pixel centers and in the ROI of the image, and not hidden by the mesh;
    - **2D mesh**: its gray level is finite (projected in the ROI of the image), the plane
      cannot hide itself.

    Use it before building a :class:`pysdic.SDICCorrelationTerm`: the term does not handle the
    occlusions, and a point out of a ROI gives NaN residuals.

    .. important::

        A 3D mesh requires the optional package ``open3d`` (``pip install open3d``).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with the displacements of the frames set.

    integration_points : :class:`pysdic.IntegrationPoints`
        The points to select, consistent with the mesh of the analysis.

    frames : Optional[Sequence[Integral]], optional
        The frames whose views are checked. By default, all the frames with views.

    tolerance : Optional[Real], optional
        The tolerance of the occlusions (3D mesh), see :func:`pysdic.compute_visibility`.

    Returns
    -------
    :class:`VisibleIntegrationPointsResult`
        The visible points, their mask and the visibility in each view.

    Raises
    ------
    ValueError
        If a frame has no view or no displacement set, or the integration points are not
        consistent with the mesh.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import create_grid_integration_points, select_visible_integration_points

        grid = create_grid_integration_points(analysis.mesh, 6)
        selection = select_visible_integration_points(analysis, grid)
        print(f"{selection.n_removed} hidden points removed")
        integration_points = selection.integration_points

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    mesh = analysis.mesh
    mesh.validate_integration_points(integration_points)
    if frames is None:
        frames = [frame for frame in range(analysis.n_frames) if analysis.n_views(frame) > 0]
    else:
        frames = [_validate_integer(frame, "frame of 'frames'", min_value=0) for frame in frames]
        empty = [frame for frame in frames if analysis.n_views(frame) == 0]
        if empty:
            raise ValueError(f"The frames {empty} have no view.")

    mask = numpy.asarray(integration_points.is_located(), dtype=bool).copy()
    visibility = {}
    for frame in frames:
        displacement = _frame_displacement(analysis, frame)
        if mesh.n_dimensions == 3:
            for index, view in enumerate(analysis.get_views(frame)):
                visibility[(frame, index)] = compute_visibility(
                    mesh, integration_points, view, warp_by=displacement, tolerance=tolerance
                )
        else:
            points = mesh.compute_property_interpolation(
                analysis.get_deformed_coordinates(frame), integration_points, allow_unlocated_points=True
            )
            for index, view in enumerate(analysis.get_views(frame)):
                gray_levels = view.image_project(points).gray_levels
                visibility[(frame, index)] = mask & numpy.isfinite(gray_levels).all(axis=1)
        for index in range(analysis.n_views(frame)):
            mask &= visibility[(frame, index)]

    return VisibleIntegrationPointsResult(
        integration_points=_subset(integration_points, mask),
        mask=mask,
        visibility=visibility,
    )