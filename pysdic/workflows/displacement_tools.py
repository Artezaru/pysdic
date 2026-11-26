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

# Tools on the displacements of an analysis: interpolation in time of the missing frames and
# median filter in space.
#
# These functions do not modify the analysis: they return the new displacements, to write with
# SDICAnalysis.set_displacement.

from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, Optional, Sequence

import numpy
import scipy.sparse
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .. import fem
from .._validation import _validate_choice, _validate_integer, _validate_real
from ..sdic.sdic_analysis import SDICAnalysis


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_frames(analysis: SDICAnalysis, frames: Sequence[Integral], name: str) -> list:
    # Distinct frames of the analysis
    frames = [_validate_integer(frame, f"frame of '{name}'", min_value=0) for frame in frames]
    if len(set(frames)) != len(frames):
        raise ValueError(f"Input '{name}' contains duplicated frames: {frames}.")
    outside = [frame for frame in frames if frame >= analysis.n_frames]
    if outside:
        raise ValueError(f"The frames {outside} of '{name}' are not in the analysis ({analysis.n_frames} frames).")
    return frames


def _require_displacements(analysis: SDICAnalysis, frames: Sequence[int], name: str) -> None:
    missing = [frame for frame in frames if not analysis.has_displacement(frame)]
    if missing:
        raise ValueError(f"The displacements of the frames {missing} of '{name}' are not set.")


def _radius_neighborhood(coordinates: numpy.ndarray, radius: float) -> scipy.sparse.csr_matrix:
    # Boolean (N_v, N_v) neighborhood of the vertices within a Euclidean distance (each vertex
    # included in its own neighborhood)
    N_v = coordinates.shape[0]
    pairs = cKDTree(coordinates).query_pairs(radius, output_type="ndarray")  # (n, 2), i < j
    rows = numpy.concatenate((pairs[:, 0], pairs[:, 1], numpy.arange(N_v)))
    columns = numpy.concatenate((pairs[:, 1], pairs[:, 0], numpy.arange(N_v)))
    data = numpy.ones(rows.size, dtype=bool)
    return scipy.sparse.csr_matrix((data, (rows, columns)), shape=(N_v, N_v))


# =============================================================================
# Interpolation in time
# =============================================================================
def compute_interpolated_displacements(
    analysis: SDICAnalysis,
    frames: Sequence[Integral],
    known_frames: Sequence[Integral],
    *,
    method: str = "linear",
) -> Dict[int, numpy.ndarray]:
    r"""
    Interpolate in time the displacements of frames from the displacements of known frames.

    The known displacements :math:`U(t_k)` are interpolated at the times :math:`t_f` of the
    frames (see :attr:`pysdic.SDICAnalysis.times`), vertex by vertex and component by
    component:

    - ``"linear"``: piecewise linear interpolation between the two closest known frames;
    - ``"cubic"``: cubic spline through the known frames (``scipy.interpolate.CubicSpline``,
      not-a-knot conditions), smoother for a regular motion.

    Before the first known time and after the last one, the displacement is constant (the
    displacement of the first or last known frame): a spline is never extrapolated.

    Typical use: the frames without views (no image, broken calibration) between measured
    frames. Do not put a reference frame outside the time series in ``known_frames`` (e.g. a
    frame 0 placed at :math:`t = 0` by convention).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with times and the displacements of ``known_frames`` set.

    frames : Sequence[Integral]
        The frames to interpolate (not in ``known_frames``).

    known_frames : Sequence[Integral]
        The frames giving the displacements, with distinct times (any order).

    method : str, optional (default: ``"linear"``)
        ``"linear"`` or ``"cubic"``.

    Returns
    -------
    Dict[int, numpy.ndarray]
        For each frame of ``frames``, its interpolated displacement :math:`(N_v, E)`.

    Raises
    ------
    ValueError
        If the analysis has no times, a known displacement is not set, two known frames have
        the same time, or a frame is in both inputs.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import compute_interpolated_displacements

        measured = [frame for frame in range(1, analysis.n_frames) if analysis.n_views(frame) > 0]
        missing = [frame for frame in range(1, analysis.n_frames) if analysis.n_views(frame) == 0]

        for frame, displacement in compute_interpolated_displacements(analysis, missing, measured).items():
            analysis.set_displacement(frame, displacement)

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    if analysis.times is None:
        raise ValueError("The analysis has no times: create it with 'times' (or use 'with_times').")
    method = _validate_choice(method, "method", ("linear", "cubic"))
    frames = _validate_frames(analysis, frames, "frames")
    known_frames = _validate_frames(analysis, known_frames, "known_frames")
    if not known_frames:
        raise ValueError("Input 'known_frames' must contain at least one frame.")
    common = sorted(set(frames) & set(known_frames))
    if common:
        raise ValueError(f"The frames {common} are in both 'frames' and 'known_frames'.")
    _require_displacements(analysis, known_frames, "known_frames")
    if not frames:
        return {}

    times = numpy.asarray(analysis.times, dtype=numpy.float64)
    order = numpy.argsort(times[known_frames], kind="stable")
    known_frames = [known_frames[k] for k in order]
    known_times = times[known_frames]
    if numpy.any(numpy.diff(known_times) <= 0.0):
        raise ValueError(f"The known frames must have distinct times. Got {dict(zip(known_frames, known_times))}.")
    known = numpy.stack([analysis.get_displacement(frame) for frame in known_frames])  # (n_k, N_v, E)

    targets = numpy.clip(times[frames], known_times[0], known_times[-1])  # Constant outside
    if known_times.size == 1:
        values = numpy.repeat(known, len(frames), axis=0)
    elif method == "cubic":
        values = CubicSpline(known_times, known, axis=0)(targets)
    else:
        after = numpy.clip(numpy.searchsorted(known_times, targets, side="right"), 1, known_times.size - 1)
        alpha = (targets - known_times[after - 1]) / (known_times[after] - known_times[after - 1])
        alpha = alpha[:, numpy.newaxis, numpy.newaxis]
        values = (1.0 - alpha) * known[after - 1] + alpha * known[after]
    return {frame: values[k] for k, frame in enumerate(frames)}


# =============================================================================
# Median filter in space
# =============================================================================
def compute_median_filtered_displacements(
    analysis: SDICAnalysis,
    frames: Sequence[Integral],
    neighborhood: Optional[scipy.sparse.spmatrix] = None,
    *,
    max_distance: Optional[Integral] = None,
    radius: Optional[Real] = None,
) -> Dict[int, numpy.ndarray]:
    r"""
    Apply a median filter to the displacements of frames: the displacement of each vertex is
    replaced, component by component, by the median of the displacements of its neighborhood.

    The neighborhood is given in one of three ways (at most one input):

    - ``neighborhood``: a boolean sparse matrix :math:`(N_v, N_v)`, the neighborhood of the
      vertex :math:`i` being ``neighborhood[i].indices`` (e.g. from
      :meth:`pysdic.Mesh.compute_vertices_neighborhood`);
    - ``max_distance``: the vertices within a topological distance (two vertices are at
      distance 1 if they share an element), see :meth:`pysdic.Mesh.compute_vertices_neighborhood`;
    - ``radius``: the vertices within a Euclidean distance on the reference vertices of the
      mesh (in the unit of the mesh), independent of the size of the elements.

    By default, ``max_distance=1``. Each vertex is in its own neighborhood with ``max_distance``
    and ``radius``.

    The median removes the isolated outliers (a vertex badly measured, e.g. on a reflection)
    while keeping the steps of the displacement, unlike a mean.

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with the displacements of ``frames`` set.

    frames : Sequence[Integral]
        The frames to filter.

    neighborhood : Optional[scipy.sparse.spmatrix], optional
        The neighborhood of the vertices.

    max_distance : Optional[Integral], optional
        The topological distance of the neighborhood (at least 1).

    radius : Optional[Real], optional
        The Euclidean distance of the neighborhood (strictly positive).

    Returns
    -------
    Dict[int, numpy.ndarray]
        For each frame, its filtered displacement :math:`(N_v, E)`.

    Raises
    ------
    ValueError
        If several neighborhoods are given, the neighborhood has a wrong shape, or a
        displacement is not set.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import compute_median_filtered_displacements

        filtered = compute_median_filtered_displacements(analysis, [3, 4, 5], radius=5.0)  # 5 mm
        for frame, displacement in filtered.items():
            analysis.set_displacement(frame, displacement)

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    frames = _validate_frames(analysis, frames, "frames")
    _require_displacements(analysis, frames, "frames")
    mesh = analysis.mesh
    given = [name for name, value in (("neighborhood", neighborhood), ("max_distance", max_distance), ("radius", radius)) if value is not None]
    if len(given) > 1:
        raise ValueError(f"Give at most one of 'neighborhood', 'max_distance' and 'radius'. Got {given}.")

    if neighborhood is not None:
        if not scipy.sparse.issparse(neighborhood):
            raise TypeError(f"Input 'neighborhood' must be a scipy sparse matrix. Got type {type(neighborhood)}.")
        if neighborhood.shape != (mesh.n_vertices, mesh.n_vertices):
            raise ValueError(
                f"Input 'neighborhood' must have the shape {(mesh.n_vertices, mesh.n_vertices)}. Got {neighborhood.shape}."
            )
        neighborhood = scipy.sparse.csr_matrix(neighborhood, dtype=bool)
    elif radius is not None:
        radius = _validate_real(radius, "radius", positive=True, finite=True)
        neighborhood = _radius_neighborhood(mesh.coordinates, radius)
    else:
        max_distance = 1 if max_distance is None else _validate_integer(max_distance, "max_distance", min_value=1)
        neighborhood = mesh.compute_vertices_neighborhood(max_distance=max_distance, self_neighborhood=True)

    return {
        frame: fem.compute_neighborhood_statistics(analysis.get_displacement(frame), neighborhood, "median")["median"]
        for frame in frames
    }