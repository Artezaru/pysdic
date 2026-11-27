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

# Private validation helpers of the SDIC module (not part of the public API), shared by the
# analysis, the terms, the parametrizations and the residuals. They do not import the analysis
# (sdic_analysis.py imports this module).

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy

from ..geometry.integration_points import IntegrationPoints
from .._validation import _validate_choice, _validate_integer, _validate_real
from .gl_correction import GrayLevelGroups

# Default options of a gray level correction (see SDICCorrelationTerm, compute_residuals)
_GL_DEFAULTS = {"mode": "ZNSSD", "jacobian": "approximate", "neighborhood": None, "scale": 1.0}


# =============================================================================
# Frames
# =============================================================================
def _validate_frames(analysis: Any, frames: Union[Integral, Sequence[Integral]], name: str, allow_empty: bool) -> List[int]:
    # Unique valid frame indices of an analysis (a single integer is accepted)
    if isinstance(frames, Integral) and not isinstance(frames, bool):
        frames = [frames]
    if not isinstance(frames, (list, tuple, numpy.ndarray)):
        raise TypeError(f"Input '{name}' must be an integer or a sequence of integers. Got type {type(frames)}.")
    frames = [analysis._validate_frame(frame, f"{name}[{k}]") for k, frame in enumerate(frames)]
    if len(set(frames)) != len(frames):
        raise ValueError(f"Input '{name}' must not contain duplicated frames. Got {frames}.")
    if not allow_empty and len(frames) == 0:
        raise ValueError(f"Input '{name}' must contain at least one frame.")
    return frames


# =============================================================================
# Selections of vertices and components (see SDICAnalysis.compute_unknowns_indices)
# =============================================================================
def _as_vertex_selection(vertices, n_vertices: int) -> numpy.ndarray:
    # Sorted unique vertex indices from None (all), indices or a boolean mask (N_v,)
    if vertices is None:
        return numpy.arange(n_vertices)
    vertices = numpy.asarray(vertices)
    if vertices.dtype == bool:
        if vertices.shape != (n_vertices,):
            raise ValueError(f"A boolean mask 'vertices' must have shape ({n_vertices},). Got {vertices.shape}.")
        return numpy.flatnonzero(vertices)
    if vertices.ndim != 1 or not numpy.issubdtype(vertices.dtype, numpy.integer):
        raise ValueError("Input 'vertices' must be a 1D array of vertex indices or a boolean mask.")
    if numpy.any((vertices < 0) | (vertices >= n_vertices)):
        raise ValueError(f"Input 'vertices' must contain indices in [0, {n_vertices - 1}].")
    return numpy.unique(vertices)


def _as_component_selection(components, n_dimensions: int) -> numpy.ndarray:
    # Sorted unique component indices
    if components is None:
        return numpy.arange(n_dimensions)
    components = numpy.atleast_1d(numpy.asarray(components))
    if components.ndim != 1 or not numpy.issubdtype(components.dtype, numpy.integer):
        raise ValueError("Input 'components' must be a sequence of component indices.")
    if numpy.any((components < 0) | (components >= n_dimensions)):
        raise ValueError(f"Input 'components' must contain indices in [0, {n_dimensions - 1}].")
    return numpy.unique(components)


# =============================================================================
# Integration points and gray level corrections (SDICCorrelationTerm, compute_residuals)
# =============================================================================
def _validate_located_integration_points(mesh: Any, integration_points: IntegrationPoints) -> None:
    # Integration points consistent with the mesh and all located
    mesh.validate_integration_points(integration_points)
    if not numpy.all(integration_points.is_located()):
        raise ValueError("The integration points must all be located (use 'remove_unlocated').")


def _validate_gl_correction(gl_correction: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if gl_correction is None:
        return None
    if not isinstance(gl_correction, Mapping):
        raise TypeError(f"Input 'gl_correction' must be a dictionary or None. Got type {type(gl_correction)}.")
    unknown = set(gl_correction) - set(_GL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown keys {sorted(unknown)} in 'gl_correction'. Valid keys: {tuple(_GL_DEFAULTS)}.")
    options = {**_GL_DEFAULTS, **gl_correction}
    options["mode"] = _validate_choice(options["mode"], "gl_correction['mode']", ("ZNSSD", "matching"))
    options["jacobian"] = _validate_choice(options["jacobian"], "gl_correction['jacobian']", ("approximate", "exact"))
    options["neighborhood"] = _validate_integer(options["neighborhood"], "gl_correction['neighborhood']", min_value=0, allow_none=True)
    options["scale"] = _validate_real(options["scale"], "gl_correction['scale']", positive=True, finite=True)
    return options

def _validate_gray_level_groups(
    gray_level_groups: Optional[GrayLevelGroups],
    gl_correction: Optional[Mapping[str, Any]],
    mesh: Any,
    integration_points: IntegrationPoints,
) -> Optional[GrayLevelGroups]:
    # Groups given by the user: consistent with the points and the mesh, and not given twice
    if gray_level_groups is None:
        return None
    if not isinstance(gray_level_groups, GrayLevelGroups):
        raise TypeError(f"Input 'gray_level_groups' must be a GrayLevelGroups or None. Got type {type(gray_level_groups)}.")
    if gl_correction is None:
        raise ValueError("Input 'gray_level_groups' requires a gray level correction ('gl_correction').")
    if "neighborhood" in gl_correction:
        raise ValueError(
            "Give either gl_correction['neighborhood'] (groups computed by the term) or 'gray_level_groups' "
            "(groups computed once with compute_gray_level_groups), not both."
        )
    if gray_level_groups.n_points != integration_points.n_points or gray_level_groups.n_elements != mesh.n_elements:
        raise ValueError(
            f"Input 'gray_level_groups' ({gray_level_groups.n_points} points, {gray_level_groups.n_elements} elements) "
            f"does not match the integration points ({integration_points.n_points}) and the mesh ({mesh.n_elements} elements)."
        )
    if not numpy.array_equal(gray_level_groups.point_elements, integration_points.element_indices):
        raise ValueError(
            "Input 'gray_level_groups' was computed for other integration points (different elements of the points)."
        )
    return gray_level_groups