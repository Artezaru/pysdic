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

# Private validation helpers of the SDIC terms (not part of the public API).

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Sequence, Tuple, Union

import numpy

from .._validation import _validate_integer


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


def _validate_transformations(
    analysis: Any, transformations: Sequence[Tuple[str, Integral]], allow_empty: bool
) -> Tuple[List[Tuple[str, int]], List[Any], List[int]]:
    # The entries (kind, index), the transformation objects and the start of their parameters in
    # the block of the transformations (the last value is the number of parameters)
    if not isinstance(transformations, (list, tuple)):
        raise TypeError(
            f"Input 'transformations' must be a list of tuples (kind, index). Got type {type(transformations)}."
        )
    if not allow_empty and len(transformations) == 0:
        raise ValueError("Input 'transformations' must contain at least one transformation.")
    entries, objects, starts = [], [], [0]
    for k, entry in enumerate(transformations):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"Input 'transformations[{k}]' must be a tuple (kind, index). Got {entry!r}.")
        transformation = analysis.get_transformation(*entry)
        entry = (entry[0], _validate_integer(entry[1], f"transformations[{k}][1]", min_value=0))
        if any(transformation is other for other in objects):
            raise ValueError(f"The transformation {entry} is given several times in 'transformations'.")
        if transformation.n_parameters == 0 or transformation.parameters is None:
            raise ValueError(f"The transformation {entry} ({type(transformation).__name__}) has no parameters to optimize.")
        if transformation.is_locked:
            raise ValueError(f"The parameters of the transformation {entry} ({type(transformation).__name__}) are locked.")
        entries.append(entry)
        objects.append(transformation)
        starts.append(starts[-1] + transformation.n_parameters)
    return entries, objects, starts