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

# =============================================================================
# FEM specific input validation helpers
# =============================================================================
# Internal use only: these helpers are shared by all the modules of pysdic.fem.
# The generic checks (booleans, numbers, arrays, ...) are in pysdic._validation
# and follow the same conventions:
#     - A wrong type raises a ``TypeError``, a wrong value or shape a ``ValueError``
#       and an out-of-range index an ``IndexError``.
#
# Each public function of pysdic.fem validates its inputs exactly once, then calls
# private ``_*_impl`` functions which assume validated inputs (no re-validation along
# a chain of calls).

from __future__ import annotations
from typing import Optional, Tuple
from numbers import Integral
from numpy.typing import ArrayLike

import numpy

from .._validation import (
    _as_integer_array,
    _validate_bool,
    _validate_choice,
    _validate_integer,
    _validate_str,
)
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES


def _validate_element_type(element_type: str) -> str:
    r"""
    Check that ``element_type`` is a supported element type (case insensitive) and
    return it in lower case.

    .. important::

        Internal use only.

    """
    element_type = _validate_str(element_type, "element_type").lower()
    return _validate_choice(
        element_type, "element_type", tuple(_MAPPING_ELEMENT_TYPE_TO_PROPERTIES)
    )


def _validate_n_vertices(
    n_vertices: Optional[Integral], connectivity_array: numpy.ndarray
) -> int:
    r"""
    Validate (or infer if None) the number of vertices :math:`N_{v}` of the mesh.

    .. important::

        Internal use only. ``connectivity_array`` is an integer array.

    The connectivity must not contain negative vertex indices. The inferred value is
    the maximum vertex index in ``connectivity_array`` plus one. A given value must be
    a positive integer greater than the maximum vertex index.

    """
    if connectivity_array.size > 0:
        if int(connectivity_array.min()) < 0:
            raise IndexError(
                "Input 'connectivity_array' contains negative vertex indices. "
                "All the vertex indices must be non-negative."
            )
        max_index = int(connectivity_array.max())
    else:
        max_index = -1

    if n_vertices is None:
        return max_index + 1

    n_vertices = _validate_integer(n_vertices, "n_vertices", min_value=1)
    if n_vertices <= max_index:
        raise ValueError(
            f"Input 'n_vertices' ({n_vertices}) must be greater than the maximum vertex "
            f"index in 'connectivity_array' ({max_index})."
        )
    return n_vertices


def _validate_connectivity(
    connectivity_array: ArrayLike,
    n_vertices: Optional[Integral] = None,
    n_vertices_per_element: Optional[int] = None,
) -> Tuple[numpy.ndarray, int]:
    r"""
    Validate the connectivity of a mesh and the number of vertices :math:`N_{v}`.

    .. important::

        Internal use only.

    The following checks are performed:

    - ``connectivity_array`` is a 2D integer array of shape :math:`(N_{e}, N_{vpe})`,
      with :math:`N_{vpe}` equal to ``n_vertices_per_element`` if given.
    - The vertex indices are non-negative and lower than :math:`N_{v}`
      (see ``_validate_n_vertices``).
    - The vertices of each element are distinct (a repeated vertex would be summed in
      sparse matrices but overwritten in dense ones).


    Parameters
    ----------
    connectivity_array: ArrayLike
        Connectivity with shape :math:`(N_{e}, N_{vpe})`.

    n_vertices: Optional[Integral], optional (default: ``None``)
        Number of vertices :math:`N_{v}`, inferred if None.

    n_vertices_per_element: Optional[int], optional (default: ``None``)
        Expected number of vertices per element (from the element type), not checked
        if None.


    Returns
    -------
    connectivity_array: numpy.ndarray
        Validated connectivity with dtype ``numpy.int64``.

    n_vertices: int
        Validated (or inferred) number of vertices.

    """
    # Integer indices only (floats are not truncated), N_vpe fixed by the element type if given
    connectivity_array = _as_integer_array(
        connectivity_array,
        "connectivity_array",
        shape=(None, n_vertices_per_element),
        shape_hint=f"(N_e, {'N_vpe' if n_vertices_per_element is None else n_vertices_per_element})",
    )
    n_vertices = _validate_n_vertices(n_vertices, connectivity_array)

    if connectivity_array.shape[0] > 0 and connectivity_array.shape[1] > 1:
        sorted_connectivity = numpy.sort(connectivity_array, axis=1)
        repeated = numpy.any(
            sorted_connectivity[:, 1:] == sorted_connectivity[:, :-1], axis=1
        )
        if numpy.any(repeated):
            first = int(numpy.flatnonzero(repeated)[0])
            raise ValueError(
                f"Input 'connectivity_array' contains {int(numpy.count_nonzero(repeated))} "
                f"elements with repeated vertices (e.g. element {first}: "
                f"{connectivity_array[first].tolist()}). The vertices of each element must "
                f"be distinct."
            )
    return connectivity_array, n_vertices


def _valid_points_mask(
    element_indices: numpy.ndarray, allow_unlocated_points: bool
) -> numpy.ndarray:
    r"""
    Return the boolean mask of the points located in an element of the mesh.

    .. important::

        Internal use only.

    The element index ``-1`` marks a point not located in any element of the mesh.

    - If ``allow_unlocated_points`` is False, any negative index raises an
      ``IndexError`` (numpy would otherwise silently use the last elements of the
      mesh).
    - If ``allow_unlocated_points`` is True, the points with element index ``-1`` are
      marked as invalid. Any other negative index raises an ``IndexError``.


    Parameters
    ----------
    element_indices: numpy.ndarray
        Validated element indices with shape :math:`(N_{p},)`.

    allow_unlocated_points: bool
        Whether the element index ``-1`` is accepted to mark unlocated points.


    Returns
    -------
    numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)`, True for the points to process.

    """
    if numpy.any(element_indices < -1):
        raise IndexError(
            "Input 'element_indices' contains negative indices other than -1. "
            "Only -1 can be used to mark points not located in any element."
        )

    unlocated = element_indices == -1
    if not allow_unlocated_points and numpy.any(unlocated):
        raise IndexError(
            f"Input 'element_indices' contains {int(numpy.count_nonzero(unlocated))} "
            "points with element index -1 (not located in any element). "
            "Set 'allow_unlocated_points=True' to treat them as invalid points."
        )
    return ~unlocated


def _validate_element_indices(
    element_indices: ArrayLike, n_elements: int, allow_unlocated_points: bool
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Validate the element indices of the points and return the mask of the located
    points.

    .. important::

        Internal use only.

    The indices must be lower than ``n_elements``. The index ``-1`` is accepted only
    if ``allow_unlocated_points`` is True (see ``_valid_points_mask``).


    Returns
    -------
    element_indices: numpy.ndarray
        Validated element indices with shape :math:`(N_{p},)` and dtype
        ``numpy.int64``.

    located_mask: numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)`, True for the points located in an
        element.

    """
    allow_unlocated_points = _validate_bool(
        allow_unlocated_points, "allow_unlocated_points"
    )
    element_indices = _as_integer_array(
        element_indices, "element_indices", shape=(None,), shape_hint="(N_p,)"
    )
    located_mask = _valid_points_mask(element_indices, allow_unlocated_points)
    if element_indices.size > 0 and int(element_indices.max()) >= n_elements:
        raise IndexError(
            f"Input 'element_indices' contains indices up to "
            f"{int(element_indices.max())}, which is not lower than the number of "
            f"elements ({n_elements})."
        )
    return element_indices, located_mask


def _check_finite_rows(
    array: numpy.ndarray, rows_mask: numpy.ndarray, name: str
) -> None:
    r"""
    Raise a ``ValueError`` if ``array`` contains non-finite values in the rows
    selected by ``rows_mask`` (the other rows are not checked).

    .. important::

        Internal use only. ``array`` has shape :math:`(N_{p}, ...)` and ``rows_mask``
        shape :math:`(N_{p},)`.

    """
    finite = numpy.all(numpy.isfinite(array.reshape(array.shape[0], -1)), axis=1)
    bad = rows_mask & ~finite
    if numpy.any(bad):
        raise ValueError(
            f"Input '{name}' contains non-finite values for "
            f"{int(numpy.count_nonzero(bad))} located points (e.g. point "
            f"{int(numpy.flatnonzero(bad)[0])})."
        )