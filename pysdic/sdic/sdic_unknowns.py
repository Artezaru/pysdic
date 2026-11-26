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

from numbers import Integral
from typing import Sequence, Tuple, Union

import numpy

from .._validation import _validate_integer

# The kinds of transformations that can be optimized (see pysdic.SDICAnalysis.get_transformation)
_TRANSFORMATION_KINDS = ("extrinsic", "distortion", "intrinsic")


class SDICUnknowns:
    r"""
    The description of the unknowns of an optimization: the frames whose displacements are
    optimized and the transformations of the cameras whose parameters are optimized, in the
    order of the vector of unknowns

    .. math::

        \vec{p} = [\vec{p}_U, \vec{p}_T]

    - :math:`\vec{p}_U`: the **total** displacements :math:`(N_v, E)` (from the reference mesh,
      not increments) of the ``optimized_frames``, each one flattened in Fortran order (all the
      :math:`x` components, then all the :math:`y` components, ...) and concatenated in the
      order of ``optimized_frames``;
    - :math:`\vec{p}_T`: the parameters of the ``optimized_transformations``, concatenated in
      the order of ``optimized_transformations``.

    The component :math:`e` of the displacement of the vertex :math:`v` at the optimized frame
    :math:`k` (in the order of ``optimized_frames``) has the index :math:`k N_v E + e N_v + v`.

    **The unknowns do not know the analysis.** As a :class:`pysdic.Pairing`, a
    :class:`SDICUnknowns` is a light frozen description, validated by an analysis with
    :meth:`pysdic.SDICAnalysis.validate_unknowns`. The same unknowns can be used with several
    analyses (e.g. an analysis and the analysis given by its ``with_mesh``). The vector itself
    is built by the analysis, which knows the sizes and holds the values:

    - :meth:`pysdic.SDICAnalysis.pack_unknowns`: the vector :math:`\vec{p}` of the current state;
    - :meth:`pysdic.SDICAnalysis.unpack_unknowns`: split a vector into displacements and
      parameters;
    - :meth:`pysdic.SDICAnalysis.apply_unknowns`: write a vector in the analysis;
    - :meth:`pysdic.SDICAnalysis.n_unknowns`, :meth:`pysdic.SDICAnalysis.get_unknowns_frame_slice`,
      :meth:`pysdic.SDICAnalysis.get_unknowns_transformation_slice` and
      :meth:`pysdic.SDICAnalysis.compute_unknowns_indices`: the layout of the vector.

    The objects of a ``pysolvegn`` problem (:class:`pysdic.SDICCorrelationTerm`,
    :class:`pysdic.SDICTemporalRegularizationTerm`, :class:`pysdic.SDICFixedParametrization`)
    are all created with the **same analysis and the same unknowns objects**, checked by
    :func:`pysdic.check_same_problem` before solving.

    .. warning::

        **With a parametrization, do not use** ``pack_unknowns``, ``unpack_unknowns`` **or**
        ``apply_unknowns`` **on the vectors of the solver**: they are reduced vectors
        :math:`\vec{q}`. Use ``parametrization.reparametrized_pack()``,
        ``reparametrized_unpack(q)`` and ``reparametrized_apply(q)`` (see
        :class:`pysdic.SDICFixedParametrization`).

    The object cannot be modified after its creation. It is compared by identity: the objects
    of a problem must share the **same** :class:`SDICUnknowns` object.

    Parameters
    ----------
    optimized_frames : Union[Integral, Sequence[Integral]]
        The frames whose displacements are optimized, in the order of the unknowns: a single
        frame index or a sequence of distinct non-negative indices (possibly empty).

    optimized_transformations : Sequence[Tuple[str, Integral]]
        The optimized transformations, as tuples ``(kind, index)`` with ``kind`` in
        ``"extrinsic"``, ``"distortion"`` and ``"intrinsic"`` (see
        :meth:`pysdic.SDICAnalysis.get_transformation`), in the order of the unknowns (possibly
        empty), without duplicates.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If a frame or a transformation is given twice, or a kind is not valid.

    Examples
    --------
    .. code-block:: python

        from pysdic import SDICUnknowns

        unknowns = SDICUnknowns([1, 2], [("extrinsic", 1)])
        analysis.validate_unknowns(unknowns)       # Frames and transformations exist, parameters not locked, ...

        p0 = analysis.pack_unknowns(unknowns)
        ...
        analysis.apply_unknowns(unknowns, result.parameters)  # Displacements and extrinsic written

    """

    __slots__ = ("_frames", "_transformations")

    def __init__(
        self,
        optimized_frames: Union[Integral, Sequence[Integral]],
        optimized_transformations: Sequence[Tuple[str, Integral]],
    ) -> None:
        if isinstance(optimized_frames, Integral) and not isinstance(optimized_frames, bool):
            optimized_frames = [optimized_frames]
        if not isinstance(optimized_frames, (list, tuple, numpy.ndarray)):
            raise TypeError(
                f"Input 'optimized_frames' must be an integer or a sequence of integers. Got type {type(optimized_frames)}."
            )
        frames = tuple(
            _validate_integer(frame, f"optimized_frames[{k}]", min_value=0) for k, frame in enumerate(optimized_frames)
        )
        if len(set(frames)) != len(frames):
            raise ValueError(f"Input 'optimized_frames' must not contain duplicated frames. Got {list(frames)}.")

        if not isinstance(optimized_transformations, (list, tuple)):
            raise TypeError(
                "Input 'optimized_transformations' must be a list of tuples (kind, index). "
                f"Got type {type(optimized_transformations)}."
            )
        entries = []
        for k, entry in enumerate(optimized_transformations):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"Input 'optimized_transformations[{k}]' must be a tuple (kind, index). Got {entry!r}.")
            kind, index = entry
            if kind not in _TRANSFORMATION_KINDS:
                raise ValueError(
                    f"The kind of 'optimized_transformations[{k}]' must be one of {_TRANSFORMATION_KINDS}. Got {kind!r}."
                )
            entries.append((kind, _validate_integer(index, f"optimized_transformations[{k}][1]", min_value=0)))
        if len(set(entries)) != len(entries):
            raise ValueError(f"Input 'optimized_transformations' must not contain duplicates. Got {entries}.")

        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_transformations", tuple(entries))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("A SDICUnknowns cannot be modified: create new unknowns.")

    # ------------------------------------------------------------------ Properties
    @property
    def optimized_frames(self) -> Tuple[int, ...]:
        r"""The optimized frames, in the order of the unknowns."""
        return self._frames

    @property
    def optimized_transformations(self) -> Tuple[Tuple[str, int], ...]:
        r"""The optimized transformations ``(kind, index)``, in the order of the unknowns."""
        return self._transformations

    @property
    def n_frames(self) -> int:
        r"""The number of optimized frames :math:`N_f`."""
        return len(self._frames)

    @property
    def n_transformations(self) -> int:
        r"""The number of optimized transformations."""
        return len(self._transformations)

    @property
    def is_empty(self) -> bool:
        r"""Whether there is neither optimized frame nor optimized transformation."""
        return not self._frames and not self._transformations

    def __repr__(self) -> str:
        return (
            f"SDICUnknowns(optimized_frames={list(self._frames)}, "
            f"optimized_transformations={list(self._transformations)})"
        )


# =============================================================================
# Selections (shared helpers, independent of the analysis)
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