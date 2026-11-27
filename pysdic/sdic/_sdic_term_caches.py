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

# Internal caches and structures of a SDIC term (see :class:`pysdic.SDICCorrelationTerm`). They are
# not part of the public API: each term owns its own objects, created with the term.
#
# - :class:`SDICTermOperatorsCache`: the operators ``(r, J)`` or ``(g, H, c)`` of the last
#   evaluations, returned again for the same parameters if nothing changed in the analysis
#   (``use_cache=True``).
# - :class:`SDICTermReferenceProjectionCache`: the projections of the reference views (views
#   without unknowns), kept between evaluations while their frame and their camera do not change
#   (``use_cache=True``).
# - :class:`SDICTermSparseStructure`: the CSR structure of the displacement blocks of the
#   Jacobian, fixed for the life of the term (always used).
# - :class:`SDICTermGeometry`: the data of the mesh and the integration points used by a term
#   (shape functions, weights, sparse structure), shared by the terms created with the with_*
#   methods (same mesh and same integration points, both locked).
#
# The state of the analysis is read through the displacements of the frames and the
# ``parameters`` of the transformations (the other attributes of the transformations are fixed).

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy
import scipy.sparse

from .. import fem

# Largest index of a sparse matrix stored with numpy.int32 indices
_INT32_MAX = numpy.iinfo(numpy.int32).max


# =============================================================================
# State of the analysis
# =============================================================================
def _update_array(digest: Any, array: Optional[numpy.ndarray]) -> None:
    # Add an array (shape, dtype and values) or None to a hash
    if array is None:
        digest.update(b"N")
        return
    array = numpy.ascontiguousarray(array)
    digest.update(b"A")
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())


def _state_key(arrays: Sequence[Optional[numpy.ndarray]]) -> bytes:
    # Hash of a sequence of arrays (or None): displacements of frames, parameters of transformations, ...
    digest = hashlib.blake2b(digest_size=32)
    for array in arrays:
        _update_array(digest, array)
    return digest.digest()


def _make_read_only(result: Any) -> Any:
    # The arrays of a projection result kept by a cache cannot be modified in place
    for name in ("x", "jacobian_dx", "jacobian_dp", "image_points"):
        array = getattr(result, name, None)
        if isinstance(array, numpy.ndarray):
            array.flags.writeable = False
    return result


# =============================================================================
# Operators
# =============================================================================
class SDICTermOperatorsCache:
    r"""
    Cache of the operators of a SDIC term: ``{"r", "J"}`` (mode ``"rJ"``) or
    ``{"g", "H", "c"}`` (mode ``"gH"``) of the last ``max_entries`` evaluations.

    The key of an evaluation (:meth:`key`) is a hash of the vector of unknowns, of the
    displacements of the reference frames and of the parameters of the transformations of the
    analysis that are not optimized: the operators are returned again only if nothing has moved.

    .. important::

        Internal use only.

    """

    def __init__(self, max_entries: int = 1) -> None:
        self.max_entries = int(max_entries)
        self._entries: List[Tuple[bytes, Dict[str, Any]]] = []  # Most recent first
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        parameters: numpy.ndarray,
        reference_displacements: Sequence[numpy.ndarray],
        transformations: Sequence[Any],
    ) -> bytes:
        return _state_key(
            [parameters, *reference_displacements, *(transformation.parameters for transformation in transformations)]
        )

    def get(self, key: bytes) -> Optional[Dict[str, Any]]:
        # The operators of the key, or None (the entry is moved to the front)
        for k, (cached_key, operators) in enumerate(self._entries):
            if cached_key == key:
                self._entries.insert(0, self._entries.pop(k))
                self.hits += 1
                return operators
        self.misses += 1
        return None

    def store(self, key: bytes, operators: Dict[str, Any]) -> None:
        self._entries = [entry for entry in self._entries if entry[0] != key]
        self._entries.insert(0, (key, operators))
        del self._entries[self.max_entries :]

    def clear(self) -> None:
        self._entries.clear()

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"SDICTermOperatorsCache(max_entries={self.max_entries}, n_entries={self.n_entries}, "
            f"hits={self.hits}, misses={self.misses})"
        )


# =============================================================================
# Reference projections
# =============================================================================
class SDICTermReferenceProjectionCache:
    r"""
    Cache of the projections (:class:`pysdic.ImageProjectionResult`, values only) of the
    reference views of a SDIC term: the views whose frame is not optimized and whose camera
    has no optimized transformation. Their projection does not depend on the unknowns.

    One entry per reference view, with the key (:meth:`key`) of the state used to compute it:
    the displacement of its frame and the parameters of the transformations of its camera.
    When the key of a view changes, its entry is removed (the projection is invalid) and the
    view must be projected again.

    The arrays of the kept results are read-only.

    .. important::

        Internal use only.

    """

    def __init__(self, views: Sequence[Tuple[int, int]]) -> None:
        self.views = tuple(views)
        self._entries: Dict[Tuple[int, int], Tuple[bytes, Any]] = {}
        self.hits = 0
        self.misses = 0

    def __contains__(self, view: Tuple[int, int]) -> bool:
        # The view is a reference view of the term
        return view in self.views

    @staticmethod
    def key(displacement: numpy.ndarray, transformations: Sequence[Any]) -> bytes:
        return _state_key([displacement, *(transformation.parameters for transformation in transformations)])

    def get(self, view: Tuple[int, int], key: bytes) -> Optional[Any]:
        # The projection of the view for the key, or None (an entry with another key is removed)
        entry = self._entries.get(view)
        if entry is not None and entry[0] == key:
            self.hits += 1
            return entry[1]
        self._entries.pop(view, None)
        self.misses += 1
        return None

    def store(self, view: Tuple[int, int], key: bytes, result: Any) -> Any:
        if view not in self.views:
            raise KeyError(f"The view {view} is not a reference view of the term.")
        self._entries[view] = (key, _make_read_only(result))
        return result

    def clear(self) -> None:
        self._entries.clear()

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return sum(
            getattr(result, name).nbytes
            for _, result in self._entries.values()
            for name in ("x", "jacobian_dx", "jacobian_dp", "image_points")
            if isinstance(getattr(result, name, None), numpy.ndarray)
        )

    def __repr__(self) -> str:
        return (
            f"SDICTermReferenceProjectionCache(views={list(self.views)}, n_entries={self.n_entries}, "
            f"nbytes={self.nbytes}, hits={self.hits}, misses={self.misses})"
        )


# =============================================================================
# Sparse structure of the displacement blocks
# =============================================================================
class SDICTermSparseStructure:
    r"""
    CSR structure of the displacement blocks :math:`(N_p C, N_v E)` of the Jacobian of a SDIC
    term (gray levels of a view w.r.t. the displacement of its frame, see
    :func:`pysdic.fem.assemble_displacement_jacobian_matrix`). It only depends on the mesh and
    on the integration points (locked by the term), so it is computed once.

    The row :math:`p + c N_p` (Fortran order of :math:`(N_p, C)`) has :math:`E N_{vpe}` entries,
    at the columns :math:`e N_v + v` for :math:`e = 0..E-1` and the vertices :math:`v` of the
    element of the point sorted in increasing order. The non-finite gradients give explicit
    zeros: the structure is the same at each evaluation, the values are always finite.

    All the matrices built by :meth:`assemble`, :meth:`add`, :meth:`scale_rows` and
    :meth:`zero_rows` share the read-only ``indices`` and ``indptr`` arrays of the structure
    (only the values are allocated). Use :meth:`owns` to know whether a matrix has this structure.

    .. important::

        Internal use only. All the integration points are located.

    """

    def __init__(
        self,
        connectivity: numpy.ndarray,
        element_indices: numpy.ndarray,
        shape_functions: numpy.ndarray,
        n_vertices: int,
        n_dimensions: int,
    ) -> None:
        self.n_points, self.n_vertices_per_element = shape_functions.shape
        self.n_vertices = int(n_vertices)
        self.n_dimensions = int(n_dimensions)
        N_p, N_vpe, N_v, E = self.n_points, self.n_vertices_per_element, self.n_vertices, self.n_dimensions
        self.entries_per_row = E * N_vpe

        # Vertices of the element of each point sorted, and the shape functions with them
        order = numpy.argsort(connectivity, axis=1)  # (N_e, N_vpe)
        sorted_connectivity = numpy.take_along_axis(connectivity, order, axis=1)
        point_order = order[element_indices]  # (N_p, N_vpe)
        self.shape_functions = numpy.take_along_axis(shape_functions, point_order, axis=1)  # (N_p, N_vpe)
        self.shape_functions.flags.writeable = False
        vertices = sorted_connectivity[element_indices]  # (N_p, N_vpe)
        self._point_columns = (
            numpy.arange(E, dtype=numpy.int64)[numpy.newaxis, :, numpy.newaxis] * N_v
            + vertices[:, numpy.newaxis, :]
        ).reshape(N_p, E * N_vpe)  # (N_p, E N_vpe), sorted in each row
        self._layouts: Dict[int, Tuple[numpy.ndarray, numpy.ndarray]] = {}  # C -> (indices, indptr)

    # ------------------------------------------------------------------ Structure
    def layout(self, n_channels: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""The read-only ``(indices, indptr)`` of the blocks with :math:`C` channels."""
        C = int(n_channels)
        if C not in self._layouts:
            nnz = C * self.n_points * self.entries_per_row
            dtype = numpy.int32 if max(nnz, self.n_vertices * self.n_dimensions) <= _INT32_MAX else numpy.int64
            indices = numpy.tile(self._point_columns.astype(dtype).ravel(), C)  # Same columns for each channel
            indptr = numpy.arange(0, nnz + 1, self.entries_per_row, dtype=dtype)
            indices.flags.writeable = False
            indptr.flags.writeable = False
            self._layouts[C] = (indices, indptr)
        return self._layouts[C]

    def shape(self, n_channels: int) -> Tuple[int, int]:
        return (int(n_channels) * self.n_points, self.n_vertices * self.n_dimensions)

    def owns(self, matrix: Any) -> bool:
        # The matrix is a csr matrix sharing the structure (built by this object)
        if not scipy.sparse.issparse(matrix) or matrix.format != "csr":
            return False
        return any(matrix.indptr is indptr and matrix.indices is indices for indices, indptr in self._layouts.values())

    def _with_data(self, data: numpy.ndarray, n_channels: int) -> scipy.sparse.csr_matrix:
        indices, indptr = self.layout(n_channels)
        matrix = scipy.sparse.csr_matrix(
            (data.ravel(), indices, indptr), shape=self.shape(n_channels), copy=False
        )
        matrix.has_sorted_indices = True  # Sorted by construction, no duplicates
        return matrix

    def _n_channels(self, matrix: scipy.sparse.csr_matrix) -> int:
        return matrix.shape[0] // self.n_points

    # ------------------------------------------------------------------ Values
    def assemble(
        self, gradients: numpy.ndarray, row_factors: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
        r"""
        Block :math:`(N_p C, N_v E)` of the gradients :math:`(N_p, C, E)` of the gray levels
        w.r.t. the positions of the points, times the shape functions. The rows can be multiplied
        by ``row_factors`` :math:`(N_p C,)` (weights, sign). The points with non-finite gradients
        give explicit zeros.
        """
        N_p, C, E = gradients.shape
        finite = numpy.isfinite(gradients)
        if not finite.all():
            gradients = numpy.where(finite.all(axis=(1, 2))[:, numpy.newaxis, numpy.newaxis], gradients, 0.0)
        if row_factors is not None:
            gradients = gradients * row_factors.reshape(C, N_p).T[:, :, numpy.newaxis]  # Row p + c N_p
        # J[p + c N_p, e N_v + v] = dg_c/dx_e(p) N_v(p), ordered by (c, p) then (e, v)
        data = numpy.einsum("pce,pv->cpev", gradients, self.shape_functions)  # (C, N_p, E, N_vpe)
        return self._with_data(data, C)

    def add(self, first: scipy.sparse.csr_matrix, second: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        # Sum of two matrices of the structure (same number of channels)
        return self._with_data(first.data + second.data, self._n_channels(first))

    def scale_rows(self, matrix: scipy.sparse.csr_matrix, weights: numpy.ndarray) -> scipy.sparse.csr_matrix:
        # Rows multiplied by the weights (N_p C,)
        data = matrix.data.reshape(-1, self.entries_per_row) * weights[:, numpy.newaxis]
        return self._with_data(data, self._n_channels(matrix))

    def zero_rows(self, matrix: scipy.sparse.csr_matrix, rows: numpy.ndarray) -> scipy.sparse.csr_matrix:
        # Rows (boolean (N_p C,)) set to zero, kept as explicit zeros
        data = numpy.where(rows[:, numpy.newaxis], 0.0, matrix.data.reshape(-1, self.entries_per_row))
        return self._with_data(data, self._n_channels(matrix))

    def __repr__(self) -> str:
        return (
            f"SDICTermSparseStructure(n_points={self.n_points}, n_vertices={self.n_vertices}, "
            f"n_dimensions={self.n_dimensions}, entries_per_row={self.entries_per_row}, "
            f"channels={sorted(self._layouts)})"
        )


# =============================================================================
# Geometry of a term
# =============================================================================
@dataclass(frozen=True, eq=False)
class SDICTermGeometry:
    r"""
    The data of a SDIC term that only depend on the mesh and on the integration points: the
    vertices and the shape functions of the points, their weights and the sparse structure of
    the displacement blocks of the Jacobian.

    The mesh (locked by the analysis) and the integration points (locked by the term) cannot be
    modified: while the same objects are used and locked, these data are valid. They are shared
    by the terms created with ``with_unknowns``, ``with_pairings`` and ``with_options`` (not with
    ``with_integration_points``), after the check of :meth:`is_valid_for`.

    .. important::

        Internal use only. The integration points are validated (consistent with the mesh, all
        located) before :meth:`compute`.

    """

    mesh: Any
    integration_points: Any
    point_vertices: numpy.ndarray  # (N_p, N_vpe)
    shape_functions: numpy.ndarray  # (N_p, N_vpe)
    sqrt_weights: numpy.ndarray  # (N_p,)
    structure: SDICTermSparseStructure

    @classmethod
    def compute(cls, mesh: Any, integration_points: Any) -> SDICTermGeometry:
        point_vertices = mesh.elements[integration_points.element_indices]
        shape_functions = fem.compute_shape_functions(integration_points.natural_coordinates, mesh.element_type)
        sqrt_weights = numpy.sqrt(integration_points.weights)
        structure = SDICTermSparseStructure(
            mesh.elements, integration_points.element_indices, shape_functions, mesh.n_vertices, mesh.n_dimensions
        )
        for array in (point_vertices, shape_functions, sqrt_weights):
            array.flags.writeable = False
        return cls(mesh, integration_points, point_vertices, shape_functions, sqrt_weights, structure)

    def is_valid_for(self, mesh: Any, integration_points: Any) -> bool:
        # The same objects (identity), both locked: their data cannot have changed since compute
        return (
            self.mesh is mesh
            and self.integration_points is integration_points
            and bool(mesh.is_locked)
            and bool(integration_points.is_locked)
        )

    def __repr__(self) -> str:
        return f"SDICTermGeometry(n_points={self.sqrt_weights.size}, structure={self.structure!r})"