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

import collections
import warnings
from numbers import Real
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from numpy.typing import ArrayLike

import numpy
import scipy.sparse
import pysolvegn

from ..geometry.integration_points import IntegrationPoints
from .._validation import _as_array, _validate_bool, _validate_choice, _validate_integer, _validate_real
from .gl_correction import (
    GrayLevelGroups,
    compute_gray_level_groups,
    matching_correction,
    matching_correction_exact,
    znssd_correction,
    znssd_correction_exact,
)
from ._sdic_term_caches import SDICTermGeometry, SDICTermOperatorsCache, SDICTermReferenceProjectionCache, SDICTermSparseStructure
from .pairing import Pairing
from .residuals import PairingResiduals, _pairing_residuals
from ._sdic_validation import _validate_gl_correction, _validate_gray_level_groups, _validate_located_integration_points
from .problem import _validate_problem
from .sdic_unknowns import _TRANSFORMATION_KINDS, SDICUnknowns
from .sdic_analysis import SDICAnalysis, _UnknownsLayout

# Gray level corrections: (mode, jacobian) -> function
_GL_CORRECTIONS = {
    ("ZNSSD", "approximate"): znssd_correction,
    ("ZNSSD", "exact"): znssd_correction_exact,
    ("matching", "approximate"): matching_correction,
    ("matching", "exact"): matching_correction_exact,
}

# Reason of the lock of the integration points used by a SDIC term
_INTEGRATION_POINTS_LOCK_MESSAGE = (
    "They are used by a SDIC correlation term (SDICCorrelationTerm), which computed the shape "
    "functions, the weights and the gray level neighborhoods of the points once: modifying them "
    "would make the term inconsistent. The locked points can still be read and used everywhere "
    "(for example by other terms), and stay locked when a new term is created with other points. "
    "To use other points, work on a copy (integration_points.copy()) and give it to a new term "
    "(SDICCorrelationTerm.with_integration_points)."
)

# =============================================================================
# Internal helpers
# =============================================================================
def _block_rows(jacobian: numpy.ndarray) -> numpy.ndarray:
    # Jacobian (N_p, C, n) -> (N_p C, n) with the rows in Fortran order of (N_p, C): row p + c N_p
    N_p, C, n = jacobian.shape
    return jacobian.transpose(1, 0, 2).reshape(C * N_p, n)


# A block of columns of a Jacobian: (index of the range of unknowns, matrix (N_eq, n)). The matrix is a
# numpy array (transformations, or everything with sparse=False) or a scipy csr_matrix (displacements
# with sparse=True). The ranges of unknowns are disjoint: the blocks of a pairing are merged by range.
ColumnBlock = Tuple[int, Union[numpy.ndarray, scipy.sparse.csr_matrix]]

# The helpers below never modify their input matrices: the csr matrices they return share the
# (read-only) structure of their input when it does not change, only the values are allocated.
# The matrices of the structure of the displacement blocks (SDICTermSparseStructure) use its fast paths.


def _csr_with_data(matrix: scipy.sparse.csr_matrix, data: numpy.ndarray) -> scipy.sparse.csr_matrix:
    # A csr matrix with the structure of 'matrix' (shared, not copied) and new values
    result = scipy.sparse.csr_matrix((data, matrix.indices, matrix.indptr), shape=matrix.shape, copy=False)
    result.has_sorted_indices = matrix.has_sorted_indices
    return result


def _add_blocks(
    blocks: List[ColumnBlock], new_blocks: List[ColumnBlock], structure: Optional[SDICTermSparseStructure] = None
) -> None:
    # Add new_blocks to blocks, summing the matrices of the same range (in place on the list)
    for index, matrix in new_blocks:
        for k, (other_index, other) in enumerate(blocks):
            if other_index == index:
                if structure is not None and structure.owns(other) and structure.owns(matrix):
                    blocks[k] = (index, structure.add(other, matrix))
                else:
                    blocks[k] = (index, other + matrix)
                break
        else:
            blocks.append((index, matrix))


def _scale_rows(
    matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix],
    weights: numpy.ndarray,
    structure: Optional[SDICTermSparseStructure] = None,
):
    # Rows of a dense or csr matrix multiplied by the weights (N_eq,)
    if scipy.sparse.issparse(matrix):
        if structure is not None and structure.owns(matrix):
            return structure.scale_rows(matrix, weights)
        matrix = matrix.tocsr()
        return _csr_with_data(matrix, matrix.data * numpy.repeat(weights, numpy.diff(matrix.indptr)))
    return matrix * weights[:, numpy.newaxis]


def _non_finite_rows(
    residuals: numpy.ndarray, blocks: List[ColumnBlock], structure: Optional[SDICTermSparseStructure] = None
) -> numpy.ndarray:
    # Boolean (N_eq,): the equations with a non-finite residual or Jacobian entry. The matrices of
    # the structure are finite by construction (non-finite gradients are explicit zeros).
    bad = ~numpy.isfinite(residuals)
    for _, matrix in blocks:
        if scipy.sparse.issparse(matrix):
            if structure is not None and structure.owns(matrix):
                continue
            finite = numpy.isfinite(matrix.data)
            if not finite.all():
                rows = numpy.repeat(numpy.arange(matrix.shape[0]), numpy.diff(matrix.indptr))
                bad[rows[~finite]] = True
        else:
            bad |= ~numpy.isfinite(matrix).all(axis=1)
    return bad


def _zero_rows(
    matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix],
    bad: numpy.ndarray,
    structure: Optional[SDICTermSparseStructure] = None,
):
    # The rows 'bad' of a dense or csr matrix set to zero (explicit zeros in a csr matrix: same structure)
    if scipy.sparse.issparse(matrix):
        if structure is not None and structure.owns(matrix):
            return structure.zero_rows(matrix, bad)
        matrix = matrix.tocsr()
        return _csr_with_data(matrix, numpy.where(numpy.repeat(bad, numpy.diff(matrix.indptr)), 0.0, matrix.data))
    return numpy.where(bad[:, numpy.newaxis], 0.0, matrix)


def _transposed_product(
    first: Union[numpy.ndarray, scipy.sparse.csr_matrix], second: Union[numpy.ndarray, scipy.sparse.csr_matrix]
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    # first.T @ second: sparse if both are sparse, a numpy array otherwise
    product = first.T @ second
    if scipy.sparse.issparse(product):
        if scipy.sparse.issparse(first) and scipy.sparse.issparse(second):
            return product.tocsr()
        return product.toarray()
    return numpy.asarray(product)


def _build_sdic_operators(
    analysis: SDICAnalysis,
    unknowns: SDICUnknowns,
    layout: _UnknownsLayout,
    integration_points: IntegrationPoints,
    pairings: Sequence[Pairing],
    *,
    mode: str = "rJ",
    gl_correction: Optional[Mapping[str, Any]] = None,
    gray_level_groups: Optional[GrayLevelGroups] = None,
    filter_nans: bool = True,
    sparse: bool = True,
    _use_cache: bool = False,
    _cache_length: int = 1,
    _geometry: Optional[SDICTermGeometry] = None,
) -> Tuple[Callable, Callable, Optional[Callable]]:
    r"""
    Build the operators of the least squares problem on the displacements of the optimized
    frames and the parameters of the optimized transformations (see
    :class:`SDICCorrelationTerm`).

    Returns the residual and Jacobian functions (mode ``"rJ"``, the third output is None) or
    the gradient, Hessian and cost functions (mode ``"gH"``) of the vector of unknowns
    :math:`\vec{p}` (see :class:`SDICUnknowns`). The analysis and the unknowns are validated, and
    ``layout`` is their layout (see :func:`pysdic.sdic.problem._validate_problem`).
    """
    # ------------- options (mode, filter_nans, sparse and pairings are validated by SDICCorrelationTerm) -------------
    gl_options = _validate_gl_correction(gl_correction)
    _use_cache = _validate_bool(_use_cache, "_use_cache")
    _cache_length = _validate_integer(_cache_length, "_cache_length", min_value=1)

    # ------------- unknowns: [displacements of the optimized frames | transformations] -------------
    mesh = analysis.mesh
    N_v, E = mesh.n_vertices, mesh.n_dimensions
    # The positions of the unknowns come from the analysis (layout): the single source of truth of the order of p
    optimized_frames = list(unknowns.optimized_frames)
    entries = list(unknowns.optimized_transformations)
    objects = list(layout.transformations)
    n_unknowns = layout.n_unknowns
    if n_unknowns == 0:
        raise ValueError("At least one optimized frame or one optimized transformation is required.")
    frame_slices = dict(layout.frame_slices)  # Optimized frame -> columns of its displacement
    starts = [layout.transformation_slices[entry].start for entry in entries] + [n_unknowns]  # Columns of the transformations

    def view_blocks(frame: int, index: int) -> List[Tuple[str, int, int]]:
        # The optimized transformations of the camera of a view: (kind, start, stop) of their columns
        camera = analysis.get_view(frame, index).camera
        blocks = []
        for kind in _TRANSFORMATION_KINDS:
            for k, transformation in enumerate(objects):
                if getattr(camera, kind) is transformation:
                    blocks.append((kind, starts[k], starts[k + 1]))
        return blocks

    # ------------- pairings -------------
    active_pairings = []
    for pairing in pairings:
        analysis.validate_pairing(pairing)
        has_unknowns = any(
            frame in frame_slices or view_blocks(frame, index) for frame, index in (pairing.view_1, pairing.view_2)
        )
        if has_unknowns:
            active_pairings.append(pairing)  # Pairings without unknowns are ignored
    used_views = sorted({view for pairing in active_pairings for view in (pairing.view_1, pairing.view_2)})
    blocks = {view: view_blocks(*view) for view in used_views}
    used_frames = {frame for frame, _ in used_views}
    missing_frames = [frame for frame in optimized_frames if frame not in used_frames]
    if missing_frames:
        raise ValueError(f"The optimized frames {missing_frames} appear in no pairing.")
    used_columns = {start for view_blocks_ in blocks.values() for _, start, _ in view_blocks_}
    missing = [entry for entry, start in zip(entries, starts) if start not in used_columns]
    if missing:
        raise ValueError(f"The transformations {missing} are used by no view of the pairings.")

    # ------------- mesh and integration points -------------
    _validate_located_integration_points(mesh, integration_points)
    N_p = integration_points.n_points
    # The data of the mesh and the points: shared with the term this term was created from (with_*), only if
    # it is the same mesh and the same integration points, both locked (their data cannot have changed)
    if _geometry is None:
        geometry = SDICTermGeometry.compute(mesh, integration_points)
    elif _geometry.is_valid_for(mesh, integration_points):
        geometry = _geometry
    else:
        raise RuntimeError(
            "Internal error: the shared geometry of a SDIC term was computed for another mesh or other "
            "integration points (or they are not locked anymore). Please report this bug."
        )
    point_vertices = geometry.point_vertices  # (N_p, N_vpe)
    shape_functions = geometry.shape_functions  # (N_p, N_vpe)
    sqrt_weights = geometry.sqrt_weights  # (N_p,)
    reference_coordinates = mesh.coordinates

    def locate(displacement: numpy.ndarray) -> numpy.ndarray:
        # Positions (N_p, E) of the integration points for the displacement (N_v, E) of the vertices
        return numpy.einsum("pk,pke->pe", shape_functions, (reference_coordinates + displacement)[point_vertices])

    # The frames of the pairings that are not optimized are reference frames: their displacements
    # are read from the analysis at each evaluation (they can be updated between two evaluations,
    # but not removed). They must be set at the creation of the term.
    reference_frames = sorted(frame for frame in used_frames if frame not in frame_slices)
    for frame in reference_frames:
        analysis.get_displacement(frame)  # Error if the displacement is not set
    # The groups of the gray level correction: given (computed once for several terms) or computed here
    groups = _validate_gray_level_groups(gray_level_groups, gl_correction, mesh, integration_points)
    if gl_options is not None and groups is None:
        groups = compute_gray_level_groups(mesh, integration_points, gl_options["neighborhood"])

    n_channels = [analysis.get_view(*pairing.view_1).image.n_channels for pairing in active_pairings]
    equation_starts = numpy.concatenate([[0], numpy.cumsum([N_p * C for C in n_channels])]).astype(int)
    n_total_equations = int(equation_starts[-1])

    # Structure of the displacement blocks (fixed: the integration points are locked by the term)
    structure = geometry.structure if optimized_frames else None

    # Reference views: no unknown in their frame nor in their camera, their projection does not
    # depend on the unknowns (kept by the reference projection cache with use_cache=True)
    def camera_transformations(view: Tuple[int, int]) -> List[Any]:
        camera = analysis.get_view(*view).camera
        return [getattr(camera, kind) for kind in _TRANSFORMATION_KINDS if getattr(camera, kind, None) is not None]

    reference_views = [view for view in used_views if view[0] not in frame_slices and not blocks[view]]
    reference_view_transformations = {view: camera_transformations(view) for view in reference_views}

    # Memory of an evaluation: the projections (and the positions of the points) are computed when a
    # pairing needs them and released after their last use, so that only the views of the current
    # pairings are kept (not the views of all the frames at once)
    view_uses = collections.Counter(view for pairing in active_pairings for view in (pairing.view_1, pairing.view_2))
    frame_uses = collections.Counter(frame for frame, _ in used_views)  # Views of each frame to project

    # ------------- evaluation -------------
    # The ranges of the unknowns, which partition [0, n_unknowns): the displacement of each optimized
    # frame, then each optimized transformation. A Jacobian is assembled by blocks of these ranges.
    unknown_ranges = [(frame_slices[frame].start, frame_slices[frame].stop) for frame in optimized_frames]
    unknown_ranges += [(starts[k], starts[k + 1]) for k in range(len(objects))]
    frame_range = {frame: k for k, frame in enumerate(optimized_frames)}
    transformation_range = {starts[k]: len(optimized_frames) + k for k in range(len(objects))}

    def view_blocks_of(
        result: Any, view: Tuple[int, int], row_factors: Optional[numpy.ndarray] = None
    ) -> List[ColumnBlock]:
        # Blocks of the Jacobian (N_p C, n) of the gray levels of a view (Fortran order: row p + c N_p)
        # w.r.t. the unknowns: displacement of its frame (columns e N_v + v of the frame block, see
        # fem.assemble_displacement_jacobian_matrix, structure of SDICTermSparseStructure), sparse if
        # sparse=True, and its optimized transformations (dense numpy blocks). The rows are multiplied
        # by row_factors (N_p C,) if given (weights and sign of the view in the pairing).
        frame = view[0]
        view_blocks_: List[ColumnBlock] = []
        if frame in frame_slices:
            block = structure.assemble(result.jacobian_dx, row_factors)  # (N_p C, N_v E), csr
            view_blocks_.append((frame_range[frame], block if sparse else block.toarray()))
        for kind, start, stop in blocks[view]:
            block = _block_rows(getattr(result, f"jacobian_d{kind}"))
            if row_factors is not None:
                block = block * row_factors[:, numpy.newaxis]
            view_blocks_.append((transformation_range[start], block))
        return view_blocks_

    def pairing_operators(
        pairing: Pairing, projections: Dict[Tuple[int, int], Any], derivatives: bool
    ) -> Tuple[numpy.ndarray, Optional[List[ColumnBlock]]]:
        # Weighted residuals (N_p C,) and blocks of the weighted Jacobian (N_p C, n) of a pairing, one
        # per range of unknowns (dense numpy blocks, and csr blocks for the displacements if sparse=True)
        result_1, result_2 = projections[pairing.view_1], projections[pairing.view_2]
        derivative = None
        if gl_options is not None:
            corrected = _GL_CORRECTIONS[(gl_options["mode"], gl_options["jacobian"])](
                result_1, result_2, groups, scale=gl_options["scale"]
            )
            if gl_options["jacobian"] == "exact":
                derivative = corrected[2]
                raw_1, raw_2 = result_1, result_2  # The exact derivative applies to the raw Jacobians
            result_1, result_2 = corrected[0], corrected[1]
        C = result_1.x.shape[1]
        row_weights = numpy.tile(sqrt_weights, C)  # Fortran order of (N_p, C)
        residuals = row_weights * (result_2.x - result_1.x).ravel(order="F")
        if not derivatives:
            return residuals, None

        pairing_blocks: List[ColumnBlock] = []
        if derivative is None:
            # w (d g_2 - d g_1): blocks of the second view times +w, of the first one times -w (summed
            # in shared ranges), the weights are applied when the blocks are built
            _add_blocks(pairing_blocks, view_blocks_of(result_2, pairing.view_2, row_weights), structure)
            _add_blocks(pairing_blocks, view_blocks_of(result_1, pairing.view_1, -row_weights), structure)
            return residuals, pairing_blocks
        else:
            # d(g_2' - g_1') = (D_21 - D_11) dg_1 + (D_22 - D_12) dg_2 (D sparse: sparse @ dense -> dense)
            NC = N_p * C
            for k, (view, result) in enumerate(((pairing.view_1, raw_1), (pairing.view_2, raw_2))):
                columns = slice(k * NC, (k + 1) * NC)
                combination = (derivative[NC:, columns] - derivative[:NC, columns]).tocsr()
                _add_blocks(
                    pairing_blocks,
                    [(index, _as_block(combination @ matrix)) for index, matrix in view_blocks_of(result, view)],
                    structure,
                )
        return residuals, [(index, _scale_rows(matrix, row_weights, structure)) for index, matrix in pairing_blocks]

    def _as_block(matrix: Any) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
        # A product with the exact derivative: csr if sparse (displacements with sparse=True), numpy otherwise
        return matrix.tocsr() if scipy.sparse.issparse(matrix) else numpy.asarray(matrix)

    cached_outputs = ("r", "J") if mode == "rJ" else ("g", "H", "c")
    # Caches (use_cache=True only: without cache, everything is computed again at each call, without
    # reading the state of the analysis): the operators of the last evaluations, and the projections
    # of the reference views. The key of the operators: the unknowns, the displacements of the
    # reference frames and the parameters of the transformations of the analysis that are not optimized
    # (fixed list, the optimized ones are set from the unknowns).
    operators_cache = SDICTermOperatorsCache(_cache_length) if _use_cache else None
    reference_cache = SDICTermReferenceProjectionCache(reference_views) if _use_cache else None
    fixed_transformations = [
        t for kind in _TRANSFORMATION_KINDS for t in analysis.get_transformations(kind) if not any(t is o for o in objects)
    ]

    def check_transformations() -> None:
        # No transformation of the analysis is locked (the parameters of the optimized transformations
        # set during the evaluation can always be restored in the 'finally'), and the optimized
        # transformations still have parameters
        if analysis._has_locked_transformation():
            raise RuntimeError(
                f"The transformations of the cameras {analysis._describe_locked_transformations()} were "
                f"locked after the creation of the analysis. The term sets the parameters of its optimized "
                f"transformations during each evaluation and restores them afterwards: all the transformations "
                f"of the analysis must remain modifiable. Create a new analysis with cameras using unlocked "
                f"transformations (e.g. the ReparametrizedTransformation that owns a locked transformation), "
                f"and a new term on it."
            )
        for entry, transformation in zip(entries, objects):
            if transformation.parameters is None:
                raise ValueError(
                    f"The optimized transformation {entry} ({type(transformation).__name__}) has no parameters "
                    f"anymore: they were unset after the creation of the term. Set them again (for example with "
                    f"analysis.set_transformation_parameters) or build a new term."
                )

    def assemble(parameters: numpy.ndarray, requested: Tuple[str, ...]) -> Dict[str, Any]:
        # Compute only the requested operators among "r", "J", "c", "g" and "H"
        need_r, need_J, need_c = "r" in requested, "J" in requested, "c" in requested
        need_g, need_H = "g" in requested, "H" in requested
        derivatives = need_J or need_g or need_H

        positions: Dict[int, numpy.ndarray] = {}  # Computed when a view of the frame is projected
        projections: Dict[Tuple[int, int], Any] = {}  # The projections still needed by the next pairings
        remaining_views = dict(view_uses)  # Pairings that still need each view
        remaining_frames = dict(frame_uses)  # Views of each frame that are not projected yet

        def position(frame: int) -> numpy.ndarray:
            if frame not in positions:
                if frame in frame_slices:
                    displacement = parameters[frame_slices[frame]].reshape((N_v, E), order="F")
                else:
                    displacement = analysis.get_displacement(frame)  # Reference frame: read from the analysis
                positions[frame] = locate(displacement)
            return positions[frame]

        def project(view: Tuple[int, int]) -> Any:
            frame = view[0]
            if reference_cache is not None and view in reference_cache:
                # Reference view: no derivative, the projection is kept while its state does not change
                key = reference_cache.key(analysis.get_displacement(frame), reference_view_transformations[view])
                result = reference_cache.get(view, key)
                if result is None:
                    result = reference_cache.store(view, key, analysis.get_view(*view).image_project(position(frame)))
                return result
            kinds = {kind for kind, _, _ in blocks[view]} if derivatives else set()
            return analysis.get_view(*view).image_project(
                position(frame),
                dx=derivatives and frame in frame_slices,
                dextrinsic="extrinsic" in kinds,
                ddistortion="distortion" in kinds,
                dintrinsic="intrinsic" in kinds,
            )

        def acquire(pairing: Pairing) -> None:
            # Project the views of a pairing that are not projected yet. The positions of a frame are
            # released once all its views are projected.
            for view in (pairing.view_1, pairing.view_2):
                if view not in projections:
                    projections[view] = project(view)
                    frame = view[0]
                    remaining_frames[frame] -= 1
                    if remaining_frames[frame] == 0:
                        positions.pop(frame, None)

        def release(pairing: Pairing) -> None:
            # Release the projections that no next pairing needs
            for view in (pairing.view_1, pairing.view_2):
                remaining_views[view] -= 1
                if remaining_views[view] == 0:
                    del projections[view]

        # Four assemblies, without going through a global sparse structure when it is not needed:
        # - "rJ", sparse=False: r and J are numpy arrays, filled pairing by pairing (block of rows, ranges of columns).
        # - "rJ", sparse=True : r numpy, J built once from the blocks of all the pairings (scipy.sparse.bmat).
        # - "gH", sparse=False: g and H numpy, no global r and J: g += B_k^T r_q and H[k, l] += B_k^T B_l.
        # - "gH", sparse=True : idem, the blocks of H are sparse (displacements) or numpy (transformations),
        #   assembled once at the end (scipy.sparse.bmat over the ranges of the unknowns).
        n_ranges = len(unknown_ranges)
        residuals = numpy.empty(n_total_equations) if need_r else None
        jacobian = numpy.zeros((n_total_equations, n_unknowns)) if (need_J and not sparse) else None
        J_rows: List[List[Any]] = []  # rJ sparse: one row of blocks per pairing (None where no block)
        gradient = numpy.zeros(n_unknowns) if need_g else None
        hessian = numpy.zeros((n_unknowns, n_unknowns)) if (need_H and not sparse) else None
        H_blocks: Dict[Tuple[int, int], Any] = {}  # gH sparse: (k, l) -> sum of B_k^T B_l, with k <= l
        cost = 0.0
        n_bad = 0
        # The transformations are shared by the cameras: their parameters are set for the whole
        # evaluation, then restored (the analysis is never modified by an evaluation)
        saved = [transformation.parameters.copy() for transformation in objects]
        try:
            for k, transformation in enumerate(objects):
                transformation.parameters = parameters[starts[k] : starts[k + 1]]
            for q, pairing in enumerate(active_pairings):
                acquire(pairing)
                r_q, blocks_q = pairing_operators(pairing, projections, derivatives)
                release(pairing)
                if filter_nans:
                    bad = _non_finite_rows(r_q, blocks_q if derivatives else [], structure)
                    n_bad_q = int(numpy.count_nonzero(bad))
                    if n_bad_q > 0:
                        n_bad += n_bad_q
                        r_q = numpy.where(bad, 0.0, r_q)
                        if derivatives:
                            blocks_q = [(index, _zero_rows(matrix, bad, structure)) for index, matrix in blocks_q]
                rows = slice(equation_starts[q], equation_starts[q + 1])
                if need_r:
                    residuals[rows] = r_q
                if need_c:
                    cost += 0.5 * float(r_q @ r_q)
                if need_J:
                    if sparse:
                        row_blocks = [None] * n_ranges
                        for index, matrix in blocks_q:
                            row_blocks[index] = matrix if scipy.sparse.issparse(matrix) else scipy.sparse.csr_matrix(matrix)
                        J_rows.append(row_blocks)  # At least one block per pairing (it has unknowns)
                    else:
                        for index, matrix in blocks_q:
                            start, stop = unknown_ranges[index]
                            jacobian[rows, start:stop] += matrix.toarray() if scipy.sparse.issparse(matrix) else matrix
                if need_g:
                    for index, matrix in blocks_q:
                        start, stop = unknown_ranges[index]
                        gradient[start:stop] += numpy.asarray(matrix.T @ r_q).ravel()
                if need_H:
                    for a, (index_a, matrix_a) in enumerate(blocks_q):
                        for index_b, matrix_b in blocks_q[a:]:
                            k, l = (index_a, index_b) if index_a <= index_b else (index_b, index_a)
                            first, second = (matrix_a, matrix_b) if index_a <= index_b else (matrix_b, matrix_a)
                            product = _transposed_product(first, second)  # B_k^T B_l
                            if sparse:
                                H_blocks[(k, l)] = product if (k, l) not in H_blocks else H_blocks[(k, l)] + product
                            else:
                                (start_k, stop_k), (start_l, stop_l) = unknown_ranges[k], unknown_ranges[l]
                                hessian[start_k:stop_k, start_l:stop_l] += product
                                if k != l:
                                    hessian[start_l:stop_l, start_k:stop_k] += product.T
        finally:
            for transformation, values in zip(objects, saved):
                transformation.parameters = values

        if n_bad == n_total_equations:
            raise ValueError("All the residuals or Jacobians are NaN or infinite, the optimization cannot proceed.")
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} out of {n_total_equations} equations were filtered because of NaN or infinite "
                f"residuals or Jacobians.",
                RuntimeWarning,
            )

        if need_J and sparse:
            # Each range of unknowns is used by at least one pairing (checked at the creation of the term):
            # bmat gets the height of every block row and the width of every block column
            jacobian = scipy.sparse.bmat(J_rows, format="csr")
        if need_H and sparse:
            grid = [[None] * n_ranges for _ in range(n_ranges)]
            for (k, l), product in H_blocks.items():
                block = product if scipy.sparse.issparse(product) else scipy.sparse.csr_matrix(product)
                grid[k][l] = block
                if k != l:
                    grid[l][k] = block.T
            for k in range(n_ranges):  # Diagonal blocks always present: bmat gets the size of every range
                if grid[k][k] is None:
                    size = unknown_ranges[k][1] - unknown_ranges[k][0]
                    grid[k][k] = scipy.sparse.csr_matrix((size, size))
            hessian = scipy.sparse.bmat(grid, format="csr")
        operators = {"r": residuals, "J": jacobian, "c": cost, "g": gradient, "H": hessian}
        return {key: operators[key] for key in requested}

    def evaluate(parameters: ArrayLike, output: str) -> Any:
        # Return the operator 'output' for the given parameters. Without cache, it is computed alone,
        # from the current state of the analysis. With cache, all the operators of the mode are
        # computed at once and kept, and returned again while the key (unknowns + state of the
        # analysis) does not change.
        parameters = _as_array(parameters, "parameters", numpy.float64, shape=(n_unknowns,), finite=True)
        check_transformations()
        if operators_cache is None:
            return assemble(parameters, (output,))[output]
        key = operators_cache.key(
            parameters, [analysis.get_displacement(frame) for frame in reference_frames], fixed_transformations
        )
        operators = operators_cache.get(key)
        if operators is None:
            operators = assemble(parameters, cached_outputs)
            operators_cache.store(key, operators)
        return operators[output]

    # Everything is validated: the integration points are locked (their shape functions and
    # neighborhoods are computed once). Points already locked (for example by another term) are shared.
    if not integration_points.is_locked:
        integration_points._set_locked(True, message=_INTEGRATION_POINTS_LOCK_MESSAGE)

    if mode == "rJ":
        functions = (lambda p: evaluate(p, "r"), lambda p: evaluate(p, "J"), None)
    else:
        functions = (lambda p: evaluate(p, "g"), lambda p: evaluate(p, "H"), lambda p: evaluate(p, "c"))
    # The caches and the structure of the term, for the inspection (tests, profiles): not public
    def compute_values(parameters: ArrayLike, value_pairings: Sequence[Pairing]) -> List[Any]:
        # PairingResiduals of pairings of the analysis for the unknowns (without derivatives, weights nor filter)
        parameters = _as_array(parameters, "parameters", numpy.float64, shape=(n_unknowns,), finite=True)
        check_transformations()
        for pairing in value_pairings:
            analysis.validate_pairing(pairing)
        frames = {frame for pairing in value_pairings for frame in (pairing.frame_index_1, pairing.frame_index_2)}
        displacements, positions = {}, {}
        for frame in frames:
            if frame in frame_slices:
                displacements[frame] = parameters[frame_slices[frame]].reshape((N_v, E), order="F").copy()
            else:
                displacements[frame] = analysis.get_displacement(frame)
            positions[frame] = locate(displacements[frame])
        saved = [transformation.parameters.copy() for transformation in objects]
        try:
            for k, transformation in enumerate(objects):
                transformation.parameters = parameters[starts[k] : starts[k + 1]]
            return [
                _pairing_residuals(analysis, pairing, positions, displacements, gl_options, groups)
                for pairing in value_pairings
            ]
        finally:
            for transformation, values in zip(objects, saved):
                transformation.parameters = values

    caches = {
        "operators": operators_cache,
        "reference_projections": reference_cache,
        "sparse_structure": structure,
        "geometry": geometry,
    }
    for function in functions:
        if function is not None:
            function._sdic_term_caches = caches
            function._sdic_term_values = compute_values
    return functions


# =============================================================================
# Term
# =============================================================================
class SDICCorrelationTerm:
    r"""
    The correlation term of a SDIC analysis: it optimizes jointly the displacements of the
    vertices of the mesh at the optimized frames and the parameters of transformations of the
    cameras of an analysis, by comparing the gray levels of material points between the views
    of several pairings. The ``pysolvegn.Term`` to give to ``pysolvegn.solve`` is the property
    :attr:`term`.

    The term is built on an analysis and on a :class:`pysdic.SDICUnknowns` (the optimized
    frames and transformations), both created by the user and shared with the other objects of
    the problem (:class:`pysdic.SDICTemporalRegularizationTerm`,
    :class:`pysdic.SDICFixedParametrization`). The unknowns are validated by the analysis
    (:meth:`pysdic.SDICAnalysis.validate_unknowns`) at the creation of the term; use
    :func:`pysdic.check_same_problem` to check that all the objects of a problem share the same
    analysis and the same unknowns.

    The same term covers the three usual problems of a SDIC analysis:

    - **measurement**: unknowns with optimized frames only (``SDICUnknowns([1], [])``),
      the cameras are fixed;
    - **calibration**: unknowns with optimized transformations only
      (``SDICUnknowns([], [("extrinsic", 1)])``), the displacements of all the frames
      are fixed;
    - **joint optimization**: both, to measure the displacements while refining some
      parameters of the cameras.

    **Residuals.** The material points are given by integration points of the mesh (the same
    for all the pairings). For a pairing between the view :math:`(f_1, v_1)` and the view
    :math:`(f_2, v_2)` (see :class:`pysdic.Pairing`), the residual of a point :math:`p` (of
    weight :math:`w_p`) and a channel is:

    .. math::

        r_p = \sqrt{w_p} \left( [I_2 \circ P_2](\vec{X}_p + \vec{u}_{f_2}(\vec{X}_p))
        - [I_1 \circ P_1](\vec{X}_p + \vec{u}_{f_1}(\vec{X}_p)) \right)

    where :math:`P_1`, :math:`P_2` are the projections of the cameras, :math:`I_1`,
    :math:`I_2` the images of the views (see :meth:`pysdic.View.image_project`) and
    :math:`\vec{u}_f(\vec{X}_p) = \sum_k N_k(\vec{\xi}_p) \vec{u}_{f,k}` is interpolated from
    the displacements :math:`\vec{u}_{f,k}` of the vertices with the shape functions
    :math:`N_k` at the natural coordinates :math:`\vec{\xi}_p` of the point. The weights
    :math:`\sqrt{w_p}` make the cost an approximation of the integral of the squared gray
    level difference over the surface.

    The residuals are stacked pairing by pairing, in Fortran order of :math:`(N_p, C)` for
    each pairing (row :math:`p + c N_p`, with :math:`C` the number of channels of the
    images of the pairing).

    **Unknowns.** The term works on the complete vector of unknowns :math:`\vec{p}` of
    ``unknowns`` in ``analysis``. It does not build the vectors: they are built by the analysis
    (``pack_unknowns``, ``unpack_unknowns``, ``apply_unknowns``), or by the parametrization with a
    reduced vector :math:`\vec{q}`.

    .. seealso::

        :class:`pysdic.SDICUnknowns` for the order of :math:`\vec{p}`, and
        :class:`pysdic.SDICFixedParametrization` for the reduced vector :math:`\vec{q}`.

    **Frames.** The frames of the pairings that are not optimized are reference frames: their
    displacements are fixed during the optimization, and must be set when the term is created
    (see :meth:`pysdic.SDICAnalysis.set_displacement` and :meth:`pysdic.SDICAnalysis.set_zero_displacement`).
    They are read from the analysis **at each evaluation**: a displacement updated after the
    creation of the term (for example the result of a previous frame) is taken into account.

    **Immutability.** The configuration of the term cannot be changed: :meth:`with_unknowns`,
    :meth:`with_pairings`, :meth:`with_integration_points` and :meth:`with_options` create a new
    term with the same configuration except the given changes (new caches; the gray level groups
    given with ``gray_level_groups`` are shared). The shape functions, the weights and the sparse
    structure of the points only depend on the mesh and on the integration points, both locked:
    they are shared by the new terms of :meth:`with_unknowns`, :meth:`with_pairings` and
    :meth:`with_options`, and computed again by :meth:`with_integration_points`. The objects of a problem
    must share the same analysis and unknowns objects: build the regularization and the
    parametrization on the same new unknowns (their ``with_unknowns``), and check them with
    :func:`pysdic.check_same_problem`.

    **State of the analysis.** The term reads the analysis at each evaluation (displacements of
    the reference frames, parameters of the transformations), so it always reflects its current
    state. Its structure (mesh, views) cannot change (see :class:`pysdic.SDICAnalysis`); a new analysis
    created with :meth:`pysdic.SDICAnalysis.with_mesh` or :meth:`pysdic.SDICAnalysis.with_views` needs a new
    term.

    **Integration points.** The shape functions, the weights and the gray level neighborhoods
    of the integration points are computed once at the creation of the term (the
    neighborhoods can also be given, see ``gray_level_groups``), and the
    integration points are **locked** (see :class:`pysdic.core.lockable.LockableMixin`): they
    can still be used everywhere (for example by other terms) but not modified. Use a copy
    (``integration_points.copy()``) to get modifiable points.

    **Transformations.** A transformation is given as a tuple ``(kind, index)``, with
    ``kind`` in ``"extrinsic"``, ``"distortion"``, ``"intrinsic"`` and ``index`` its index in
    the structure of its kind (see :meth:`pysdic.SDICAnalysis.display_structure` and
    :meth:`pysdic.SDICAnalysis.get_transformation`). A transformation is identified by its object: it
    can be used by several cameras (for example the same intrinsic for two views, or the same
    extrinsic at all the frames), and is then one unknown for all of them.

    - The optimized transformations must have parameters, must not be locked and must not be
      given twice (see :meth:`pysdic.SDICAnalysis.validate_unknowns`).
    - The entries of the analysis must be independent: two different entries
      ``(kind, index)`` must not share a unit transformation (see
      :meth:`pysdic.SDICAnalysis.check_consistency`).
    - Some parameters of a transformation can be kept fixed by building the cameras with a
      :class:`pysdic.ReparametrizedTransformation`, which is then optimized.
    - At each evaluation, the parameters of the optimized transformations are set once for all
      the cameras that use them, then restored: the analysis is never modified by an
      evaluation. Apply the result with :meth:`pysdic.SDICAnalysis.apply_unknowns` (or
      ``parametrization.reparametrized_apply`` with a parametrization).
    - At each evaluation (residuals, Jacobian, gradient, Hessian, cost, :meth:`compute_residuals`),
      no transformation of the analysis must have been locked after its creation, and the optimized
      transformations must still have parameters (an explicit error is raised otherwise, before any
      parameter is set: they can always be restored after the evaluation).

    **Jacobian.** For each view of a pairing, the Jacobian of its gray levels is:

    - for the displacement of its frame (if optimized): the gradient of :math:`I \circ P`
      with respect to the points (``jacobian_dx`` of :meth:`pysdic.View.image_project`) times
      the shape functions (see :func:`pysdic.fem.assemble_displacement_jacobian_matrix`). It
      is sparse: each row depends on the vertices of the element of its point only;
    - for each optimized transformation of its camera: the Jacobian block of
      :meth:`pysdic.View.image_project` (``jacobian_dextrinsic``, ``jacobian_ddistortion`` or
      ``jacobian_dintrinsic``), a dense block of :math:`n_p` columns.

    with a sign :math:`-` for the first view of the pairing and :math:`+` for the second one.
    The contributions of a shared transformation, or of the same frame in a stereo pairing
    at an optimized frame, are summed in the same columns.

    **Pairings.**

    - The pairings without unknown (no optimized frame and no optimized transformation in
      their views, e.g. between two reference frames for a measurement) are ignored.
    - Each optimized frame and each optimized transformation must appear in a (non-ignored)
      pairing.
    - The two views of a pairing must exist and have images with the same number of
      channels (see :meth:`pysdic.SDICAnalysis.validate_pairing`).
    - The visibility of the points is not checked: the points hidden in a view still get the
      gray level of the image at their projection (see :func:`pysdic.compute_visibility` to
      remove them from the integration points).

    **Non-finite values.** The points projected out of the ROI of an image (or out of the
    image) get non-finite gray levels or gradients. With ``filter_nans=True``, the
    corresponding equations are set to zero (residual and Jacobian row) and a
    ``RuntimeWarning`` gives their number. An error is raised if all the equations are
    filtered.

    **Gray level corrections.** The gray levels of the same material point can differ between
    two views (lighting, exposure, reflections, ...). With ``gl_correction``, an affine change
    of the gray levels is removed from the two views of each pairing before computing the
    residuals, with the mean :math:`m` and the standard deviation :math:`s` of groups of
    points (see :func:`pysdic.compute_gray_level_groups`):

    - ``"ZNSSD"``: both views are normalized, :math:`g' = \text{scale} \cdot (g - m) / s`
      (see :func:`pysdic.znssd_correction`). Each pairing gets residuals of the same order of
      magnitude.
    - ``"matching"``: the second view is mapped onto the statistics of the first one,
      :math:`g_2' = \text{scale} \cdot ((g_2 - m_2) \, s_1 / s_2 + m_1)` and
      :math:`g_1' = \text{scale} \cdot g_1` (see :func:`pysdic.matching_correction`).

    The groups only depend on the connectivity of the mesh and on the elements of the
    integration points. By default they are computed at the creation of the term (from
    ``gl_correction["neighborhood"]``). When several terms are built with the same mesh and
    integration points (e.g. one term per frame), compute them once with
    :func:`pysdic.compute_gray_level_groups` and give them with ``gray_level_groups``.

    The derivative of the correction is either ``"approximate"`` (:math:`m` and :math:`s`
    constant, the Jacobians stay sparse) or ``"exact"`` (derivatives of :math:`m` and
    :math:`s` included, see :func:`pysdic.znssd_correction_exact` and
    :func:`pysdic.matching_correction_exact`). The exact derivative couples all the points of
    a group: with a global group (``"neighborhood": None``) it is dense and limited to small
    numbers of points.

    .. warning::

        With several pairings, the ``"matching"`` correction keeps the contrast of the first
        view of each pairing: the pairings no longer have the same weight in the
        optimization. Prefer ``"ZNSSD"`` when several pairings are combined.

    **Modes.** The term can be built in two modes of ``pysolvegn``:

    - ``"rJ"``: residuals :math:`\vec{r}` and Jacobian :math:`J` of all the equations. The
      robust losses of ``pysolvegn`` (``loss``, ``loss_scale``) are available.
    - ``"gH"``: gradient :math:`J^T \vec{r}`, Hessian :math:`J^T J` and cost
      :math:`\frac{1}{2} \|\vec{r}\|^2` assembled pairing by pairing, without storing the
      full Jacobian (lower memory for many points). Only the ``"linear"`` loss is available.

    .. warning::

        Displacements and transformations are not always separable: for example, a rigid
        motion of the surface is equivalent to a rigid motion of the extrinsics of all the
        cameras. Optimize only the transformations that the pairings can determine (e.g. the
        extrinsic of the second camera relatively to the first one, with stereo pairings).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis. The displacements of the reference frames (the frames of the pairings
        that are not optimized) must be set.

    unknowns : :class:`pysdic.SDICUnknowns`
        The unknowns of the problem: the optimized frames and the optimized transformations (at
        least one of them), validated by the analysis.

    integration_points : :class:`pysdic.IntegrationPoints`
        The material points, all located in the mesh of the analysis (use
        ``remove_unlocated``). Their weights weight the residuals (:math:`\sqrt{w_p}`). They
        are locked by the term (not copied).

    pairings : Sequence[:class:`pysdic.Pairing`]
        The pairings of views to compare (a list or a tuple).

    mode : str, optional (default: ``"rJ"``)
        The mode of the term, ``"rJ"`` or ``"gH"``.

    gl_correction : Optional[Mapping[str, Any]], optional
        The gray level correction applied to the two views of each pairing, or None
        (default) for no correction. The keys (all optional) are:

        - ``"mode"``: ``"ZNSSD"`` (default, both views normalized) or ``"matching"`` (the
          second view mapped onto the first one).
        - ``"jacobian"``: ``"approximate"`` (default, constant statistics) or ``"exact"``.
        - ``"neighborhood"``: None (default, global statistics over all the points) or the
          size :math:`n \geq 0` of the neighborhood of the elements for local statistics
          (``0`` for the element alone, ``1`` with the elements sharing a vertex, ...).
        - ``"scale"``: the factor applied after the correction (strictly positive, default
          ``1.0``, e.g. ``255`` for readable gray levels).

    gray_level_groups : Optional[:class:`pysdic.GrayLevelGroups`], optional
        The groups of the gray level correction, computed once with
        :func:`pysdic.compute_gray_level_groups` for the same mesh and integration points, to
        share them between several terms. Requires ``gl_correction`` without the key
        ``"neighborhood"`` (the groups replace it). By default, the groups are computed by the
        term from ``gl_correction["neighborhood"]``.

    weight : Real, optional (default: ``1.0``)
        The weight of the term in the ``pysolvegn`` problem.

    loss : Optional[Union[str, Callable]], optional
        The loss function of the term (see ``pysolvegn.Term``), only for the mode ``"rJ"``.
        ``None`` means ``"linear"``.

    loss_scale : Real, optional (default: ``1.0``)
        The soft threshold of the loss function (strictly positive), in gray levels (after
        correction). Only for the mode ``"rJ"``.

    filter_nans : bool, optional (default: ``True``)
        If True, the equations with non-finite residuals or Jacobians are set to zero, with
        a ``RuntimeWarning``.

    sparse : bool, optional (default: ``True``)
        If True, the Jacobian (``"rJ"``) or the Hessian (``"gH"``) is a
        ``scipy.sparse.csr_matrix``, otherwise a dense array. Keep True when displacements are
        optimized; a dense array can be faster for a calibration with few parameters.

    use_cache : bool, optional (default: ``False``)
        The evaluation strategy of the functions of the term. If False, each function
        (residuals, Jacobian, ...) computes only its own output, projecting the points again.
        If True, the first call for given parameters computes all the outputs of the mode at
        once (``r`` and ``J``, or ``g``, ``H`` and the cost) and keeps them for the next calls
        with the same parameters: fewer projections, more memory. The cache key is a hash of
        the parameters, of the displacements of the reference frames and of the parameters of
        the transformations of the analysis that are not optimized. In addition, the
        projections of the reference views (views whose frame is not optimized and whose camera
        has no optimized transformation, e.g. the reference frame of a measurement) are kept
        between the evaluations: a reference view is projected again only when the
        displacement of its frame or the parameters of the transformations of its camera
        change. A change of the state of the analysis between two calls is never hidden by the
        caches. Only the ``parameters`` of the transformations are compared: their other
        attributes (e.g. the ellipse of a Zernike distortion) are fixed.

    Raises
    ------
    TypeError
        If an input has a wrong type (``analysis`` must be a :class:`pysdic.SDICAnalysis` and
        ``unknowns`` a :class:`pysdic.SDICUnknowns`).

    ValueError
        If there is no unknown, the optimized frames or transformations are not used by the
        pairings, the pairings are not consistent, the integration
        points are not located in the mesh, a needed displacement is not set, a loss other
        than ``"linear"`` is used in mode ``"gH"``, an option of ``gl_correction`` is not
        valid, or ``gray_level_groups`` is given without ``gl_correction``, with
        ``gl_correction["neighborhood"]`` or for other integration points.

    IndexError
        If a frame, a view or a transformation index does not exist.

    Examples
    --------
    **Measurement.** Measure the displacement of the frame 1 with two cameras, the frame 0
    being the reference, then the frame 2 with new unknowns:

    .. code-block:: python

        import pysolvegn
        from pysdic import Pairing, SDICCorrelationTerm, SDICUnknowns

        analysis.set_zero_displacement(0)
        analysis.set_zero_displacement(1)  # Initial guess

        unknowns = SDICUnknowns([1], [])
        pairings = [Pairing(0, 0, 1, 0), Pairing(0, 1, 1, 1), Pairing.stereo(1, 0, 1)]
        correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings,
                                          gl_correction={"mode": "ZNSSD", "scale": 255}, use_cache=True)

        result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
        analysis.apply_unknowns(unknowns, result.parameters)  # Displacement of the frame 1 written in the analysis

        analysis.set_displacement(2, analysis.get_displacement(1))  # Initial guess of the frame 2
        unknowns = SDICUnknowns([2], [])
        correlation = correlation.with_unknowns(  # Unknowns and pairings changed together
            unknowns, pairings=[Pairing(0, 0, 2, 0), Pairing(0, 1, 2, 1), Pairing.stereo(2, 0, 1)]
        )
        result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
        analysis.apply_unknowns(unknowns, result.parameters)

    **Calibration.** Calibrate the extrinsic of the second camera of a stereo setup on the
    reference frame:

    .. code-block:: python

        analysis.display_structure("extrinsic")  # Frame 0: [0, 1], ...
        unknowns = SDICUnknowns([], [("extrinsic", 1)])
        correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, [Pairing.stereo(0, 0, 1)], sparse=False)
        result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
        analysis.apply_unknowns(unknowns, result.parameters)  # Updates all the cameras using the extrinsic

    **Shared gray level groups.** Calibrate the distortion of each frame, the groups of the
    ZNSSD correction being computed once and shared by all the terms:

    .. code-block:: python

        from pysdic import compute_gray_level_groups

        groups = compute_gray_level_groups(analysis.mesh, integration_points, neighborhood=10)
        correlation = None
        for frame in frames:
            unknowns = SDICUnknowns([], [("distortion", index_of_the_distortion_of(frame))])
            pairings = [Pairing.stereo(frame, 0, 1)]
            if correlation is None:
                correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings,
                                                  gl_correction={"mode": "ZNSSD"}, gray_level_groups=groups,
                                                  sparse=False)
            else:
                correlation = correlation.with_unknowns(unknowns, pairings=pairings)
            result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns))
            analysis.apply_unknowns(unknowns, result.parameters)

    **Joint optimization.** Measure the displacement of the frame 1 and refine the extrinsic
    of the second camera:

    .. code-block:: python

        unknowns = SDICUnknowns([1], [("extrinsic", 1)])
        pairings = [Pairing(0, 0, 1, 0), Pairing(0, 1, 1, 1), Pairing.stereo(0, 0, 1), Pairing.stereo(1, 0, 1)]
        correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)
        result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
        displacements, parameters = analysis.unpack_unknowns(unknowns, result.parameters)

    """

    # Options of the term (everything except the unknowns, the integration points and the pairings)
    _OPTION_KEYS = (
        "mode",
        "gl_correction",
        "gray_level_groups",
        "weight",
        "loss",
        "loss_scale",
        "filter_nans",
        "sparse",
        "use_cache",
    )

    def __init__(
        self,
        analysis: SDICAnalysis,
        unknowns: SDICUnknowns,
        integration_points: IntegrationPoints,
        pairings: Sequence[Pairing],
        *,
        mode: str = "rJ",
        gl_correction: Optional[Mapping[str, Any]] = None,
        gray_level_groups: Optional[GrayLevelGroups] = None,
        weight: Real = 1.0,
        loss: Optional[Union[str, Callable]] = None,
        loss_scale: Real = 1.0,
        filter_nans: bool = True,
        sparse: bool = True,
        use_cache: bool = False,
        _geometry: Optional[SDICTermGeometry] = None,
    ) -> None:
        # _geometry (private): the data of the mesh and the points of the term this one is created from by a
        # with_* method (see SDICTermGeometry), checked before use. Never given by the user.
        layout = _validate_problem(analysis, unknowns)
        if not isinstance(pairings, (list, tuple)):
            raise TypeError(f"Input 'pairings' must be a list or a tuple of Pairing. Got type {type(pairings)}.")
        mode = _validate_choice(mode, "mode", ("rJ", "gH"))
        if isinstance(gl_correction, Mapping):
            gl_correction = dict(gl_correction)  # Copy of the user's dictionary
        weight = _validate_real(weight, "weight", finite=True)
        loss_scale = _validate_real(loss_scale, "loss_scale", positive=True, finite=True)
        filter_nans = _validate_bool(filter_nans, "filter_nans")
        sparse = _validate_bool(sparse, "sparse")
        use_cache = _validate_bool(use_cache, "use_cache")
        if mode == "gH" and (loss not in (None, "linear") or loss_scale != 1.0):
            raise ValueError(
                f"Only the 'linear' loss (with loss_scale=1.0) is available in mode 'gH', got {loss!r} "
                f"with loss_scale={loss_scale}."
            )

        func_1, func_2, cost_func = _build_sdic_operators(
            analysis,
            unknowns,
            layout,
            integration_points,
            list(pairings),
            mode=mode,
            gl_correction=gl_correction,
            gray_level_groups=gray_level_groups,
            filter_nans=filter_nans,
            sparse=sparse,
            _use_cache=use_cache,
            _geometry=_geometry,
        )
        if mode == "rJ":
            term = pysolvegn.Term.from_rJ(
                residual_func=func_1, jacobian_func=func_2, weight=weight, loss=loss, loss_scale=loss_scale
            )
        else:
            term = pysolvegn.Term.from_gH(gradient_func=func_1, hessian_func=func_2, weight=weight, cost_func=cost_func)

        self._analysis = analysis
        self._unknowns = unknowns
        self._integration_points = integration_points
        self._pairings = tuple(pairings)
        self._options = {
            "mode": mode,
            "gl_correction": gl_correction,
            "gray_level_groups": gray_level_groups,
            "weight": weight,
            "loss": loss,
            "loss_scale": loss_scale,
            "filter_nans": filter_nans,
            "sparse": sparse,
            "use_cache": use_cache,
        }
        self._term = term
        self._caches = func_1._sdic_term_caches
        self._geometry = self._caches["geometry"]
        self._values = func_1._sdic_term_values

    # ------------------------------------------------------------------ New terms
    def with_unknowns(self, unknowns: SDICUnknowns, *, pairings: Optional[Sequence[Pairing]] = None) -> SDICCorrelationTerm:
        r"""
        A new term with other unknowns (e.g. the next frame of a measurement), the same
        integration points and options.

        Parameters
        ----------
        unknowns : :class:`pysdic.SDICUnknowns`
            The new unknowns.

        pairings : Optional[Sequence[:class:`pysdic.Pairing`]], optional
            New pairings, given with the unknowns when the new optimized frames or
            transformations are not in the current pairings. By default, the current pairings.

        """
        return SDICCorrelationTerm(
            self._analysis,
            unknowns,
            self._integration_points,
            self._pairings if pairings is None else pairings,
            **self._options,
            _geometry=self._geometry,
        )

    def with_pairings(self, pairings: Sequence[Pairing]) -> SDICCorrelationTerm:
        r"""A new term with other pairings, the same unknowns, integration points and options."""
        return SDICCorrelationTerm(
            self._analysis, self._unknowns, self._integration_points, pairings, **self._options, _geometry=self._geometry
        )

    def with_integration_points(self, integration_points: IntegrationPoints) -> SDICCorrelationTerm:
        r"""A new term with other integration points, the same unknowns, pairings and options (the shape
        functions and the sparse structure are computed again for the new points)."""
        return SDICCorrelationTerm(self._analysis, self._unknowns, integration_points, self._pairings, **self._options)

    def with_options(self, **changes: Any) -> SDICCorrelationTerm:
        r"""
        A new term with the same unknowns, integration points and pairings, and other options
        (any keyword parameter of the constructor), e.g. ``with_options(weight=10.0, use_cache=True)``.
        """
        unknown = sorted(set(changes) - set(self._OPTION_KEYS))
        if unknown:
            raise TypeError(f"Unknown options {unknown}. Valid options: {list(self._OPTION_KEYS)}.")
        return SDICCorrelationTerm(
            self._analysis,
            self._unknowns,
            self._integration_points,
            self._pairings,
            **{**self._options, **changes},
            _geometry=self._geometry,
        )

    # ------------------------------------------------------------------ Properties
    @property
    def unknowns(self) -> SDICUnknowns:
        r"""The unknowns of the problem (given at the creation, shared with the other objects of the problem)."""
        return self._unknowns

    @property
    def analysis(self) -> SDICAnalysis:
        r"""The analysis (given at the creation, shared with the other objects of the problem)."""
        return self._analysis

    @property
    def term(self) -> pysolvegn.Term:
        r"""The ``pysolvegn.Term``, to give to ``pysolvegn.solve``."""
        return self._term

    @property
    def integration_points(self) -> IntegrationPoints:
        r"""The integration points (locked)."""
        return self._integration_points

    @property
    def pairings(self) -> Tuple[Pairing, ...]:
        r"""The pairings, including those without unknown (ignored by the term)."""
        return self._pairings

    @property
    def mode(self) -> str:
        r"""The mode, ``"rJ"`` or ``"gH"``."""
        return self._options["mode"]

    @property
    def gl_correction(self) -> Optional[Dict[str, Any]]:
        r"""The gray level correction as given (a copy), or None."""
        value = self._options["gl_correction"]
        return dict(value) if isinstance(value, dict) else value

    @property
    def gray_level_groups(self) -> Optional[GrayLevelGroups]:
        r"""The gray level groups given at the creation, or None."""
        return self._options["gray_level_groups"]

    @property
    def weight(self) -> float:
        r"""The weight of the term in the ``pysolvegn`` problem."""
        return self._options["weight"]

    @property
    def loss(self) -> Optional[Union[str, Callable]]:
        r"""The loss function of the term (mode ``"rJ"``)."""
        return self._options["loss"]

    @property
    def loss_scale(self) -> float:
        r"""The soft threshold of the loss function."""
        return self._options["loss_scale"]

    @property
    def filter_nans(self) -> bool:
        r"""Whether the non-finite equations are set to zero."""
        return self._options["filter_nans"]

    @property
    def sparse(self) -> bool:
        r"""Whether the Jacobian (``"rJ"``) or the Hessian (``"gH"``) is sparse."""
        return self._options["sparse"]

    @property
    def use_cache(self) -> bool:
        r"""Whether the caches of the operators and of the reference projections are used."""
        return self._options["use_cache"]

    @property
    def options(self) -> Dict[str, Any]:
        r"""The options of the term (a copy): the keyword parameters of the constructor."""
        return dict(self._options)

    # ------------------------------------------------------------------ Residuals
    def compute_residuals(
        self, parameters: Optional[ArrayLike] = None, pairings: Optional[Sequence[Pairing]] = None
    ) -> List["PairingResiduals"]:
        r"""
        Compute the gray levels and the residuals of the integration points of the term in the views of
        pairings, for a vector of unknowns: what the term compares, with its gray level correction, without the
        weights of the points and without filtering the non-finite values (see
        :class:`pysdic.PairingResiduals`).

        Use it to inspect the term at any vector of unknowns (e.g. at each iteration of a solver history),
        then :func:`pysdic.compute_pixel_residual_maps` or :func:`pysdic.display_residual_maps` to draw the
        residuals on the views. The analysis is not modified.

        Parameters
        ----------
        parameters : Optional[ArrayLike], optional
            The complete vector of unknowns, of shape :math:`(N_f N_v E + \sum n_p,)` (with a parametrization,
            ``parametrization.p_func(q)``). By default, the current state of the analysis
            (``analysis.pack_unknowns(unknowns)``).

        pairings : Optional[Sequence[:class:`pysdic.Pairing`]], optional
            The pairings to evaluate: any valid pairing of the analysis (also a pairing without unknown, or not
            in the term). By default, the pairings of the term.

        Returns
        -------
        List[:class:`pysdic.PairingResiduals`]
            The residuals of each pairing, in the order of ``pairings``.

        """
        if parameters is None:
            parameters = self._analysis.pack_unknowns(self._unknowns)
        if pairings is None:
            pairings = self.pairings
        if not isinstance(pairings, (list, tuple)):
            raise TypeError(f"Input 'pairings' must be a list or a tuple of Pairing. Got type {type(pairings)}.")
        return self._values(parameters, list(pairings))

    def __repr__(self) -> str:
        return (
            f"SDICCorrelationTerm(mode={self.mode!r}, {len(self.pairings)} pairings, {self._unknowns!r}, "
            f"{self.integration_points.n_points} integration points, "
            f"sparse={self.sparse}, use_cache={self.use_cache})"
        )