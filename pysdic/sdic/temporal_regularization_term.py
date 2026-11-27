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

from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import scipy.sparse
import pysolvegn
from numpy.typing import ArrayLike

from ..numeric import assemble_finite_difference_matrix
from .._validation import _as_array, _validate_bool, _validate_choice, _validate_integer, _validate_real
from ._sdic_validation import _validate_frames
from .problem import _validate_problem
from .sdic_analysis import SDICAnalysis
from .sdic_unknowns import SDICUnknowns


# =============================================================================
# Internal helpers
# =============================================================================
def _time_weights(times: numpy.ndarray, rows: numpy.ndarray, max_gap: Optional[float]) -> numpy.ndarray:
    # Duration represented by the evaluated samples rows: half the distance between the
    # previous and the next samples of the same segment (half the step at the ends)
    segment = numpy.zeros(times.size, dtype=numpy.int64)
    if max_gap is not None:
        segment[1:] = numpy.cumsum(numpy.diff(times) > max_gap)
    previous = numpy.maximum(rows - 1, 0)
    following = numpy.minimum(rows + 1, times.size - 1)
    previous = numpy.where(segment[previous] == segment[rows], previous, rows)
    following = numpy.where(segment[following] == segment[rows], following, rows)
    return 0.5 * (times[following] - times[previous])


def _build_regularization_operators(
    analysis: SDICAnalysis,
    unknowns: SDICUnknowns,
    order: Integral,
    *,
    frames: Optional[Sequence[Integral]] = None,
    accuracy: Integral = 2,
    boundary: str = "one_sided",
    max_gap: Optional[Real] = None,
    target: Optional[ArrayLike] = None,
    component_weights: Optional[ArrayLike] = None,
    row_weights: str = "uniform",
    sparse: bool = True,
) -> Tuple[numpy.ndarray, Union[numpy.ndarray, scipy.sparse.csr_matrix], Callable[[], numpy.ndarray], int]:
    r"""
    Build the operators of the linear residual :math:`\vec{r}(\vec{p}) = J \vec{p} + \vec{b}`
    of the temporal regularization (see :class:`SDICTemporalRegularizationTerm`).

    Returns the indices of the evaluated frames, the constant Jacobian :math:`J` of shape
    :math:`(N_r N_v E, n_{unknowns})`, a function computing the vector :math:`\vec{b}` from
    the current displacements of the fixed frames of the analysis (read at each call), and
    the number of unknowns.
    """
    layout = _validate_problem(analysis, unknowns)  # The positions of the unknowns in p, from the analysis
    mesh = analysis.mesh
    N_v, E, N_f = mesh.n_vertices, mesh.n_dimensions, analysis.n_frames
    n_dim = N_v * E

    # ------------- unknowns: [displacements of the optimized frames | transformations] -------------
    optimized_frames = list(unknowns.optimized_frames)
    if not optimized_frames:
        raise ValueError("The temporal regularization needs at least one optimized frame in 'unknowns'.")
    n_unknowns = layout.n_unknowns
    frame_starts = {frame: columns.start for frame, columns in layout.frame_slices.items()}

    # ------------- frames of the regularization, in chronological order -------------
    if frames is None:
        frames = sorted(set(optimized_frames) | {f for f in range(N_f) if analysis.has_displacement(f)})
    else:
        frames = sorted(_validate_frames(analysis, frames, "frames", allow_empty=False))
        missing = [frame for frame in optimized_frames if frame not in frames]
        if missing:
            raise ValueError(f"The optimized frames {missing} must be in 'frames'.")
    for frame in frames:
        if frame not in frame_starts and not analysis.has_displacement(frame):
            raise ValueError(f"The displacement of the frame {frame} (fixed frame of the regularization) is not set.")

    times = analysis.times
    if times is None:
        raise ValueError(
            "The times of acquisition of the frames are not defined in the analysis: the temporal "
            "regularization needs them to compute the finite differences. Create the analysis with "
            "SDICAnalysis(..., times=times), or get an analysis with times with analysis.with_times(times)."
        )
    frame_times = times[frames]  # Strictly increasing: the frames are sorted and the times increasing

    # ------------- options -------------
    order = _validate_integer(order, "order", min_value=1)
    boundary = _validate_choice(boundary, "boundary", ("one_sided", "valid"))
    max_gap = _validate_real(max_gap, "max_gap", positive=True, finite=True, allow_none=True)
    row_weights = _validate_choice(row_weights, "row_weights", ("uniform", "time"))
    sparse = _validate_bool(sparse, "sparse")
    if component_weights is None:
        component_weights = numpy.ones(E)
    else:
        component_weights = _as_array(
            component_weights, "component_weights", numpy.float64, shape=(E,), shape_hint=f"({E},)", finite=True
        )
        if numpy.any(component_weights < 0):
            raise ValueError("Input 'component_weights' must be non-negative.")
    if target is not None:
        target = _as_array(
            target, "target", numpy.float64, shape=(N_f, N_v, E), shape_hint=f"({N_f}, {N_v}, {E})", finite=True
        )

    # ------------- finite difference operator on the frames of the regularization -------------
    D, rows = assemble_finite_difference_matrix(
        frame_times, order, accuracy=accuracy, boundary=boundary, max_gap=max_gap, n_dim=n_dim,
        sparse=True, return_indices=True,
    )  # (N_r n_dim, len(frames) n_dim), rows in chronological order of the frames
    D = D.tocoo()
    is_optimized = numpy.array([frame in frame_starts for frame in frames], dtype=bool)  # Per frame of 'frames'
    starts = numpy.array([frame_starts.get(frame, -1) for frame in frames], dtype=numpy.int64)
    block, local = numpy.divmod(D.col, n_dim)  # Frame (index in 'frames') and component of each entry
    optimized = is_optimized[block]

    # Rows involving no optimized frame are constant: they are removed
    kept_blocks = numpy.unique(D.row[optimized] // n_dim)
    rows = rows[kept_blocks]
    row_map = numpy.full(D.shape[0] // n_dim, -1, dtype=numpy.int64)
    row_map[kept_blocks] = numpy.arange(kept_blocks.size)
    keep = row_map[D.row // n_dim] >= 0
    entry_rows = row_map[D.row // n_dim] * n_dim + D.row % n_dim
    n_rows = kept_blocks.size * n_dim

    # Jacobian: entries of the optimized frames in the columns of their unknowns
    selected = keep & optimized
    J = scipy.sparse.csr_matrix(
        (D.data[selected], (entry_rows[selected], starts[block[selected]] + local[selected])), shape=(n_rows, n_unknowns)
    )

    # Vector b = B u_fixed - a: B maps the displacements of the fixed frames used by the kept rows
    # (concatenated in Fortran order) to the rows. The displacements are read at each evaluation.
    fixed = keep & ~optimized
    fixed_blocks = numpy.unique(block[fixed])  # Fixed frames used (indices in 'frames')
    fixed_frames: List[int] = [frames[k] for k in fixed_blocks]
    fixed_index = numpy.full(len(frames), -1, dtype=numpy.int64)
    fixed_index[fixed_blocks] = numpy.arange(fixed_blocks.size)
    B = scipy.sparse.csr_matrix(
        (D.data[fixed], (entry_rows[fixed], fixed_index[block[fixed]] * n_dim + local[fixed])),
        shape=(n_rows, fixed_blocks.size * n_dim),
    )
    evaluated_frames = numpy.asarray(frames, dtype=numpy.int64)[rows]
    target_part = numpy.zeros(n_rows)
    if target is not None and rows.size > 0:
        target_part = -target[evaluated_frames].transpose(0, 2, 1).reshape(-1)  # Fortran order of (N_v, E) per frame

    # Row weights: component weights (Fortran order: component e on the rows e N_v ... (e + 1) N_v - 1
    # of each frame block) and optionally the duration represented by each evaluated frame
    sqrt_weights = numpy.tile(numpy.repeat(numpy.sqrt(component_weights), N_v), rows.size)
    if row_weights == "time":
        sqrt_weights *= numpy.repeat(numpy.sqrt(_time_weights(frame_times, rows, max_gap)), n_dim)
    W = scipy.sparse.diags(sqrt_weights)
    J = W @ J
    B = (W @ B).tocsr()
    target_part = sqrt_weights * target_part

    def compute_b() -> numpy.ndarray:
        # Current displacements of the fixed frames (they can be updated, not removed)
        if not fixed_frames:
            return target_part.copy()
        values = numpy.concatenate([analysis.get_displacement(frame).ravel(order="F") for frame in fixed_frames])
        return B @ values + target_part

    return evaluated_frames, (J.tocsr() if sparse else J.toarray()), compute_b, n_unknowns


# =============================================================================
# Term
# =============================================================================
class SDICTemporalRegularizationTerm:
    r"""
    The temporal regularization of the displacements of the vertices of the mesh
    along the frames of an analysis, by penalizing their temporal derivative of order
    :math:`n`. The ``pysolvegn.Term`` to give to ``pysolvegn.solve`` is the property
    :attr:`term`.

    The term works on the complete vector of unknowns :math:`\vec{p}` of ``unknowns`` in
    ``analysis``: give it the same analysis and the same unknowns objects as the correlation term
    to combine both terms in the same problem (see :func:`pysdic.check_same_problem`). The
    regularization does not depend on the transformations: their columns of the Jacobian are
    zero. The term does not build the vectors (see the ``*_unknowns`` methods of the analysis).

    .. seealso::

        :class:`pysdic.SDICUnknowns` for the order of :math:`\vec{p}`, and
        :class:`pysdic.SDICFixedParametrization` for the reduced vector :math:`\vec{q}`.

    The object is immutable: :meth:`with_unknowns` and :meth:`with_options` create a new
    regularization (e.g. with other unknowns, another order or another weight).

    **Residuals.** The frames of the regularization (``frames``) are sorted by time. The
    optimized frames take their displacement in :math:`\vec{p}_U`, the other ones (fixed
    frames) their displacement set in the analysis. For each evaluated frame :math:`f_i`,
    each vertex :math:`v` and each component :math:`e`:

    .. math::

        r_{i,v,e} = \sqrt{w_e \, \omega_i} \left( \sum_{k} c_{i,k} \, u_{f_k, v, e} - a_{f_i, v, e} \right)

    where :math:`c_{i,k}` are the finite difference coefficients of the derivative of order
    :math:`n` at the time :math:`t_{f_i}` (see
    :func:`pysdic.numeric.assemble_finite_difference_matrix`), :math:`a` the ``target``
    derivative (zero by default), :math:`w_e` the ``component_weights`` and
    :math:`\omega_i` the ``row_weights``.

    **Sampling.** The coefficients are computed for the times of acquisition of the frames,
    defined in the analysis (:attr:`pysdic.SDICAnalysis.times`, required): the frames that are not in
    the regularization (frames without displacement, or not in ``frames``) only make the step
    between their neighbors larger.
    No displacement is interpolated or extrapolated:

    - centered stencils when the neighbors are uniformly spaced, the most centered window of
      :math:`n + p` frames otherwise (accuracy :math:`p`);
    - at the first and last frames, one-sided stencils (``boundary="one_sided"``) or no
      equation (``boundary="valid"``);
    - with ``max_gap``, no stencil spans a step larger than ``max_gap``: the frames on each
      side are regularized separately.

    **Equations.** Only the equations involving at least one optimized frame are kept (the
    other ones are constant). The equations of the fixed frames next to optimized frames
    are kept: for example, the equation of the reference frame links it with the first
    optimized frames.

    **Weights of the rows.** The one-sided stencils have larger coefficients than the
    centered ones: the first and last frames are more regularized. The coefficients of a
    frame across a large step are smaller: a displacement change over a longer duration is a
    smaller derivative. With ``row_weights="time"``, each equation is weighted by the
    duration :math:`\omega_i` it represents (half the distance between the previous and the
    next frames, half the step at the ends), so that the cost approximates
    :math:`\int \| \vec{u}^{(n)}(t) - \vec{a}(t) \|^2 dt` independently of the sampling of the
    frames.

    The residual is linear: :math:`\vec{r} = J \vec{p} + \vec{b}`, with a constant Jacobian
    computed once. The vector :math:`\vec{b}` (contribution of the fixed frames, minus the
    target) is computed **at each evaluation** from the current displacements of the fixed
    frames: a displacement updated after the creation of the term (for example the result of
    a previous measurement) is taken into account.

    .. note::

        The frames of the regularization are fixed at the creation of the term: a frame whose
        displacement is set later is not added (build a new term to include it). The
        displacements of the fixed frames cannot be removed from the analysis, so they are
        always available.

    The residuals are ordered by evaluated frame (chronological order), each frame in
    Fortran order of :math:`(N_v, E)` (all the :math:`x` components, then all the :math:`y`
    components, ...).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with the times of the frames defined (:attr:`pysdic.SDICAnalysis.times`)
        and the displacements of the fixed frames of the regularization set.

    unknowns : :class:`pysdic.SDICUnknowns`
        The unknowns of the problem, with at least one optimized frame, validated by the
        analysis. The optimized transformations only define the size of the vector of unknowns.

    order : Integral
        The order :math:`n \geq 1` of the penalized derivative (1 for the velocity, 2 for the
        acceleration).

    frames : Optional[Sequence[Integral]], optional
        The frames of the regularization, which must contain the optimized frames. By
        default, the optimized frames and all the frames whose displacement is set.

    accuracy : Integral, optional (default: ``2``)
        The order :math:`p` of the error of the finite differences, a positive even integer.

    boundary : str, optional (default: ``"one_sided"``)
        The equations at the first and last frames: ``"one_sided"`` or ``"valid"``.

    max_gap : Optional[Real], optional
        The largest step between two frames that a stencil can span, in the unit of the times
        of the analysis. By default, no limit.

    target : Optional[ArrayLike], optional
        The target derivative :math:`a`, of shape :math:`(N_f, N_v, E)` (only the evaluated
        frames are used). By default, zero.

    component_weights : Optional[ArrayLike], optional
        The non-negative weights :math:`w_e` of the components of the displacement, of shape
        :math:`(E,)`, e.g. ``[1, 1, 0]`` to regularize only the in-plane components. By
        default, 1.

    row_weights : str, optional (default: ``"uniform"``)
        The weights :math:`\omega_i` of the evaluated frames: ``"uniform"`` (1) or ``"time"``
        (duration represented by each frame, see above).

    mode : str, optional (default: ``"rJ"``)
        The mode of the term: ``"rJ"`` (residuals and Jacobian) or ``"gH"`` (gradient
        :math:`J^T \vec{r}`, Hessian :math:`J^T J` and cost, only the ``"linear"`` loss).

    weight : Real, optional (default: ``1.0``)
        The weight of the term in the ``pysolvegn`` problem.

    loss : Optional[Union[str, Callable]], optional
        The loss function of the term (see ``pysolvegn.Term``), only for the mode ``"rJ"``.
        ``None`` means ``"linear"``.

    loss_scale : Real, optional (default: ``1.0``)
        The soft threshold of the loss function. Only for the mode ``"rJ"``.

    sparse : bool, optional (default: ``True``)
        If True, the Jacobian (``"rJ"``) or the Hessian (``"gH"``) is a
        ``scipy.sparse.csr_matrix``, otherwise a dense array.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the times of the analysis are not defined, the frames or the transformations are not
        consistent, the displacement of a fixed frame of the regularization is not set, a
        loss other than ``"linear"`` is used in mode ``"gH"``, or an input has a wrong shape
        or value.

    Examples
    --------
    Measure the displacements of the frames 1 to 9 with a penalty on the acceleration, the
    frame 0 being the reference:

    .. code-block:: python

        import pysolvegn
        from pysdic import SDICCorrelationTerm, SDICTemporalRegularizationTerm, SDICUnknowns

        unknowns = SDICUnknowns(list(range(1, 10)), [])
        correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)
        regularization = SDICTemporalRegularizationTerm(analysis, unknowns, order=2, weight=1e-2)

        result = pysolvegn.solve([correlation.term, regularization.term], analysis.pack_unknowns(unknowns), max_iteration=20)
        analysis.apply_unknowns(unknowns, result.parameters)

        # The same problem with a stronger regularization of the velocity
        regularization = regularization.with_options(order=1, weight=1e-1)

    """

    # Options of the regularization (everything except the analysis and the unknowns)
    _OPTION_KEYS = (
        "order", "frames", "accuracy", "boundary", "max_gap", "target", "component_weights", "row_weights",
        "mode", "weight", "loss", "loss_scale", "sparse",
    )

    def __init__(
        self,
        analysis: SDICAnalysis,
        unknowns: SDICUnknowns,
        order: Integral,
        *,
        frames: Optional[Sequence[Integral]] = None,
        accuracy: Integral = 2,
        boundary: str = "one_sided",
        max_gap: Optional[Real] = None,
        target: Optional[ArrayLike] = None,
        component_weights: Optional[ArrayLike] = None,
        row_weights: str = "uniform",
        mode: str = "rJ",
        weight: Real = 1.0,
        loss: Optional[Union[str, Callable]] = None,
        loss_scale: Real = 1.0,
        sparse: bool = True,
    ) -> None:
        mode = _validate_choice(mode, "mode", ("rJ", "gH"))
        weight = _validate_real(weight, "weight", finite=True)
        loss_scale = _validate_real(loss_scale, "loss_scale", positive=True, finite=True)
        if mode == "gH" and (loss not in (None, "linear") or loss_scale != 1.0):
            raise ValueError(
                f"Only the 'linear' loss (with loss_scale=1.0) is available in mode 'gH', got {loss!r} "
                f"with loss_scale={loss_scale}."
            )
        evaluated_frames, J, compute_b, n_unknowns = _build_regularization_operators(
            analysis,
            unknowns,
            order,
            frames=frames,
            accuracy=accuracy,
            boundary=boundary,
            max_gap=max_gap,
            target=target,
            component_weights=component_weights,
            row_weights=row_weights,
            sparse=sparse,
        )

        def residuals(parameters: ArrayLike) -> numpy.ndarray:
            parameters = _as_array(parameters, "parameters", numpy.float64, shape=(n_unknowns,), finite=True)
            return J @ parameters + compute_b()

        if mode == "rJ":
            term = pysolvegn.Term.from_rJ(
                residual_func=residuals, jacobian_func=lambda parameters: J, weight=weight, loss=loss, loss_scale=loss_scale
            )
        else:
            hessian = (J.T @ J).tocsr() if sparse else J.T @ J  # Constant
            term = pysolvegn.Term.from_gH(
                gradient_func=lambda parameters: J.T @ residuals(parameters),
                hessian_func=lambda parameters: hessian,
                weight=weight,
                cost_func=lambda parameters: 0.5 * float(numpy.sum(residuals(parameters) ** 2)),
            )

        self._analysis = analysis
        self._unknowns = unknowns
        self._options = {
            "order": int(order),
            "frames": None if frames is None else tuple(int(frame) for frame in frames),
            "accuracy": accuracy,
            "boundary": boundary,
            "max_gap": max_gap,
            "target": None if target is None else numpy.array(target, dtype=numpy.float64),
            "component_weights": None if component_weights is None else numpy.array(component_weights, dtype=numpy.float64),
            "row_weights": row_weights,
            "mode": mode,
            "weight": weight,
            "loss": loss,
            "loss_scale": loss_scale,
            "sparse": bool(sparse),
        }
        evaluated_frames.setflags(write=False)
        self._evaluated_frames = evaluated_frames
        self._term = term

    # ------------------------------------------------------------------ Properties
    @property
    def unknowns(self) -> SDICUnknowns:
        r"""The unknowns of the problem (shared with the correlation term)."""
        return self._unknowns

    @property
    def analysis(self) -> SDICAnalysis:
        r"""The analysis (shared with the other objects of the problem)."""
        return self._analysis

    @property
    def term(self) -> pysolvegn.Term:
        r"""The ``pysolvegn.Term``, to give to ``pysolvegn.solve``."""
        return self._term

    @property
    def order(self) -> int:
        r"""The order :math:`n` of the penalized derivative."""
        return self._options["order"]

    @property
    def weight(self) -> float:
        r"""The weight of the term in the ``pysolvegn`` problem."""
        return self._options["weight"]

    @property
    def options(self) -> Dict[str, Any]:
        r"""The options of the regularization (a copy): the parameters of the constructor except ``analysis`` and ``unknowns``."""
        return dict(self._options)

    @property
    def evaluated_frames(self) -> numpy.ndarray:
        r"""The frames with an equation, in chronological order (read-only)."""
        return self._evaluated_frames

    # ------------------------------------------------------------------ New terms
    def with_unknowns(self, unknowns: SDICUnknowns) -> SDICTemporalRegularizationTerm:
        r"""A new regularization with other unknowns and the same options."""
        return SDICTemporalRegularizationTerm(self._analysis, unknowns, **self._options)

    def with_options(self, **changes: Any) -> SDICTemporalRegularizationTerm:
        r"""
        A new regularization with the same unknowns and other options (any parameter of the
        constructor except ``analysis`` and ``unknowns``), e.g. ``with_options(order=1, weight=1e-1)``.
        """
        unknown = sorted(set(changes) - set(self._OPTION_KEYS))
        if unknown:
            raise TypeError(f"Unknown options {unknown}. Valid options: {list(self._OPTION_KEYS)}.")
        return SDICTemporalRegularizationTerm(self._analysis, self._unknowns, **{**self._options, **changes})

    def __repr__(self) -> str:
        return (
            f"SDICTemporalRegularizationTerm(order={self.order}, weight={self.weight}, "
            f"{self._evaluated_frames.size} evaluated frames, {self._unknowns!r})"
        )