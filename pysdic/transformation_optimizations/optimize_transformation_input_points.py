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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import pysolvegn

from ..transformations.transformation import ComposedTransformation, Transformation
from .._optionals import import_optional
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _validate_choice,
    _validate_kwargs_sequence,
    _as_array,
    _as_batch_array,
    _as_index_array,
)


# ======================================================================
# Internal helpers
# ======================================================================
def _build_batch_operators(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_outputs: Sequence[ArrayLike],
    guess: ArrayLike,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    _use_cache: bool = False,
    _cache_length: int = 1,
) -> Tuple[Callable, Callable, Optional[Callable], numpy.ndarray, Tuple[int, ...]]:
    r"""
    Build the batched operators of the least squares problems to optimize the input points of several
    chains of transformations.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a sequence of :math:`N_T` transformations, and
    :math:`(C_0, C_1, ..., C_{N_C-1})` be a sequence of :math:`N_C` chains of transformations.
    A chain :math:`C_c` is a sequence of indices of transformations, applied in this order:

    .. code-block:: text

        C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

    All the chains share the same input points :math:`X` and each chain :math:`C_c` has its own output
    points :math:`Y_c`. The parameters of the transformations are fixed, so each point :math:`k` is an
    independent problem with the residuals and Jacobian:

    .. math::

        \vec{r}_k(\vec{X}_k) = \begin{bmatrix} C_0(\vec{X}_k) - \vec{Y}_{0,k} \\ \vdots \\ C_{N_C-1}(\vec{X}_k) - \vec{Y}_{N_C-1,k} \end{bmatrix},
        \qquad
        J_k = \frac{\partial \vec{r}_k}{\partial \vec{X}_k}

    The operators follow the convention of ``pysolvegn.BatchTerm``: they take the input points of the
    :math:`m` problems in processing with shape :math:`(m, n_i)` and their indices with shape :math:`(m,)`.
    Two modes are available:

    - ``"rJ"``: residuals :math:`(m, N_{eq})` and Jacobians :math:`(m, N_{eq}, n_i)`.
    - ``"gH"``: gradients :math:`J_k^T \vec{r}_k` :math:`(m, n_i)`, Hessians :math:`J_k^T J_k`
      :math:`(m, n_i, n_i)` and costs :math:`\frac{1}{2} \|\vec{r}_k\|^2` :math:`(m,)`.

    Two evaluation strategies are available with ``_use_cache``:

    - ``False``: each function computes only what it returns and nothing is kept in memory.
      The residuals and the costs are computed without the Jacobians.
    - ``True``: the first call for given points computes all the operators of the mode
      (residuals and Jacobians in mode ``"rJ"``, gradients, Hessians and costs in mode ``"gH"``)
      and keeps them in cache, so the other calls for the same points return the cached operators.

    .. note::

        The operators work on **copies** of the transformations: the given transformations are
        never modified.

    Parameters
    ----------
    seq_transformations : Sequence[Transformation]
        The transformations :math:`(T_0, T_1, ..., T_{N_T-1})`. Their parameters must be set and each
        chain must support ``apply`` with ``dx``.

    seq_chains : Sequence[Sequence[Integral]]
        The chains, each chain being a non-empty sequence of indices of ``seq_transformations``,
        in the order of application.

    seq_outputs : Sequence[ArrayLike]
        The output points :math:`Y_c` of each chain with shape :math:`(\dots, n_{o,c})`,
        with the same batch shape as ``guess``.

    guess : ArrayLike
        The initial guess of the input points with shape :math:`(\dots, n_i)`.

    seq_kwargs : Optional[Sequence[Optional[Dict[str, Any]]]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of each transformation: one
        dictionary (or None) per transformation of ``seq_transformations``.

    mode : str, optional (default: ``"rJ"``)
        The mode of the operators, ``"rJ"`` or ``"gH"``.

    _use_cache : bool, optional (default: ``False``)
        If True, all the operators of the mode are computed together and kept in cache (see above).
        If False, each function computes only what it returns, without cache.

    _cache_length : int, optional (default: ``1``)
        The number of previous evaluations kept in cache when ``_use_cache`` is True.


    Returns
    -------
    Callable
        The residual function (mode ``"rJ"``) or the gradient function (mode ``"gH"``).

    Callable
        The Jacobian function (mode ``"rJ"``) or the Hessian function (mode ``"gH"``).

    Optional[Callable]
        The cost function in mode ``"gH"``, or None in mode ``"rJ"``.

    numpy.ndarray
        The initial input points with shape :math:`(k, n_i)` where :math:`k` is the number of points.

    Tuple[int, ...]
        The batch shape :math:`(\dots)` of ``guess``.

    """
    # ------------- options -------------
    mode = _validate_choice(mode, "mode", ("rJ", "gH"))
    _use_cache = _validate_bool(_use_cache, "_use_cache")
    _cache_length = _validate_integer(_cache_length, "_cache_length", min_value=1)

    # ------------- transformations and chains -------------
    if not isinstance(seq_transformations, Sequence):
        raise TypeError(
            f"Input 'seq_transformations' must be a sequence of Transformation. "
            f"Got type {type(seq_transformations)}."
        )
    if len(seq_transformations) == 0:
        raise ValueError("Input 'seq_transformations' must contain at least one Transformation.")
    for k, t in enumerate(seq_transformations):
        if not isinstance(t, Transformation):
            raise TypeError(
                f"Input 'seq_transformations[{k}]' must be a Transformation. Got type {type(t)}."
            )
        if not t.is_setup:
            raise ValueError(f"The parameters of seq_transformations[{k}] ({t.__class__.__name__}) are not set.")
    n_transformations = len(seq_transformations)
    transformations = [t.copy() for t in seq_transformations]  # Work on copies

    if not isinstance(seq_chains, Sequence):
        raise TypeError(f"Input 'seq_chains' must be a sequence of chains. Got type {type(seq_chains)}.")
    if len(seq_chains) == 0:
        raise ValueError("Input 'seq_chains' must contain at least one chain.")
    # Indices in [0, N_T) (integers only, no negative index counting from the end)
    seq_chains = [
        _as_index_array(chain, n_transformations, f"seq_chains[{c}]").tolist()
        for c, chain in enumerate(seq_chains)
    ]
    for c, chain in enumerate(seq_chains):
        if len(chain) == 0:
            raise ValueError(f"Input 'seq_chains[{c}]' must not be empty.")
    n_chains = len(seq_chains)

    compositions = [ComposedTransformation([transformations[k] for k in chain]) for chain in seq_chains]
    for c, composition in enumerate(compositions):
        if not composition.has_apply_dx:
            raise ValueError(
                f"The chain {c} cannot compute the Jacobian with respect to the input points "
                f"(some transformations do not support dx)."
            )

    seq_kwargs = _validate_kwargs_sequence(seq_kwargs, n_transformations, "seq_kwargs")
    chain_kwargs = [[seq_kwargs[k] for k in chain] for chain in seq_chains]  # kwargs of T_k in each chain

    # ------------- points -------------
    # All the chains share the input points X: same input dimension
    input_dim = compositions[0].input_dim
    for c, composition in enumerate(compositions):
        if composition.input_dim != input_dim:
            raise ValueError(
                f"All the chains must have the same input dimension. Got {input_dim} for the chain 0 "
                f"and {composition.input_dim} for the chain {c}."
            )
    # NaN accepted in the points (e.g. points outside the domain of a transformation)
    guess = _as_batch_array(guess, "guess", input_dim, numpy.float64, finite=False, min_ndim=1)
    if guess.size == 0:
        raise ValueError(f"Input 'guess' must contain at least one point. Got shape {guess.shape}.")
    batch_shape = guess.shape[:-1]

    if not isinstance(seq_outputs, Sequence):
        raise TypeError(f"Input 'seq_outputs' must be a sequence of arrays. Got type {type(seq_outputs)}.")
    if len(seq_outputs) != n_chains:
        raise ValueError(
            f"Input 'seq_outputs' must contain one element per chain ({n_chains}). Got {len(seq_outputs)}."
        )
    outputs = []
    for c, (composition, y) in enumerate(zip(compositions, seq_outputs)):
        y = _as_array(
            y, f"seq_outputs[{c}]", numpy.float64, shape=(*batch_shape, composition.output_dim), finite=False
        )
        outputs.append(y.reshape(-1, composition.output_dim))

    output_starts = numpy.concatenate([[0], numpy.cumsum([y.shape[1] for y in outputs])]).astype(int)
    n_equations = int(output_starts[-1])
    if input_dim > n_equations:
        raise ValueError(
            f"The problem is under-determined: the input dimension ({input_dim}) is greater than the total "
            f"output dimension of the chains ({n_equations})."
        )

    # ------------- evaluation -------------
    # Without cache, each function computes only what it returns (the residuals and the cost are computed
    # without the Jacobians, for example to test a Levenberg-Marquardt step). With cache, all the operators
    # of the mode are computed together at the first call for given points.
    cached_outputs = ("r", "J") if mode == "rJ" else ("g", "H", "c")
    cache: List[Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]] = []

    def assemble(points: numpy.ndarray, indices: numpy.ndarray, requested: Tuple[str, ...]) -> Dict[str, Any]:
        # Compute only the requested operators among "r", "J", "c", "g" and "H"
        need_r, need_J, need_c = "r" in requested, "J" in requested, "c" in requested
        need_g, need_H = "g" in requested, "H" in requested
        need_residuals = need_r or need_c or need_g
        derivatives = need_J or need_g or need_H

        m = points.shape[0]
        residuals = numpy.empty((m, n_equations), dtype=float) if need_residuals else None
        jacobian = numpy.empty((m, n_equations, input_dim), dtype=float) if derivatives else None
        for c, composition in enumerate(compositions):
            result = composition.apply(points, dx=derivatives, kwargs=chain_kwargs[c])
            start, stop = output_starts[c], output_starts[c + 1]
            if need_residuals:
                residuals[:, start:stop] = result.x - outputs[c][indices]
            if derivatives:
                jacobian[:, start:stop, :] = result.jacobian_dx

        operators: Dict[str, Any] = {}
        if need_r:
            operators["r"] = residuals  # (m, n_eq)
        if need_J:
            operators["J"] = jacobian  # (m, n_eq, n_i)
        if need_c:
            operators["c"] = 0.5 * numpy.sum(residuals**2, axis=1)  # (m,)
        if need_g or need_H:
            jacobian_t = jacobian.transpose(0, 2, 1)
            if need_g:
                operators["g"] = (jacobian_t @ residuals[..., None])[..., 0]  # (m, n_i)
            if need_H:
                operators["H"] = jacobian_t @ jacobian  # (m, n_i, n_i)
        return operators

    def evaluate(points: ArrayLike, indices: ArrayLike, output: str) -> numpy.ndarray:
        # Return the operator ``output`` for the given points, with or without cache
        points = _as_array(points, "points", numpy.float64, shape=(None, input_dim), finite=False)
        indices = _as_array(indices, "indices", numpy.intp, shape=(points.shape[0],), finite=False)
        if not _use_cache:
            return assemble(points, indices, (output,))[output]

        for cached_points, cached_indices, cached_operators in cache:
            if numpy.array_equal(indices, cached_indices) and numpy.array_equal(points, cached_points):
                return cached_operators[output]
        operators = assemble(points, indices, cached_outputs)
        cache.insert(0, (points.copy(), indices.copy(), operators))
        del cache[_cache_length:]
        return operators[output]

    initial_points = guess.reshape(-1, input_dim).copy()

    if mode == "rJ":
        return (
            lambda points, indices: evaluate(points, indices, "r"),
            lambda points, indices: evaluate(points, indices, "J"),
            None,
            initial_points,
            batch_shape,
        )
    return (
        lambda points, indices: evaluate(points, indices, "g"),
        lambda points, indices: evaluate(points, indices, "H"),
        lambda points, indices: evaluate(points, indices, "c"),
        initial_points,
        batch_shape,
    )


def _build_batch_term(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_outputs: Sequence[ArrayLike],
    guess: ArrayLike,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]],
    mode: str,
    weight: Union[Real, ArrayLike],
    loss: Optional[Union[str, Callable]],
    loss_scale: Real,
    use_cache: bool,
) -> Tuple[pysolvegn.BatchTerm, numpy.ndarray, Tuple[int, ...]]:
    # Build the BatchTerm, the initial points (k, n_i) and the batch shape of the points
    loss_scale = _validate_real(loss_scale, "loss_scale", positive=True, finite=True)
    use_cache = _validate_bool(use_cache, "use_cache")
    if mode == "gH" and (loss not in (None, "linear") or loss_scale != 1.0):
        raise ValueError(
            f"Only the 'linear' loss (with loss_scale=1.0) is available in mode 'gH', got {loss!r} "
            f"with loss_scale={loss_scale}."
        )
    func_1, func_2, cost_func, initial_points, batch_shape = _build_batch_operators(
        seq_transformations, seq_chains, seq_outputs, guess, seq_kwargs=seq_kwargs, mode=mode, _use_cache=use_cache
    )

    if isinstance(weight, Real):  # One weight for all the points (booleans are rejected)
        weight = _validate_real(weight, "weight", finite=True)
    else:  # One weight per point with the batch shape of the points
        weight = _as_array(weight, "weight", numpy.float64, shape=batch_shape, finite=True).ravel()

    if mode == "rJ":
        term = pysolvegn.BatchTerm.from_rJ(
            residual_func=func_1,
            jacobian_func=func_2,
            weight=weight,
            loss=loss,
            loss_scale=loss_scale,
        )
    else:
        term = pysolvegn.BatchTerm.from_gH(
            gradient_func=func_1,
            hessian_func=func_2,
            cost_func=cost_func,
            weight=weight,
        )
    return term, initial_points, batch_shape


# ======================================================================
# Terms
# ======================================================================
def build_chains_transformation_input_points_term(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_outputs: Sequence[ArrayLike],
    guess: ArrayLike,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    weight: Union[Real, ArrayLike] = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    use_cache: bool = False,
) -> Tuple[pysolvegn.BatchTerm, numpy.ndarray]:
    r"""
    Build a ``pysolvegn.BatchTerm`` to optimize the input points :math:`X` such that their images by several
    chains of transformations match the given output points (for example triangulation of points seen by
    several cameras).

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a sequence of transformations, and :math:`(C_0, C_1, ..., C_{N_C-1})`
    a sequence of chains of indices of these transformations, applied in this order:

    .. code-block:: text

        C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

    All the chains share the same input points :math:`X` and each chain :math:`C_c` has its own output
    points :math:`Y_c`. The parameters of the transformations are fixed, so each point :math:`k` is an
    independent problem of the batch:

    .. math::

        \min_{\vec{X}_k} \frac{1}{2} \sum_c \sum_j \rho\left( \left( C_c(\vec{X}_k) - \vec{Y}_{c,k} \right)_j^2 \right)

    The problems are solved together with ``pysolvegn.solve_batch``: each point is removed from the
    processing as soon as it satisfies a stopping criterion.

    The term can be built in two modes:

    - ``"rJ"``: residuals and Jacobians of each point. Robust losses (``"soft_l1"``, ``"cauchy"``,
      ``"huber"``, ...) are available, with a soft threshold ``loss_scale`` in the unit of the output points.
    - ``"gH"``: gradients :math:`J_k^T \vec{r}_k` and Hessians :math:`J_k^T J_k` of each point, with its cost
      (only the ``"linear"`` loss).

    .. note::

        The term works on copies of the transformations: the given transformations are never modified.

    .. warning::

        The dimension :math:`n_i` of the input points must be lower than or equal to the total dimension
        of the outputs :math:`\sum_c n_{o,c}`, otherwise the problems are under-determined.

    .. seealso::

        - Function :func:`pysdic.build_transformation_input_points_term` for a single transformation.
        - Function :func:`pysdic.optimize_chains_transformation_input_points` to directly solve the problems.
        - Function :func:`pysdic.build_chains_transformation_term` to optimize the parameters instead.


    Parameters
    ----------
    seq_transformations : Sequence[Transformation]
        The transformations :math:`(T_0, T_1, ..., T_{N_T-1})`. Their parameters must be set.

    seq_chains : Sequence[Sequence[Integral]]
        The chains, each chain being a non-empty sequence of indices of ``seq_transformations``,
        in the order of application. The first transformation of each chain must have the
        dimension :math:`n_i` of the input points.

    seq_outputs : Sequence[ArrayLike]
        The output points :math:`Y_c` of each chain with shape :math:`(\dots, n_{o,c})`,
        with the same batch shape as ``guess``.

    guess : ArrayLike
        The initial guess of the input points with shape :math:`(\dots, n_i)`.

    seq_kwargs : Optional[Sequence[Optional[Dict[str, Any]]]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of each transformation: one dictionary
        (or None) per transformation of ``seq_transformations``.

        .. warning::

            The keyword arguments must not depend on the points, since the solver evaluates only
            the points still in processing.

    mode : str, optional (default: ``"rJ"``)
        The mode of the term, ``"rJ"`` or ``"gH"``.

    weight : Union[Real, ArrayLike], optional (default: ``1.0``)
        The weight of the term: a scalar for all the points, or one weight per point with the batch
        shape :math:`(\dots)` of ``guess``.

    loss : Optional[Union[str, Callable]], optional (default: ``None``)
        The loss function of the term (see ``pysolvegn.BatchTerm``), only for the mode ``"rJ"``.
        ``None`` means ``"linear"``.

    loss_scale : Real, optional (default: ``1.0``)
        The soft threshold :math:`C` of the loss function in the unit of the output points (for example in
        pixels). Only for the mode ``"rJ"``, no effect on the ``"linear"`` loss.

    use_cache : bool, optional (default: ``False``)
        The evaluation strategy of the functions of the term:

        - ``False``: each function computes only what it returns and nothing is kept in memory. The residuals
          (``"rJ"``) and the costs (``"gH"``) are computed without the Jacobians, which is efficient when many
          evaluated points are trial points whose Jacobians are never required (damped steps of
          Levenberg-Marquardt). In mode ``"gH"``, the gradients and the Hessians each compute the Jacobians.
        - ``True``: the first call for given points computes all the operators of the mode (residuals and
          Jacobians in ``"rJ"``, gradients, Hessians and costs in ``"gH"``) and keeps them in cache, so the
          other calls for the same points are free. It is efficient when the Jacobians are required at every
          evaluated point (Gauss-Newton without damping).


    Returns
    -------
    pysolvegn.BatchTerm
        The term to give to ``pysolvegn.solve_batch``.

    numpy.ndarray
        The initial input points with shape :math:`(k, n_i)`, to use as ``p0``, where :math:`k` is the
        number of points (the batch dimensions of ``guess`` are flattened).


    Examples
    --------
    Triangulate 3D points seen by two calibrated cameras:

    .. code-block:: python

        seq_transformations = [extrinsic_0, extrinsic_1, distortion, intrinsic]
        seq_chains = [(0, 2, 3), (1, 2, 3)]

        term, p0 = build_chains_transformation_input_points_term(
            seq_transformations, seq_chains, [image_points_0, image_points_1], guess
        )
        result = pysolvegn.solve_batch(term, p0, max_iteration=20, xtol=1e-10)
        world_points = result.parameters.reshape(guess.shape)

    """
    term, initial_points, _ = _build_batch_term(
        seq_transformations,
        seq_chains,
        seq_outputs,
        guess,
        seq_kwargs=seq_kwargs,
        mode=mode,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
        use_cache=use_cache,
    )
    return term, initial_points


def build_transformation_input_points_term(
    transformation: Transformation,
    output_points: ArrayLike,
    guess: ArrayLike,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "rJ",
    weight: Union[Real, ArrayLike] = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    use_cache: bool = False,
) -> Tuple[pysolvegn.BatchTerm, numpy.ndarray]:
    r"""
    Build a ``pysolvegn.BatchTerm`` to optimize the input points :math:`X` of a transformation :math:`T`
    such that :math:`T(X) \approx Y`. Each point :math:`k` is an independent problem of the batch:

    .. math::

        \min_{\vec{X}_k} \frac{1}{2} \sum_j \rho\left( \left( T(\vec{X}_k) - \vec{Y}_k \right)_j^2 \right)

    It is a numerical inverse of the transformation, for example when ``invert`` is not implemented
    or to refine its result.

    .. warning::

        The input dimension :math:`n_i` must be lower than or equal to the output dimension :math:`n_o`.

    .. seealso::

        - Function :func:`pysdic.build_chains_transformation_input_points_term` for the description of the
          modes and arguments.


    Parameters
    ----------
    transformation : Transformation
        The transformation. Its parameters must be set.

    output_points : ArrayLike
        The output points :math:`Y` with shape :math:`(\dots, n_o)`.

    guess : ArrayLike
        The initial guess of the input points with shape :math:`(\dots, n_i)`.

    kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the transformation
        (they must not depend on the points).

    mode, weight, loss, loss_scale, use_cache
        See :func:`pysdic.build_chains_transformation_input_points_term`.


    Returns
    -------
    pysolvegn.BatchTerm
        The term to give to ``pysolvegn.solve_batch``.

    numpy.ndarray
        The initial input points with shape :math:`(k, n_i)`, to use as ``p0``.

    """
    return build_chains_transformation_input_points_term(
        [transformation],
        [(0,)],
        [output_points],
        guess,
        seq_kwargs=[kwargs],
        mode=mode,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
        use_cache=use_cache,
    )


# ======================================================================
# Optimization
# ======================================================================
def optimize_chains_transformation_input_points(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_outputs: Sequence[ArrayLike],
    guess: ArrayLike,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    use_cache: bool = False,
    **solve_kwargs,
) -> Tuple[numpy.ndarray, pysolvegn.BatchSolveResult]:
    r"""
    Optimize the input points :math:`X` such that their images by several chains of transformations
    match the given output points, with ``pysolvegn.solve_batch``.

    The problems are built with :func:`pysdic.build_chains_transformation_input_points_term` (one independent
    problem per point) and the optimized points are returned with the batch shape of ``guess``:

    .. code-block:: python

        points, result = optimize_chains_transformation_input_points(seq_transformations, seq_chains, seq_outputs, guess, ...)

    The ``pysolvegn.BatchSolveResult`` gives the convergence information of each point (``success``,
    ``reasons``, ``cost``, ``n_iterations``, ...), with the points flattened along the first dimension:
    use ``result.success.reshape(points.shape[:-1])`` to recover the batch shape.

    .. seealso::

        - Function :func:`pysdic.build_chains_transformation_input_points_term` for the description of the arguments.
        - Function :func:`pysdic.optimize_transformation_input_points` for a single transformation.
        - Function :func:`pysdic.optimize_chains_transformation_parameters` to optimize the parameters instead.


    Parameters
    ----------
    seq_transformations, seq_chains, seq_outputs, guess, seq_kwargs, mode, loss, loss_scale, use_cache
        See :func:`pysdic.build_chains_transformation_input_points_term`.

    **solve_kwargs
        Additional keyword arguments passed to ``pysolvegn.solve_batch`` (``max_iteration``, ``xtol``,
        ``ptol``, ``damping``, ``verbosity``, ``history``, ...). The ``parametrization`` argument is not
        supported here: build the term with :func:`pysdic.build_chains_transformation_input_points_term`
        and call ``pysolvegn.solve_batch``.


    Returns
    -------
    numpy.ndarray
        The optimized input points with shape :math:`(\dots, n_i)`.

    pysolvegn.BatchSolveResult
        The result of ``pysolvegn.solve_batch``.


    Examples
    --------
    Triangulate 3D points seen by two calibrated cameras:

    .. code-block:: python

        seq_transformations = [extrinsic_0, distortion_0, intrinsic_0, extrinsic_1, distortion_1, intrinsic_1]
        seq_chains = [(0, 1, 2), (3, 4, 5)]

        world_points, result = optimize_chains_transformation_input_points(
            seq_transformations, seq_chains, [image_points_0, image_points_1], guess, max_iteration=20, xtol=1e-10
        )

    """
    if "parametrization" in solve_kwargs:
        raise TypeError(
            "The 'parametrization' argument is not supported by this function. Use "
            "build_chains_transformation_input_points_term and pysolvegn.solve_batch."
        )

    term, initial_points, batch_shape = _build_batch_term(
        seq_transformations,
        seq_chains,
        seq_outputs,
        guess,
        seq_kwargs=seq_kwargs,
        mode=mode,
        weight=1.0,
        loss=loss,
        loss_scale=loss_scale,
        use_cache=use_cache,
    )
    result = pysolvegn.solve_batch(term, initial_points, **solve_kwargs)
    return result.parameters.reshape(*batch_shape, initial_points.shape[1]), result


def optimize_transformation_input_points(
    transformation: Transformation,
    output_points: ArrayLike,
    guess: ArrayLike,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "rJ",
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    use_cache: bool = False,
    **solve_kwargs,
) -> Tuple[numpy.ndarray, pysolvegn.BatchSolveResult]:
    r"""
    Optimize the input points :math:`X` of a transformation :math:`T` such that :math:`T(X) \approx Y`,
    with ``pysolvegn.solve_batch`` (one independent problem per point).

    It is a numerical inverse of the transformation, for example when ``invert`` is not implemented
    or to refine its result.

    .. seealso::

        - Function :func:`pysdic.build_transformation_input_points_term` to build only the term.
        - Function :func:`pysdic.optimize_chains_transformation_input_points` for several chains.


    Parameters
    ----------
    transformation, output_points, guess, kwargs
        See :func:`pysdic.build_transformation_input_points_term`.

    mode, loss, loss_scale, use_cache
        See :func:`pysdic.build_chains_transformation_input_points_term`.

    **solve_kwargs
        Additional keyword arguments passed to ``pysolvegn.solve_batch``.


    Returns
    -------
    numpy.ndarray
        The optimized input points with shape :math:`(\dots, n_i)`.

    pysolvegn.BatchSolveResult
        The result of ``pysolvegn.solve_batch``.


    Examples
    --------
    .. code-block:: python

        normalized_points, result = optimize_transformation_input_points(
            distortion, distorted_points, guess=distorted_points, max_iteration=20, ptol=1e-12
        )

    """
    return optimize_chains_transformation_input_points(
        [transformation],
        [(0,)],
        [output_points],
        guess,
        seq_kwargs=[kwargs],
        mode=mode,
        loss=loss,
        loss_scale=loss_scale,
        use_cache=use_cache,
        **solve_kwargs,
    )