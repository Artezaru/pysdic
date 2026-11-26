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

import warnings
import numpy
import scipy.sparse
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
def _split_parameters(
    seq_transformations: Sequence[Transformation], parameters: numpy.ndarray
) -> List[numpy.ndarray]:
    # Split the global parameter vector into the parameters of each transformation
    stops = numpy.cumsum([t.n_parameters for t in seq_transformations])
    return [p.copy() for p in numpy.split(parameters, stops[:-1])]


def _build_operators(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    seq_parameters: Optional[Sequence[Optional[ArrayLike]]] = None,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    filter_nans: bool = False,
    sparse: bool = False,
    _use_cache: bool = False,
    _cache_length: int = 1,
) -> Tuple[Callable, Callable, Optional[Callable], numpy.ndarray]:
    r"""
    Build the operators of the least squares problem to optimize the parameters of several
    transformations combined in chains.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a sequence of :math:`N_T` transformations, and
    :math:`(C_0, C_1, ..., C_{N_C-1})` be a sequence of :math:`N_C` chains of transformations.
    A chain :math:`C_c` is a sequence of indices of transformations, applied in this order:

    .. code-block:: text

        C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

    A transformation can be used in several chains (for example the same intrinsic transformation
    for several views): its parameters are shared by these chains.

    The optimized vector :math:`\vec{p}` is the concatenation of the parameters of **all** the
    transformations, in the order of ``seq_transformations``:

    .. math::

        \vec{p} = \begin{bmatrix} \vec{p}_0 & \vec{p}_1 & \dots & \vec{p}_{N_T - 1} \end{bmatrix}

    For each chain :math:`C_c`, with input points :math:`X_c` and output points :math:`Y_c`, the residuals
    and their Jacobian with respect to the parameters of the chain are:

    .. math::

        \vec{r}_c(\vec{p}) = C_c(X_c; \vec{p}) - Y_c,
        \qquad
        J_c(\vec{p}) = \frac{\partial C_c(X_c; \vec{p})}{\partial \vec{p}}

    The problem solved is :math:`\min_{\vec{p}} \frac{1}{2} \sum_c \|\vec{r}_c(\vec{p})\|^2`.
    Two modes are available to give it to ``pysolvegn``:

    - ``"rJ"``: the residuals of all the chains are stacked in a single vector
      :math:`\vec{r} = (\vec{r}_0, \vec{r}_1, \dots)` with its Jacobian :math:`J` of shape
      :math:`(N_{eq}, N_p)`.
    - ``"gH"``: the gradient :math:`\vec{g} = J^T \vec{r}` and the Gauss-Newton Hessian
      :math:`H = J^T J` are built directly, chain by chain. Only the columns of the transformations
      of the chain are non-zero in :math:`J_c`, so the blocks :math:`J_c^T J_c` and :math:`J_c^T \vec{r}_c`
      are added at the rows and columns of these transformations:

      .. math::

          H = \sum_c J_c^T J_c, \qquad \vec{g} = \sum_c J_c^T \vec{r}_c

      The full Jacobian :math:`J` is never stored, only matrices of size :math:`(N_p, N_p)`.

    Two evaluation strategies are available with ``_use_cache``:

    - ``False``: each function computes only what it returns and nothing is kept in memory.
      The residuals and the cost are computed without the Jacobians with respect to the parameters.
    - ``True``: the first call for given parameters computes all the operators of the mode
      (:math:`\vec{r}` and :math:`J` in mode ``"rJ"``, :math:`\vec{g}`, :math:`H` and the cost in mode ``"gH"``)
      and keeps them in cache, so the other calls for the same parameters return the cached operators.

    .. note::

        The operators work on **copies** of the transformations: the given transformations are
        never modified.

    Parameters
    ----------
    seq_transformations : Sequence[Transformation]
        The transformations :math:`(T_0, T_1, ..., T_{N_T-1})` whose parameters are optimized.
        Each transformation must appear only once, must not be locked and must support
        ``apply`` with ``dp`` (and ``dx`` if it is not the first of a chain).

    seq_chains : Sequence[Sequence[Integral]]
        The chains :math:`(C_0, C_1, ..., C_{N_C-1})`, each chain being a non-empty sequence of
        indices of transformations (without duplicates), in the order of application.

    seq_inputs : Sequence[ArrayLike]
        The input points :math:`X_c` of each chain with shape :math:`(\dots, n_i)`, where :math:`n_i`
        is the input dimension of the first transformation of the chain.

    seq_outputs : Sequence[ArrayLike]
        The output points :math:`Y_c` of each chain with shape :math:`(\dots, n_o)`, where :math:`n_o`
        is the output dimension of the last transformation of the chain.

    seq_parameters : Optional[Sequence[Optional[ArrayLike]]], optional (default: ``None``)
        The initial parameters of each transformation. ``None`` (for the whole sequence or for an
        element) means the current parameters of the transformation.

    seq_kwargs : Optional[Sequence[Optional[Dict[str, Any]]]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of each transformation: one dictionary
        (or None) per transformation of ``seq_transformations``. They are the same for all the chains
        using the transformation.

    mode : str, optional (default: ``"rJ"``)
        The mode of the operators, ``"rJ"`` or ``"gH"``.

    filter_nans : bool, optional (default: ``False``)
        If True, the equations with non-finite residuals or Jacobians (for example points outside
        the domain of a transformation) are set to zero, and a ``RuntimeWarning`` is emitted.

    sparse : bool, optional (default: ``False``)
        If True, the Jacobian (mode ``"rJ"``) or the Hessian (mode ``"gH"``) is returned as a
        ``scipy.sparse.csr_matrix``, which is efficient when there are many transformations with
        few shared parameters.

    _use_cache : bool, optional (default: ``False``)
        If True, all the operators of the mode are computed together and kept in cache (see above).
        If False, each function computes only what it returns, without cache.

    _cache_length : int, optional (default: ``1``)
        The number of previous evaluations kept in cache when ``_use_cache`` is True.


    Returns
    -------
    Callable
        The residual function :math:`\vec{p} \mapsto \vec{r}` (mode ``"rJ"``) or the gradient function
        :math:`\vec{p} \mapsto \vec{g}` (mode ``"gH"``).

    Callable
        The Jacobian function :math:`\vec{p} \mapsto J` (mode ``"rJ"``) or the Hessian function
        :math:`\vec{p} \mapsto H` (mode ``"gH"``).

    Optional[Callable]
        The cost function :math:`\vec{p} \mapsto \frac{1}{2} \|\vec{r}\|^2` in mode ``"gH"``, or None in
        mode ``"rJ"`` (the cost is computed by ``pysolvegn`` from the residuals and the loss).

    numpy.ndarray
        The initial parameters :math:`\vec{p}` with shape :math:`(N_p,)`.

    """
    # ------------- options -------------
    mode = _validate_choice(mode, "mode", ("rJ", "gH"))
    filter_nans = _validate_bool(filter_nans, "filter_nans")
    sparse = _validate_bool(sparse, "sparse")
    _use_cache = _validate_bool(_use_cache, "_use_cache")
    _cache_length = _validate_integer(_cache_length, "_cache_length", min_value=1)

    # ------------- transformations -------------
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
        if t.is_locked:
            raise ValueError(
                f"The parameters of seq_transformations[{k}] ({t.__class__.__name__}) are locked. "
                f"Remove it from the optimized transformations or use a ReparametrizedTransformation."
            )
    if len({id(t) for t in seq_transformations}) != len(seq_transformations):
        raise ValueError(
            "Each transformation must appear only once in seq_transformations. "
            "Use the same index in several chains to share a transformation."
        )
    n_transformations = len(seq_transformations)

    # ------------- initial parameters -------------
    if seq_parameters is None:
        seq_parameters = [None] * n_transformations
    if not isinstance(seq_parameters, Sequence):
        raise TypeError(
            f"Input 'seq_parameters' must be a sequence of arrays (or None). Got type {type(seq_parameters)}."
        )
    if len(seq_parameters) != n_transformations:
        raise ValueError(
            f"Input 'seq_parameters' must contain one element per transformation ({n_transformations}). "
            f"Got {len(seq_parameters)}."
        )
    transformations = [t.copy() for t in seq_transformations]  # Work on copies
    for k, (t, p) in enumerate(zip(transformations, seq_parameters)):
        if p is not None:
            t.set_parameters(p)  # Validated by the transformation (shape and finite values)
        if not t.is_setup:
            raise ValueError(
                f"The parameters of seq_transformations[{k}] ({t.__class__.__name__}) are not set "
                f"and no initial parameters are given in seq_parameters."
            )
    initial_parameters = numpy.concatenate(
        [numpy.asarray(t.parameters, dtype=float).ravel() for t in transformations]
    )

    parameter_starts = numpy.concatenate(
        [[0], numpy.cumsum([t.n_parameters for t in transformations])]
    ).astype(int)
    n_total_parameters = int(parameter_starts[-1])
    if n_total_parameters == 0:
        raise ValueError("The transformations have no parameters to optimize.")

    # ------------- chains -------------
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
        if len(set(chain)) != len(chain):
            raise ValueError(f"Input 'seq_chains[{c}]' must not contain duplicate indices. Got {chain}.")
    n_chains = len(seq_chains)

    compositions = [ComposedTransformation([transformations[k] for k in chain]) for chain in seq_chains]
    for c, composition in enumerate(compositions):
        if not composition.has_apply_dp:
            raise ValueError(
                f"The chain {c} cannot compute the Jacobian with respect to the parameters "
                f"(some transformations do not support dp or dx)."
            )

    # ------------- kwargs -------------
    seq_kwargs = _validate_kwargs_sequence(seq_kwargs, n_transformations, "seq_kwargs")
    chain_kwargs = [[seq_kwargs[k] for k in chain] for chain in seq_chains]  # kwargs of T_k in each chain

    # ------------- points -------------
    for name, seq in (("seq_inputs", seq_inputs), ("seq_outputs", seq_outputs)):
        if not isinstance(seq, Sequence):
            raise TypeError(f"Input '{name}' must be a sequence of arrays. Got type {type(seq)}.")
        if len(seq) != n_chains:
            raise ValueError(
                f"Input '{name}' must contain one element per chain ({n_chains}). Got {len(seq)}."
            )
    inputs, outputs = [], []
    for c, (composition, x, y) in enumerate(zip(compositions, seq_inputs, seq_outputs)):
        # NaN accepted (see filter_nans)
        x = _as_batch_array(
            x, f"seq_inputs[{c}]", composition.input_dim, numpy.float64, finite=False, min_ndim=2
        )
        y = _as_batch_array(
            y, f"seq_outputs[{c}]", composition.output_dim, numpy.float64, finite=False, min_ndim=2
        )
        if x.shape[:-1] != y.shape[:-1] or x.size == 0:
            raise ValueError(
                f"The input and output points of the chain {c} must contain the same non-zero number "
                f"of points, got shapes {x.shape} and {y.shape}."
            )
        inputs.append(x.reshape(-1, composition.input_dim))
        outputs.append(y.reshape(-1, composition.output_dim))

    # ------------- indexing -------------
    chain_columns = [
        numpy.concatenate(
            [numpy.arange(parameter_starts[k], parameter_starts[k + 1]) for k in chain]
        ).astype(int)
        for chain in seq_chains
    ]  # Columns of the global vector for the parameters of each chain
    n_equations = [x.shape[0] * c.output_dim for x, c in zip(inputs, compositions)]
    equation_starts = numpy.concatenate([[0], numpy.cumsum(n_equations)]).astype(int)
    n_total_equations = int(equation_starts[-1])

    # ------------- evaluation -------------
    # Without cache, each function computes only what it returns (the residuals and the cost are computed
    # without the Jacobians, for example to test a Levenberg-Marquardt step). With cache, all the operators
    # of the mode are computed together at the first call for given parameters.
    cached_outputs = ("r", "J") if mode == "rJ" else ("g", "H", "c")
    cache: List[Tuple[numpy.ndarray, Dict[str, Any]]] = []

    def chain_operators(c: int, derivatives: bool) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], int]:
        # Residuals (n_eq_c,) and Jacobian (n_eq_c, n_p_c) (None if not derivatives) of the chain c
        result = compositions[c].apply(inputs[c], dp=derivatives, kwargs=chain_kwargs[c])
        residuals = (result.x - outputs[c]).ravel()
        jacobian = result.jacobian_dp.reshape(n_equations[c], -1) if derivatives else None
        n_bad = 0
        if filter_nans:
            bad_rows = ~numpy.isfinite(residuals)
            if derivatives:
                bad_rows |= ~numpy.all(numpy.isfinite(jacobian), axis=1)
            n_bad = int(numpy.sum(bad_rows))
            if n_bad > 0:
                residuals[bad_rows] = 0.0
                if derivatives:
                    jacobian[bad_rows, :] = 0.0
        return residuals, jacobian, n_bad

    def assemble(parameters: numpy.ndarray, requested: Tuple[str, ...]) -> Dict[str, Any]:
        # Compute only the requested operators among "r", "J", "c", "g" and "H"
        need_r, need_J, need_c = "r" in requested, "J" in requested, "c" in requested
        need_g, need_H = "g" in requested, "H" in requested
        derivatives = need_J or need_g or need_H

        for k, t in enumerate(transformations):
            t.set_parameters(parameters[parameter_starts[k] : parameter_starts[k + 1]])

        residuals = numpy.empty(n_total_equations, dtype=float) if need_r else None
        jacobian = numpy.zeros((n_total_equations, n_total_parameters)) if need_J and not sparse else None
        gradient = numpy.zeros(n_total_parameters, dtype=float) if need_g else None
        hessian = numpy.zeros((n_total_parameters, n_total_parameters)) if need_H and not sparse else None
        cost = 0.0
        J_rows, J_columns, J_values = [], [], []  # Sparse Jacobian
        H_rows, H_columns, H_values = [], [], []  # Sparse Hessian

        n_bad = 0
        for c in range(n_chains):
            r_c, J_c, n_bad_c = chain_operators(c, derivatives)
            n_bad += n_bad_c
            start, stop = equation_starts[c], equation_starts[c + 1]
            columns_c = chain_columns[c]

            if need_r:
                residuals[start:stop] = r_c
            if need_c:
                cost += 0.5 * float(r_c @ r_c)
            if need_J:
                if sparse:
                    local_rows, local_columns = numpy.nonzero(J_c)
                    J_rows.append(local_rows + start)
                    J_columns.append(columns_c[local_columns])
                    J_values.append(J_c[local_rows, local_columns])
                else:
                    jacobian[start:stop, columns_c] = J_c
            if need_g:
                gradient[columns_c] += J_c.T @ r_c
            if need_H:
                H_c = J_c.T @ J_c  # (n_p_c, n_p_c)
                if sparse:
                    H_rows.append(numpy.repeat(columns_c, columns_c.size))
                    H_columns.append(numpy.tile(columns_c, columns_c.size))
                    H_values.append(H_c.ravel())
                else:
                    hessian[numpy.ix_(columns_c, columns_c)] += H_c

        if need_J and sparse:
            jacobian = scipy.sparse.coo_matrix(
                (numpy.concatenate(J_values), (numpy.concatenate(J_rows), numpy.concatenate(J_columns))),
                shape=(n_total_equations, n_total_parameters),
            ).tocsr()
        if need_H and sparse:
            # Duplicated entries (shared transformations) are summed by the conversion
            hessian = scipy.sparse.coo_matrix(
                (numpy.concatenate(H_values), (numpy.concatenate(H_rows), numpy.concatenate(H_columns))),
                shape=(n_total_parameters, n_total_parameters),
            ).tocsr()

        if n_bad == n_total_equations:
            raise ValueError(
                "All the residuals or Jacobians are NaN or infinite, the optimization cannot proceed."
            )
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} out of {n_total_equations} equations were filtered "
                f"because of NaN or infinite residuals or Jacobians.",
                RuntimeWarning,
            )

        operators = {"r": residuals, "J": jacobian, "c": cost, "g": gradient, "H": hessian}
        return {key: operators[key] for key in requested}

    def evaluate(parameters: ArrayLike, output: str) -> Any:
        # Return the operator ``output`` for the given parameters, with or without cache
        parameters = _as_array(
            parameters, "parameters", numpy.float64, shape=(n_total_parameters,), finite=False
        )
        if not _use_cache:
            return assemble(parameters, (output,))[output]

        for cached_parameters, cached_operators in cache:
            if numpy.array_equal(parameters, cached_parameters):
                return cached_operators[output]
        operators = assemble(parameters, cached_outputs)
        cache.insert(0, (parameters.copy(), operators))
        del cache[_cache_length:]
        return operators[output]

    if mode == "rJ":
        return (
            lambda parameters: evaluate(parameters, "r"),
            lambda parameters: evaluate(parameters, "J"),
            None,
            initial_parameters,
        )
    return (
        lambda parameters: evaluate(parameters, "g"),
        lambda parameters: evaluate(parameters, "H"),
        lambda parameters: evaluate(parameters, "c"),
        initial_parameters,
    )


# ======================================================================
# Terms
# ======================================================================
def build_chains_transformation_term(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    seq_parameters: Optional[Sequence[Optional[ArrayLike]]] = None,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    weight: Real = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    filter_nans: bool = False,
    sparse: bool = False,
    use_cache: bool = False,
) -> Tuple[pysolvegn.Term, numpy.ndarray]:
    r"""
    Build a ``pysolvegn.Term`` to optimize the parameters of several transformations combined in chains.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a sequence of transformations and :math:`(C_0, C_1, ..., C_{N_C-1})`
    a sequence of chains of indices of these transformations (a transformation can be shared by several chains).
    The term minimizes the distance between the transformed input points and the output points of all the chains:

    .. math::

        \min_{\vec{p}} \frac{1}{2} \sum_c \sum_j \rho\left( \| C_c(X_{c,j}; \vec{p}) - Y_{c,j} \|^2 \right)

    where :math:`\vec{p}` is the concatenation of the parameters of all the transformations,
    in the order of ``seq_transformations``, and :math:`\rho` is the loss function.

    All the chains are gathered in a **single term** with the same loss and weight. To give different
    losses or weights to some chains, call this function once per group of chains **with the same**
    ``seq_transformations`` (so the parameter vectors are identical) and give the list of terms to ``pysolvegn.solve``.

    The term can be built in two modes:

    - ``"rJ"``: residuals and Jacobian of all the equations. Robust losses (``"soft_l1"``, ``"cauchy"``,
      ``"huber"``, ...) are available, with a soft threshold ``loss_scale`` in the unit of the output points.
    - ``"gH"``: gradient :math:`J^T \vec{r}` and Hessian :math:`J^T J` assembled chain by chain.
      The full Jacobian is never stored, which saves a lot of memory for many points, but only the
      ``"linear"`` loss is available.

    .. note::

        The term works on copies of the transformations: the given transformations are never modified.
        To optimize only some parameters of a transformation, give a :class:`pysdic.ReparametrizedTransformation`
        or use the ``parametrization`` argument of ``pysolvegn.solve``.

    .. seealso::

        - Function :func:`pysdic.build_transformation_term` for a single transformation.
        - Function :func:`pysdic.optimize_chains_transformation_parameters` to directly solve the problem.


    Parameters
    ----------
    seq_transformations : Sequence[Transformation]
        The transformations :math:`(T_0, T_1, ..., T_{N_T-1})` whose parameters are optimized.
        Each transformation must appear only once and must not be locked.

    seq_chains : Sequence[Sequence[Integral]]
        The chains, each chain being a non-empty sequence of indices of ``seq_transformations``
        (without duplicates), in the order of application. For example ``(1, 4, 8)`` is
        :math:`T_8 \circ T_4 \circ T_1`.

    seq_inputs : Sequence[ArrayLike]
        The input points :math:`X_c` of each chain with shape :math:`(\dots, n_i)`.

    seq_outputs : Sequence[ArrayLike]
        The output points :math:`Y_c` of each chain with shape :math:`(\dots, n_o)`.

    seq_parameters : Optional[Sequence[Optional[ArrayLike]]], optional (default: ``None``)
        The initial parameters of each transformation. ``None`` means the current parameters.

    seq_kwargs : Optional[Sequence[Optional[Dict[str, Any]]]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of each transformation: one dictionary
        (or None) per transformation of ``seq_transformations``. They are the same for all the chains
        using the transformation.

    mode : str, optional (default: ``"rJ"``)
        The mode of the term, ``"rJ"`` or ``"gH"``.

    weight : Real, optional (default: ``1.0``)
        The weight of the term.

    loss : Optional[Union[str, Callable]], optional (default: ``None``)
        The loss function of the term (see ``pysolvegn.Term``), only for the mode ``"rJ"``.
        ``None`` means ``"linear"``.

    loss_scale : Real, optional (default: ``1.0``)
        The soft threshold :math:`C` of the loss function in the unit of the output points (for example in
        pixels): the residuals with :math:`|r| \lesssim C` are treated as inliers and the larger ones as outliers.
        Only for the mode ``"rJ"``, no effect on the ``"linear"`` loss.

    filter_nans : bool, optional (default: ``False``)
        If True, the equations with non-finite residuals or Jacobians are set to zero.

    sparse : bool, optional (default: ``False``)
        If True, the Jacobian (``"rJ"``) or the Hessian (``"gH"``) is a ``scipy.sparse.csr_matrix``.

    use_cache : bool, optional (default: ``False``)
        The evaluation strategy of the functions of the term:

        - ``False``: each function computes only what it returns and nothing is kept in memory. The residuals
          (``"rJ"``) and the cost (``"gH"``) are computed without the Jacobian, which is efficient when many
          evaluated points are trial points whose Jacobian is never required (damped steps of Levenberg-Marquardt).
          In mode ``"gH"``, the gradient and the Hessian each compute the Jacobian.
        - ``True``: the first call for given parameters computes all the operators of the mode (residuals and
          Jacobian in ``"rJ"``, gradient, Hessian and cost in ``"gH"``) and keeps them in cache, so the other
          calls for the same parameters are free. It is efficient when the Jacobian is required at every
          evaluated point (Gauss-Newton without damping).

        .. note::

            With ``filter_nans=True``, the residuals and the cost computed without the Jacobian only filter the
            equations with non-finite residuals, so they can differ from those computed with the Jacobian
            (equations with finite residuals but non-finite Jacobians).


    Returns
    -------
    pysolvegn.Term
        The term to give to ``pysolvegn.solve``.

    numpy.ndarray
        The initial parameters :math:`\vec{p}` with shape :math:`(N_p,)`, to use as ``p0``.


    Examples
    --------
    Calibrate one camera observing a target in two positions. The camera parameters
    (``intrinsic``, ``distortion``) are shared by the two chains:

    .. code-block:: python

        seq_transformations = [extrinsic_0, extrinsic_1, distortion, intrinsic]
        seq_chains = [(0, 2, 3), (1, 2, 3)]

        term, p0 = build_chains_transformation_term(
            seq_transformations, seq_chains, [world_0, world_1], [image_0, image_1],
            loss="soft_l1", loss_scale=1.0,  # outliers above about 1 pixel
        )
        result = pysolvegn.solve(term, p0, max_iteration=50, xtol=1e-10, damping="lm-diag")

    """
    weight = _validate_real(weight, "weight", finite=True)
    loss_scale = _validate_real(loss_scale, "loss_scale", positive=True, finite=True)
    use_cache = _validate_bool(use_cache, "use_cache")
    if mode == "gH" and (loss not in (None, "linear") or loss_scale != 1.0):
        raise ValueError(
            f"Only the 'linear' loss (with loss_scale=1.0) is available in mode 'gH', got {loss!r} "
            f"with loss_scale={loss_scale}."
        )

    func_1, func_2, cost_func, initial_parameters = _build_operators(
        seq_transformations,
        seq_chains,
        seq_inputs,
        seq_outputs,
        seq_parameters,
        seq_kwargs=seq_kwargs,
        mode=mode,
        filter_nans=filter_nans,
        sparse=sparse,
        _use_cache=use_cache,
    )

    if mode == "rJ":
        term = pysolvegn.Term.from_rJ(
            residual_func=func_1,
            jacobian_func=func_2,
            weight=weight,
            loss=loss,
            loss_scale=loss_scale,
        )
    else:
        term = pysolvegn.Term.from_gH(
            gradient_func=func_1,
            hessian_func=func_2,
            weight=weight,
            cost_func=cost_func,
        )
    return term, initial_parameters


def build_transformation_term(
    transformation: Transformation,
    input_points: ArrayLike,
    output_points: ArrayLike,
    parameters: Optional[ArrayLike] = None,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "rJ",
    weight: Real = 1.0,
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    filter_nans: bool = False,
    sparse: bool = False,
    use_cache: bool = False,
) -> Tuple[pysolvegn.Term, numpy.ndarray]:
    r"""
    Build a ``pysolvegn.Term`` to optimize the parameters :math:`\vec{p}` of a single transformation :math:`T`
    such that :math:`T(X; \vec{p}) \approx Y`:

    .. math::

        \min_{\vec{p}} \frac{1}{2} \sum_j \rho\left( \| T(X_j; \vec{p}) - Y_j \|^2 \right)

    .. seealso::

        - Function :func:`pysdic.build_chains_transformation_term` for the description of the modes and arguments.


    Parameters
    ----------
    transformation : Transformation
        The transformation to optimize. It must not be locked.

    input_points : ArrayLike
        The input points :math:`X` with shape :math:`(\dots, n_i)`.

    output_points : ArrayLike
        The output points :math:`Y` with shape :math:`(\dots, n_o)`.

    parameters : Optional[ArrayLike], optional (default: ``None``)
        The initial parameters. ``None`` means the current parameters of the transformation.

    kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the transformation.

    mode, weight, loss, loss_scale, filter_nans, sparse, use_cache
        See :func:`pysdic.build_chains_transformation_term`.


    Returns
    -------
    pysolvegn.Term
        The term to give to ``pysolvegn.solve``.

    numpy.ndarray
        The initial parameters with shape :math:`(n_p,)`, to use as ``p0``.

    """
    return build_chains_transformation_term(
        [transformation],
        [[0]],
        [input_points],
        [output_points],
        [parameters],
        seq_kwargs=[kwargs],
        mode=mode,
        weight=weight,
        loss=loss,
        loss_scale=loss_scale,
        filter_nans=filter_nans,
        sparse=sparse,
        use_cache=use_cache,
    )


# ======================================================================
# Optimization
# ======================================================================
def optimize_chains_transformation_parameters(
    seq_transformations: Sequence[Transformation],
    seq_chains: Sequence[Sequence[Integral]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    seq_parameters: Optional[Sequence[Optional[ArrayLike]]] = None,
    *,
    seq_kwargs: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    mode: str = "rJ",
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    filter_nans: bool = False,
    sparse: bool = False,
    use_cache: bool = False,
    **solve_kwargs,
) -> Tuple[List[numpy.ndarray], pysolvegn.SolveResult]:
    r"""
    Optimize the parameters of several transformations combined in chains with ``pysolvegn.solve``.

    The problem is built with :func:`pysdic.build_chains_transformation_term` and the optimized
    parameters are split per transformation:

    .. code-block:: python

        (p_0, p_1, p_2), result = optimize_chains_transformation_parameters([T_0, T_1, T_2], ...)

    The ``pysolvegn.SolveResult`` gives the convergence information (``success``, ``message``, ``cost``,
    ``n_iterations``, ``history``, ...).

    .. note::

        The given transformations are not modified. Set the optimized parameters yourself,
        for example with ``T_0.parameters = p_0``.

    .. seealso::

        - Function :func:`pysdic.build_chains_transformation_term` for the description of the arguments.
        - Function :func:`pysdic.optimize_transformation_parameters` for a single transformation.


    Parameters
    ----------
    seq_transformations, seq_chains, seq_inputs, seq_outputs, seq_parameters, seq_kwargs, mode, loss, loss_scale, filter_nans, sparse, use_cache
        See :func:`pysdic.build_chains_transformation_term`.

    **solve_kwargs
        Additional keyword arguments passed to ``pysolvegn.solve`` (``max_iteration``, ``xtol``,
        ``ftol``, ``damping``, ``verbosity``, ``history``, ...). The ``parametrization`` argument is not
        supported here: build the term with :func:`pysdic.build_chains_transformation_term` and call
        ``pysolvegn.solve``.


    Returns
    -------
    List[numpy.ndarray]
        The optimized parameters of each transformation, in the order of ``seq_transformations``.

    pysolvegn.SolveResult
        The result of ``pysolvegn.solve``.

    """
    if "parametrization" in solve_kwargs:
        raise TypeError(
            "The 'parametrization' argument is not supported by this function. Use "
            "build_chains_transformation_term and pysolvegn.solve, or a ReparametrizedTransformation."
        )

    term, initial_parameters = build_chains_transformation_term(
        seq_transformations,
        seq_chains,
        seq_inputs,
        seq_outputs,
        seq_parameters,
        seq_kwargs=seq_kwargs,
        mode=mode,
        loss=loss,
        loss_scale=loss_scale,
        filter_nans=filter_nans,
        sparse=sparse,
        use_cache=use_cache,
    )
    result = pysolvegn.solve(term, initial_parameters, **solve_kwargs)
    return _split_parameters(seq_transformations, result.parameters), result


def optimize_transformation_parameters(
    transformation: Transformation,
    input_points: ArrayLike,
    output_points: ArrayLike,
    parameters: Optional[ArrayLike] = None,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    mode: str = "rJ",
    loss: Optional[Union[str, Callable]] = None,
    loss_scale: Real = 1.0,
    filter_nans: bool = False,
    sparse: bool = False,
    use_cache: bool = False,
    **solve_kwargs,
) -> Tuple[numpy.ndarray, pysolvegn.SolveResult]:
    r"""
    Optimize the parameters :math:`\vec{p}` of a single transformation :math:`T` such that
    :math:`T(X; \vec{p}) \approx Y` with ``pysolvegn.solve``:

    .. math::

        \min_{\vec{p}} \frac{1}{2} \sum_j \rho\left( \| T(X_j; \vec{p}) - Y_j \|^2 \right)

    .. note::

        The given transformation is not modified. Set the optimized parameters yourself,
        for example with ``transformation.parameters = p``.

    .. seealso::

        - Function :func:`pysdic.build_transformation_term` to build only the term.
        - Function :func:`pysdic.optimize_chains_transformation_parameters` for several transformations and chains.


    Parameters
    ----------
    transformation, input_points, output_points, parameters, kwargs
        See :func:`pysdic.build_transformation_term`.

    mode, loss, loss_scale, filter_nans, sparse, use_cache
        See :func:`pysdic.build_chains_transformation_term`.

    **solve_kwargs
        Additional keyword arguments passed to ``pysolvegn.solve``.


    Returns
    -------
    numpy.ndarray
        The optimized parameters with shape :math:`(n_p,)`.

    pysolvegn.SolveResult
        The result of ``pysolvegn.solve``.


    Examples
    --------
    .. code-block:: python

        p, result = optimize_transformation_parameters(homography, points_1, points_2, max_iteration=20, xtol=1e-10)
        if result.success:
            homography.parameters = p

    """
    seq_parameters, result = optimize_chains_transformation_parameters(
        [transformation],
        [[0]],
        [input_points],
        [output_points],
        [parameters],
        seq_kwargs=[kwargs],
        mode=mode,
        loss=loss,
        loss_scale=loss_scale,
        filter_nans=filter_nans,
        sparse=sparse,
        use_cache=use_cache,
        **solve_kwargs,
    )
    return seq_parameters[0], result