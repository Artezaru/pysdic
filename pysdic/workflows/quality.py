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

# Quality control of a measure: RMS of the residuals per element and per pairing, detection of
# the badly correlated elements, and uncertainty of the unknowns from the Hessian and the noise.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence, Tuple

import numpy
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike

from ..geometry.integration_points import IntegrationPoints
from .._validation import _as_array, _validate_integer, _validate_real
from ..sdic.pairing import Pairing
from ..sdic.residuals import PairingResiduals
from ..sdic.sdic_analysis import SDICAnalysis
from ..sdic.sdic_term import SDICCorrelationTerm


# =============================================================================
# RMS of the residuals per element
# =============================================================================
@dataclass(frozen=True)
class ElementsResidualRMSResult:
    r"""
    The RMS of the residuals per element and per pairing, returned by
    :func:`compute_elements_residual_rms`.

    Attributes
    ----------
    pairings : Tuple[:class:`pysdic.Pairing`, ...]
        The pairings, in the order of the rows.

    rms : numpy.ndarray
        The RMS of the residuals of each element in each pairing, shape
        :math:`(n_{pairings}, N_e)` (NaN for an element without finite residual).

    n_points : numpy.ndarray
        The number of points with finite residuals of each element in each pairing, shape
        :math:`(n_{pairings}, N_e)`.

    pairing_rms : numpy.ndarray
        The RMS of all the finite residuals of each pairing, shape :math:`(n_{pairings},)`.

    outliers : numpy.ndarray
        The badly correlated elements of each pairing, boolean of shape
        :math:`(n_{pairings}, N_e)`.

    """

    pairings: Tuple[Pairing, ...]
    rms: numpy.ndarray
    n_points: numpy.ndarray
    pairing_rms: numpy.ndarray
    outliers: numpy.ndarray

    @property
    def bad_elements(self) -> numpy.ndarray:
        r"""The indices of the elements that are outliers in at least one pairing."""
        return numpy.flatnonzero(self.outliers.any(axis=0))


def compute_elements_residual_rms(
    analysis: SDICAnalysis,
    integration_points: IntegrationPoints,
    residuals: Sequence[PairingResiduals],
    *,
    threshold: Real = 3.0,
) -> ElementsResidualRMSResult:
    r"""
    Compute the RMS of the residuals of each element in each pairing, and detect the badly
    correlated elements (reflections, occlusions, damaged speckle, ...).

    For an element :math:`e` and a pairing, with the points :math:`p \in e` whose residuals
    are finite in all the channels, of weights :math:`w_p`:

    .. math::

        \text{RMS}_e = \sqrt{\frac{\sum_{p \in e} w_p \sum_c r_{p,c}^2}{C \sum_{p \in e} w_p}}

    An element is an outlier of a pairing if its RMS is far above the RMS of the other
    elements of the pairing, with a robust estimate of their spread (median and median
    absolute deviation, MAD):

    .. math::

        \text{RMS}_e > \text{med} + k \times 1.4826 \, \text{MAD}

    with :math:`k` the ``threshold``. The elements without finite residual are not outliers
    (their RMS is NaN).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis (its mesh gives the elements).

    integration_points : :class:`pysdic.IntegrationPoints`
        The integration points of the residuals.

    residuals : Sequence[:class:`pysdic.PairingResiduals`]
        The residuals of the pairings, computed with ``integration_points`` (see
        :func:`pysdic.compute_residuals` or :meth:`pysdic.SDICCorrelationTerm.compute_residuals`).

    threshold : Real, optional (default: ``3.0``)
        The number :math:`k` of robust standard deviations above the median.

    Returns
    -------
    :class:`ElementsResidualRMSResult`
        The RMS per element and per pairing, and the outliers.

    Raises
    ------
    ValueError
        If the residuals do not match the integration points.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import compute_elements_residual_rms

        quality = compute_elements_residual_rms(analysis, integration_points, correlation.compute_residuals())
        for pairing, rms in zip(quality.pairings, quality.pairing_rms):
            print(f"{pairing}: RMS {rms:.2f}")
        print(f"Badly correlated elements: {quality.bad_elements}")

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    mesh = analysis.mesh
    mesh.validate_integration_points(integration_points)
    threshold = _validate_real(threshold, "threshold", non_negative=True, finite=True)
    residuals = list(residuals)
    if not residuals:
        raise ValueError("Input 'residuals' must contain at least one PairingResiduals.")
    N_p, N_e = integration_points.n_points, mesh.n_elements
    elements = integration_points.element_indices
    weights = numpy.asarray(integration_points.weights, dtype=numpy.float64)

    rms = numpy.full((len(residuals), N_e), numpy.nan)
    n_points = numpy.zeros((len(residuals), N_e), dtype=numpy.int64)
    pairing_rms = numpy.full(len(residuals), numpy.nan)
    outliers = numpy.zeros((len(residuals), N_e), dtype=bool)
    for k, result in enumerate(residuals):
        if not isinstance(result, PairingResiduals):
            raise TypeError(f"Input 'residuals[{k}]' must be a PairingResiduals. Got type {type(result)}.")
        if result.n_points != N_p:
            raise ValueError(
                f"The residuals of {result.pairing} have {result.n_points} points, the integration points {N_p}."
            )
        valid = numpy.isfinite(result.residuals).all(axis=1) & (elements >= 0)
        squares = numpy.sum(numpy.square(result.residuals[valid]), axis=1) / result.n_channels  # (n,)
        w, e = weights[valid], elements[valid]
        sums = numpy.bincount(e, weights=w * squares, minlength=N_e)
        totals = numpy.bincount(e, weights=w, minlength=N_e)
        n_points[k] = numpy.bincount(e, minlength=N_e)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            rms[k] = numpy.where(totals > 0, numpy.sqrt(sums / totals), numpy.nan)
        if valid.any():
            pairing_rms[k] = numpy.sqrt(numpy.sum(w * squares) / numpy.sum(w))
        finite = numpy.isfinite(rms[k])
        if finite.any():
            median = numpy.median(rms[k, finite])
            spread = 1.4826 * numpy.median(numpy.abs(rms[k, finite] - median))
            outliers[k, finite] = rms[k, finite] > median + threshold * spread

    return ElementsResidualRMSResult(
        pairings=tuple(result.pairing for result in residuals),
        rms=rms,
        n_points=n_points,
        pairing_rms=pairing_rms,
        outliers=outliers,
    )


# =============================================================================
# Uncertainty of the unknowns
# =============================================================================
@dataclass(frozen=True)
class UncertaintyResult:
    r"""
    The approximate covariance of the unknowns of a correlation term, returned by
    :func:`compute_uncertainty`.

    Attributes
    ----------
    covariances : Dict[int, numpy.ndarray]
        For each optimized frame, the covariance :math:`(N_v, E, E)` of the displacement of each
        vertex (NaN for the vertices not computed or not constrained by the term).

    standard_deviations : Dict[int, numpy.ndarray]
        For each optimized frame, the standard deviation :math:`(N_v, E)` of each component
        (square root of the diagonal of the covariances).

    transformation_covariances : Dict[Tuple[str, int], numpy.ndarray]
        For each optimized transformation, the covariance :math:`(n_p, n_p)` of its parameters.

    residual_variance : float
        The variance :math:`s^2` of the residuals (gray levels²) used for the covariance.

    n_residuals : int
        The number of finite residuals :math:`m`.

    n_unknowns : int
        The number of unknowns constrained by the term :math:`n` (non-zero column of the
        Jacobian).

    """

    covariances: Dict[int, numpy.ndarray]
    standard_deviations: Dict[int, numpy.ndarray]
    transformation_covariances: Dict[Tuple[str, int], numpy.ndarray]
    residual_variance: float
    n_residuals: int
    n_unknowns: int


def _row_weights(residuals: List[PairingResiduals], weights: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Weights and raw residuals of the rows of the term: pairing by pairing, Fortran order of (N_p, C)
    row_weights = numpy.concatenate([numpy.tile(weights, result.n_channels) for result in residuals])
    raw = numpy.concatenate([result.residuals.ravel(order="F") for result in residuals])
    return row_weights, raw


def _solve_columns(factor, matrix: scipy.sparse.csc_matrix, columns: numpy.ndarray, n: int) -> numpy.ndarray:
    # Columns of H^-1 M H^-1 (n, len(columns)) for the selected columns
    unit = numpy.zeros((n, columns.size))
    unit[columns, numpy.arange(columns.size)] = 1.0
    first = factor.solve(unit)  # H^-1 e
    return factor.solve(matrix @ first)  # H^-1 M H^-1 e


def compute_uncertainty(
    correlation: SDICCorrelationTerm,
    parameters: Optional[ArrayLike] = None,
    *,
    image_noise_std: Optional[Real] = None,
    vertices: Optional[ArrayLike] = None,
    batch_size: Integral = 256,
) -> UncertaintyResult:
    r"""
    Estimate the covariance of the unknowns of a correlation term (displacements and
    parameters of transformations) from its Jacobian at convergence and from the noise of the
    gray levels.

    Near the solution, the Gauss-Newton step is linear in the residuals,
    :math:`\delta\vec{p} = -H^{-1} J^T \vec{r}` with :math:`H = J^T J`. The term weights the
    residuals by :math:`\sqrt{w_p}`: with a noise of variance :math:`s^2` on the raw residuals
    :math:`g_2 - g_1`, the covariance of the weighted residuals is :math:`s^2 W` with
    :math:`W = \text{diag}(w_p)`, and

    .. math::

        \text{Cov}(\vec{p}) \approx s^2 \, H^{-1} (J^T W J) H^{-1}

    (equal to :math:`s^2 w H^{-1}` for equal weights :math:`w`). The variance :math:`s^2` is:

    - by default, estimated a posteriori from the raw residuals at ``parameters``,
      :math:`s^2 = \sum r^2 / (m - n)`: it includes the noise of the images and the errors of
      the model (interpolation, gray level changes, ...);
    - with ``image_noise_std`` :math:`\sigma`, the noise of the two images of a pairing,
      :math:`s^2 = 2 \sigma^2`: only valid for the raw gray levels (term without gray level
      correction, the correction rescales the noise).

    Only the unknowns constrained by the term are kept (non-zero column of the Jacobian, e.g.
    not the vertices without integration points); a direction not observable by the term
    (e.g. the tangential displacement with stereo pairings only) gives a huge or infinite
    variance. The covariance does not include the other terms of a problem (e.g. a temporal
    regularization), which reduce the uncertainty.

    .. note::

        The covariance of a vertex needs two sparse solves per unknown of the vertex: give
        ``vertices`` to compute only some of them on a large mesh.

    Parameters
    ----------
    correlation : :class:`pysdic.SDICCorrelationTerm`
        The term, in mode ``"rJ"``.

    parameters : Optional[ArrayLike], optional
        The complete vector of unknowns at convergence (with a parametrization,
        ``parametrization.p_func(result.parameters)``). By default, the current state of the
        analysis (:meth:`pysdic.SDICCorrelationTerm.pack`).

    image_noise_std : Optional[Real], optional
        The standard deviation of the noise of the images (gray levels). By default, the
        variance is estimated from the residuals.

    vertices : Optional[ArrayLike], optional
        The vertices whose covariance is computed (indices). By default, all the vertices.

    batch_size : Integral, optional (default: ``256``)
        The number of columns solved at once (memory :math:`n \times` ``batch_size``).

    Returns
    -------
    :class:`UncertaintyResult`
        The covariances of the displacements and of the transformations.

    Raises
    ------
    ValueError
        If the term is not in mode ``"rJ"``, ``image_noise_std`` is used with a gray level
        correction, there are not more residuals than unknowns, or the Hessian is singular.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import compute_uncertainty

        analysis.apply_unknowns(unknowns, result.parameters)  # Measure done
        uncertainty = compute_uncertainty(correlation)  # At the current state of the analysis
        std = uncertainty.standard_deviations[1]  # (N_v, 3), frame 1
        print(f"Median uncertainty: {numpy.nanmedian(std, axis=0)} mm")

    """
    if not isinstance(correlation, SDICCorrelationTerm):
        raise TypeError(f"Input 'correlation' must be a SDICCorrelationTerm. Got type {type(correlation)}.")
    if correlation.mode != "rJ":
        raise ValueError(f"The term must be in mode 'rJ' (Jacobian of the residuals). Got {correlation.mode!r}.")
    if image_noise_std is not None:
        image_noise_std = _validate_real(image_noise_std, "image_noise_std", non_negative=True, finite=True)
        if correlation.gl_correction is not None:
            raise ValueError(
                "Input 'image_noise_std' is only valid without gray level correction (the correction rescales the "
                "noise): leave it to None to estimate the variance from the residuals."
            )
    batch_size = _validate_integer(batch_size, "batch_size", min_value=1)
    unknowns = correlation.unknowns
    analysis = correlation.analysis
    n_unknowns = analysis.n_unknowns(unknowns)
    mesh = correlation.analysis.mesh
    N_v, E = mesh.n_vertices, mesh.n_dimensions
    if vertices is None:
        vertices = numpy.arange(N_v)
    else:
        vertices = numpy.unique(_as_array(numpy.atleast_1d(vertices), "vertices", numpy.int64, shape=(None,), finite=True))
        if numpy.any((vertices < 0) | (vertices >= N_v)):
            raise ValueError(f"Input 'vertices' must be in [0, {N_v - 1}].")
    if parameters is None:
        parameters = correlation.pack()
    parameters = _as_array(parameters, "parameters", numpy.float64, shape=(n_unknowns,), finite=True)

    # Jacobian of the finite residuals and weights of the rows
    residuals = correlation.compute_residuals(parameters)
    row_weights, raw = _row_weights(residuals, numpy.asarray(correlation.integration_points.weights, dtype=numpy.float64))
    jacobian = scipy.sparse.csr_matrix(correlation.term.jacobian_func(parameters))
    if jacobian.shape[0] != raw.size:
        raise ValueError(f"The Jacobian has {jacobian.shape[0]} rows, the residuals of the pairings {raw.size}.")
    rows = numpy.isfinite(raw)
    jacobian = jacobian[rows]
    active = numpy.flatnonzero(numpy.diff(jacobian.tocsc().indptr) > 0)  # Constrained unknowns
    jacobian = jacobian[:, active]
    m, n = jacobian.shape
    if m <= n:
        raise ValueError(f"The term must have more finite residuals than unknowns. Got m={m}, n={n}.")
    if image_noise_std is None:
        variance = float(numpy.sum(numpy.square(raw[rows])) / (m - n))
    else:
        variance = 2.0 * image_noise_std**2

    hessian = (jacobian.T @ jacobian).tocsc()
    middle = (jacobian.T @ scipy.sparse.diags(row_weights[rows]) @ jacobian).tocsc()
    try:
        factor = scipy.sparse.linalg.splu(hessian)
    except RuntimeError as error:
        raise ValueError(
            "The Hessian of the term is singular: some unknowns are not observable (e.g. the tangential displacement "
            "with stereo pairings only). Reduce the unknowns (normal displacement, fixed parametrization)."
        ) from error
    position = numpy.full(n_unknowns, -1)
    position[active] = numpy.arange(n)

    # Groups of unknowns whose covariance block is computed: E components of a vertex, parameters of a transformation
    groups = []
    for frame in unknowns.optimized_frames:
        start = analysis.get_unknowns_frame_slice(unknowns, frame).start
        for vertex in vertices:
            groups.append((frame, int(vertex), start + numpy.arange(E) * N_v + vertex))
    for entry in unknowns.optimized_transformations:
        groups.append((entry, None, numpy.arange(n_unknowns)[analysis.get_unknowns_transformation_slice(unknowns, *entry)]))

    covariances = {frame: numpy.full((N_v, E, E), numpy.nan) for frame in unknowns.optimized_frames}
    transformation_covariances = {}
    batch: list = []

    def flush() -> None:
        columns = numpy.concatenate([position[indices][position[indices] >= 0] for _, _, indices in batch])
        solved = _solve_columns(factor, middle, columns, n) if columns.size else numpy.empty((n, 0))
        offset = 0
        for key, vertex, indices in batch:
            local = position[indices]
            kept = local >= 0
            block = numpy.full((indices.size, indices.size), numpy.nan)
            count = int(numpy.count_nonzero(kept))
            block[numpy.ix_(kept, kept)] = variance * solved[local[kept], offset : offset + count]
            offset += count
            if vertex is None:
                transformation_covariances[key] = block
            else:
                covariances[key][vertex] = block
        batch.clear()

    size = 0
    for group in groups:
        batch.append(group)
        size += group[2].size
        if size >= batch_size:
            flush()
            size = 0
    if batch:
        flush()

    standard_deviations = {
        frame: numpy.sqrt(numpy.diagonal(covariance, axis1=1, axis2=2)) for frame, covariance in covariances.items()
    }
    return UncertaintyResult(
        covariances=covariances,
        standard_deviations=standard_deviations,
        transformation_covariances=transformation_covariances,
        residual_variance=variance,
        n_residuals=m,
        n_unknowns=n,
    )