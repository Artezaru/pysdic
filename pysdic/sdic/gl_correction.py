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

r"""
Gray level corrections between the two views of a pairing.

The gray levels of the same material point can differ between two views (lighting,
exposure, reflections, ...). The corrections remove an affine change of the gray levels,
estimated by statistics (mean :math:`m` and standard deviation :math:`s`) over groups of
integration points (see :func:`compute_gray_level_groups`):

- ``ZNSSD`` (zero-mean normalized): both views are normalized,
  :math:`g' = \text{scale} \cdot (g - m) / s`. Each pairing gets residuals of the same
  order of magnitude (``scale = 255`` gives readable gray levels).
- ``matching``: the second view is mapped onto the statistics of the first one,
  :math:`g_2' = \text{scale} \cdot ((g_2 - m_2) \, s_1 / s_2 + m_1)` and
  :math:`g_1' = \text{scale} \cdot g_1`.

  .. warning::

      With several pairings, the ``matching`` correction keeps the contrast of the first
      view of each pairing: two pairings no longer have the same weight in the
      optimization (a pairing with a low contrast first view counts less). Prefer
      ``ZNSSD`` when several pairings are combined.

The statistics depend on the gray levels, so on the unknowns. Two derivatives are
available for each correction:

- approximate (default): :math:`m` and :math:`s` are considered constant, the Jacobians of
  the gray levels are only multiplied by the factor of the correction. The corrected
  values are exact, the Jacobians remain sparse (one point per row).
- exact: the derivatives of :math:`m` and :math:`s` are included. The derivative of a
  corrected gray level depends on all the points of its group: the Jacobian is returned as
  a sparse matrix ``derivative`` to apply to the raw Jacobians (see the exact functions).

The non-finite gray levels (points out of the ROI) are ignored by the statistics, and a
group with a zero standard deviation gives non-finite corrected values.

The groups are neighborhoods of elements: the statistics of a group are computed from the
sums of the values of each element (one pass over the points), combined with the sparse
matrix of the elements of each group (see :class:`GrayLevelGroups`). They only depend on the
connectivity of the mesh and on the elements of the integration points: compute them once
with :func:`compute_gray_level_groups` and share them between the terms
(``gray_level_groups`` of :class:`pysdic.SDICCorrelationTerm`).
"""


import dataclasses
from dataclasses import dataclass
from numbers import Integral, Real
from typing import List, Optional, Tuple

import numpy
import scipy.sparse

from .. import fem
from ..geometry.mesh import Mesh
from ..geometry.integration_points import IntegrationPoints
from ..imaging.view import ImageProjectionResult
from .._validation import _as_integer_array, _validate_integer, _validate_real

# Maximum number of non-zero entries of an exact derivative (the global group is dense)
_MAX_EXACT_ENTRIES = 50_000_000


# =============================================================================
# Internal helpers
# =============================================================================
def _check_inputs(
    result_1: ImageProjectionResult, result_2: ImageProjectionResult, groups: "GrayLevelGroups", scale: Real
) -> float:
    for name, result in (("result_1", result_1), ("result_2", result_2)):
        if not isinstance(result, ImageProjectionResult):
            raise TypeError(f"Input '{name}' must be an ImageProjectionResult. Got type {type(result)}.")
        if result.x.ndim != 2 or result.x.shape[0] != groups.n_points:
            raise ValueError(
                f"The gray levels of '{name}' must have shape ({groups.n_points}, C). Got {result.x.shape}."
            )
    if result_1.x.shape != result_2.x.shape:
        raise ValueError(
            f"The gray levels of the two views must have the same shape. Got {result_1.x.shape} "
            f"and {result_2.x.shape}."
        )
    if not isinstance(groups, GrayLevelGroups):
        raise TypeError(f"Input 'groups' must be a GrayLevelGroups. Got type {type(groups)}.")
    return _validate_real(scale, "scale", positive=True, finite=True)


def _statistics(values: numpy.ndarray, groups: "GrayLevelGroups") -> Tuple[numpy.ndarray, ...]:
    # Number of finite values n, mean m and standard deviation s (ddof=0) of each group and
    # channel (N_g, C), ignoring the non-finite values. NaN for the groups without values.
    #
    # The sums over the points of a group are the sums over its elements of the sums over the
    # points of each element: (members @ v)[g] = (group_elements @ S)[g] with
    # S[e] = sum of v over the points of the element e (bincount, one pass over the points).
    # The 3 C sums (count, values, squared values) are combined in a single sparse product.
    valid = numpy.isfinite(values)  # (N_p, C)
    C = values.shape[1]
    zeroed = numpy.where(valid, values, 0.0)
    stacked = numpy.concatenate((valid.astype(numpy.float64), zeroed, zeroed * zeroed), axis=1)  # (N_p, 3 C)
    element_sums = numpy.empty((groups.n_elements, 3 * C), dtype=numpy.float64)
    for column in range(3 * C):
        element_sums[:, column] = numpy.bincount(
            groups.point_elements, weights=stacked[:, column], minlength=groups.n_elements
        )
    sums = groups.group_elements @ element_sums  # (N_g, 3 C)
    n, total, total_squares = sums[:, :C], sums[:, C : 2 * C], sums[:, 2 * C :]
    with numpy.errstate(invalid="ignore", divide="ignore"):
        m = total / n
        variance = total_squares / n - m**2
    s = numpy.sqrt(numpy.maximum(variance, 0.0))
    return n, m, s


def _scaled_result(result: ImageProjectionResult, values: numpy.ndarray, factor: numpy.ndarray) -> ImageProjectionResult:
    # Result with corrected values and Jacobians multiplied by the factor (N_p, C) of each row
    def scaled(jacobian):
        return None if jacobian is None else jacobian * factor[..., numpy.newaxis]

    return dataclasses.replace(
        result, x=values, jacobian_dx=scaled(result.jacobian_dx), jacobian_dp=scaled(result.jacobian_dp)
    )


def _values_result(result: ImageProjectionResult, values: numpy.ndarray) -> ImageProjectionResult:
    # Result with corrected values and without Jacobians (exact corrections)
    return dataclasses.replace(result, x=values, jacobian_dx=None, jacobian_dp=None)


def _group_blocks(groups: "GrayLevelGroups") -> List[Tuple[numpy.ndarray, numpy.ndarray]]:
    # For each group: the points corrected with its statistics and the points of its statistics
    order = numpy.argsort(groups.point_groups, kind="stable")
    bounds = numpy.searchsorted(groups.point_groups[order], numpy.arange(groups.n_groups + 1))
    members = groups.members
    blocks = []
    for group in range(groups.n_groups):
        rows = order[bounds[group] : bounds[group + 1]]
        if rows.size > 0:
            blocks.append((rows, members.indices[members.indptr[group] : members.indptr[group + 1]]))
    n_entries = sum(rows.size * columns.size for rows, columns in blocks)
    if n_entries > _MAX_EXACT_ENTRIES:
        raise ValueError(
            f"The exact derivative of the correction would have {n_entries} entries per channel. Use the "
            f"approximate correction or smaller groups (neighborhood of the elements)."
        )
    return blocks


def _derivative_block(
    blocks: List[Tuple[numpy.ndarray, numpy.ndarray]],
    n_points: int,
    diagonal: Optional[numpy.ndarray],
    coefficient: numpy.ndarray,
    z_rows: numpy.ndarray,
    column_values: numpy.ndarray,
    column_mean: numpy.ndarray,
    column_std: numpy.ndarray,
    point_groups: numpy.ndarray,
) -> scipy.sparse.csr_matrix:
    # Sparse (N_p, N_p) matrix of one channel with the entries
    # D[i, j] = diagonal[i] * delta_ij + coefficient[g] * (1 + z_rows[i] * z_g(j))
    # for the points i of each group g and the finite points j of its statistics, where
    # z_g(j) = (column_values[j] - column_mean[g]) / column_std[g] is normalized with the
    # statistics of the group g (not of the group of j: the groups can overlap)
    rows, columns, values = [], [], []
    valid_columns = numpy.isfinite(column_values)
    for group_rows, group_columns in blocks:
        group_columns = group_columns[valid_columns[group_columns]]
        if group_columns.size == 0:
            continue
        group = point_groups[group_rows[0]]
        z_columns = (column_values[group_columns] - column_mean[group]) / column_std[group]
        block = coefficient[group] * (1.0 + numpy.outer(z_rows[group_rows], z_columns))
        rows.append(numpy.repeat(group_rows, group_columns.size))
        columns.append(numpy.tile(group_columns, group_rows.size))
        values.append(block.ravel())
    if diagonal is not None:
        rows.append(numpy.arange(n_points))
        columns.append(numpy.arange(n_points))
        values.append(diagonal)
    if not rows:
        return scipy.sparse.csr_matrix((n_points, n_points))
    return scipy.sparse.coo_matrix(
        (numpy.concatenate(values), (numpy.concatenate(rows), numpy.concatenate(columns))), shape=(n_points, n_points)
    ).tocsr()


@dataclass(frozen=True, slots=True)
class GrayLevelGroups:
    r"""
    The groups of integration points used to compute the statistics of the gray level
    corrections (see :func:`compute_gray_level_groups`).

    A group is a set of elements: the statistics of the group :math:`k` are computed over the
    integration points located in the elements of the row :math:`k` of ``group_elements``.
    They are computed from the sums over each element (one pass over the points) and a single
    product with ``group_elements``, whose size only depends on the mesh (not on the number of
    points per element).

    The groups only depend on the connectivity of the mesh and on the elements of the points:
    compute them once and share them between the terms of the same points (see
    ``gray_level_groups`` of :class:`pysdic.SDICCorrelationTerm`).

    Parameters
    ----------
    point_groups : numpy.ndarray
        An integer array of shape :math:`(N_p,)`: the group of each point, whose statistics
        correct its gray level.

    point_elements : numpy.ndarray
        An integer array of shape :math:`(N_p,)`: the element of each point (all the points
        must be located, indices in :math:`[0, N_e)`).

    group_elements : scipy.sparse.csr_matrix
        A matrix of shape :math:`(N_g, N_e)`: the non-zero entries of the row :math:`k` are
        the elements whose points are used to compute the statistics of the group :math:`k`.
        Converted once to a ``csr_matrix`` of ones in float64 (sorted indices).

    Raises
    ------
    ValueError
        If the shapes are not consistent or an index is out of range.

    """

    point_groups: numpy.ndarray
    point_elements: numpy.ndarray
    group_elements: scipy.sparse.csr_matrix

    def __post_init__(self) -> None:
        point_groups = _as_integer_array(self.point_groups, "point_groups", shape=(None,), shape_hint="(N_p,)")
        point_elements = _as_integer_array(self.point_elements, "point_elements", shape=(None,), shape_hint="(N_p,)")
        if point_groups.shape != point_elements.shape:
            raise ValueError(
                f"Inputs 'point_groups' and 'point_elements' must have the same shape (N_p,). "
                f"Got {point_groups.shape} and {point_elements.shape}."
            )
        if not scipy.sparse.issparse(self.group_elements) or self.group_elements.ndim != 2:
            raise TypeError(
                f"Input 'group_elements' must be a 2D scipy sparse matrix (N_g, N_e). Got type {type(self.group_elements)}."
            )
        group_elements = scipy.sparse.csr_matrix(self.group_elements, dtype=numpy.float64)
        group_elements.eliminate_zeros()
        group_elements.data[:] = 1.0  # Membership: ones
        group_elements.sort_indices()
        N_g, N_e = group_elements.shape
        if point_groups.size > 0:
            if point_groups.min() < 0 or point_groups.max() >= N_g:
                raise ValueError(f"Input 'point_groups' must contain indices in [0, {N_g}).")
            if point_elements.min() < 0 or point_elements.max() >= N_e:
                raise ValueError(f"Input 'point_elements' must contain indices in [0, {N_e}) (located points).")
        for array in (point_groups, point_elements):
            array.flags.writeable = False
        object.__setattr__(self, "point_groups", point_groups)
        object.__setattr__(self, "point_elements", point_elements)
        object.__setattr__(self, "group_elements", group_elements)

    @property
    def n_points(self) -> int:
        r"""[Get] The number of integration points :math:`N_p`."""
        return self.point_groups.shape[0]

    @property
    def n_groups(self) -> int:
        r"""[Get] The number of groups :math:`N_g`."""
        return self.group_elements.shape[0]

    @property
    def n_elements(self) -> int:
        r"""[Get] The number of elements :math:`N_e` of the mesh."""
        return self.group_elements.shape[1]

    @property
    def members(self) -> scipy.sparse.csr_matrix:
        r"""
        [Get] The points of each group, a boolean ``csr_matrix`` of shape :math:`(N_g, N_p)`
        (sorted indices), computed at each call: ``group_elements @ location`` with
        :math:`\text{location}_{e,p} = 1` if the point :math:`p` is in the element :math:`e`.

        Only needed by the exact derivatives of the corrections: its size is about the number
        of points per element times the size of ``group_elements``.
        """
        location = scipy.sparse.csr_matrix(
            (numpy.ones(self.n_points, dtype=bool), (self.point_elements, numpy.arange(self.n_points))),
            shape=(self.n_elements, self.n_points),
        )
        members = (self.group_elements.astype(bool) @ location).tocsr()
        members.sort_indices()
        return members


def compute_gray_level_groups(
    mesh: Mesh,
    integration_points: IntegrationPoints,
    neighborhood: Optional[Integral] = None,
) -> GrayLevelGroups:
    r"""
    Compute the groups of integration points for the statistics of the gray level
    corrections.

    - ``neighborhood=None``: a single group with all the points (global correction).
    - ``neighborhood=n``: one group per element. The gray levels of the points of an
      element are corrected with the statistics of the points located in the elements
      within a distance :math:`n` (see :func:`pysdic.fem.compute_elements_neighborhood`):
      ``0`` for the element alone, ``1`` for the element and the elements sharing a vertex
      with it, ...

    Only the neighborhood of the elements is computed (not the points of each group, see
    :class:`GrayLevelGroups`). The groups only depend on the connectivity of the mesh and on
    the elements of the points: compute them once and give them to all the terms built with
    the same mesh and integration points (``gray_level_groups`` of
    :class:`pysdic.SDICCorrelationTerm`).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh in which the points are located.

    integration_points : :class:`pysdic.IntegrationPoints`
        The integration points, located in the mesh.

    neighborhood : Optional[Integral], optional
        The size of the neighborhood (non-negative), or None for a global correction.

    Returns
    -------
    :class:`GrayLevelGroups`
        The groups of the points.

    Raises
    ------
    ValueError
        If the points are not consistent with the mesh or not all located.

    """
    mesh.validate_integration_points(integration_points)
    if not numpy.all(integration_points.is_located()):
        raise ValueError("The integration points must all be located (use 'remove_unlocated').")
    neighborhood = _validate_integer(neighborhood, "neighborhood", min_value=0, allow_none=True)
    N_p, N_e = integration_points.n_points, mesh.n_elements
    point_elements = numpy.asarray(integration_points.element_indices, dtype=numpy.int64)
    if neighborhood is None:
        group_elements = scipy.sparse.csr_matrix(numpy.ones((1, N_e), dtype=numpy.float64))
        return GrayLevelGroups(numpy.zeros(N_p, dtype=numpy.int64), point_elements, group_elements)
    group_elements = fem.compute_elements_neighborhood(
        mesh.elements,
        max_distance=neighborhood,
        self_neighborhood=True,
    )  # (N_e, N_e)
    return GrayLevelGroups(point_elements, point_elements, group_elements)


# =============================================================================
# ZNSSD
# =============================================================================
def znssd_correction(
    result_1: ImageProjectionResult,
    result_2: ImageProjectionResult,
    groups: GrayLevelGroups,
    *,
    scale: Real = 1.0,
) -> Tuple[ImageProjectionResult, ImageProjectionResult]:
    r"""
    Normalize the gray levels of both views (ZNSSD) with the statistics of their groups,
    with the approximate derivative (constant statistics):

    .. math::

        g' = \text{scale} \cdot \frac{g - m}{s},
        \qquad
        \frac{\partial g'}{\partial \gamma} \approx \frac{\text{scale}}{s} \frac{\partial g}{\partial \gamma}

    Parameters
    ----------
    result_1 : :class:`pysdic.ImageProjectionResult`
        The projection in the first view, with gray levels of shape :math:`(N_p, C)`.

    result_2 : :class:`pysdic.ImageProjectionResult`
        The projection in the second view, with gray levels of shape :math:`(N_p, C)`.

    groups : :class:`GrayLevelGroups`
        The groups of the points (see :func:`compute_gray_level_groups`).

    scale : Real, optional (default: ``1.0``)
        The factor applied after the normalization (strictly positive), e.g. ``255`` to get
        readable gray levels.

    Returns
    -------
    Tuple[:class:`pysdic.ImageProjectionResult`, :class:`pysdic.ImageProjectionResult`]
        The corrected projections (new objects): gray levels and Jacobians (``jacobian_dx``
        and ``jacobian_dp`` if present) corrected.

    """
    scale = _check_inputs(result_1, result_2, groups, scale)
    corrected = []
    for result in (result_1, result_2):
        _, m, s = _statistics(result.x, groups)
        m, s = m[groups.point_groups], s[groups.point_groups]  # (N_p, C)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            corrected.append(_scaled_result(result, scale * (result.x - m) / s, scale / s))
    return corrected[0], corrected[1]


def znssd_correction_exact(
    result_1: ImageProjectionResult,
    result_2: ImageProjectionResult,
    groups: GrayLevelGroups,
    *,
    scale: Real = 1.0,
) -> Tuple[ImageProjectionResult, ImageProjectionResult, scipy.sparse.csr_matrix]:
    r"""
    Normalize the gray levels of both views (ZNSSD) with the statistics of their groups,
    with the exact derivative. For a point :math:`i` of the group :math:`G` (statistics over
    :math:`n` finite points) and :math:`z = (g - m) / s` (statistics of :math:`G`, also for the
    points :math:`j` of overlapping groups):

    .. math::

        \frac{\partial g'_i}{\partial g_j} = \frac{\text{scale}}{s}
        \left( \delta_{ij} - \frac{1 + z_i z_j}{n} \right) \quad (j \in G)

    The corrected Jacobians are ``derivative @ [J_1; J_2]`` where :math:`J_k` are the raw
    Jacobians of the gray levels of the view :math:`k` (rows in Fortran order of
    :math:`(N_p, C)`: row :math:`p + c N_p`).

    .. note::

        The exact derivative has :math:`\sum_G |G|^2` entries per channel: a global group
        (``neighborhood=None``) gives a dense matrix, limited to small numbers of points.

    Parameters
    ----------
    result_1, result_2, groups, scale
        See :func:`znssd_correction`.

    Returns
    -------
    result_1 : :class:`pysdic.ImageProjectionResult`
        The first projection with corrected gray levels, **without** Jacobians.

    result_2 : :class:`pysdic.ImageProjectionResult`
        The second projection with corrected gray levels, **without** Jacobians.

    derivative : scipy.sparse.csr_matrix
        The derivative of the corrected gray levels of both views with respect to the raw
        ones, of shape :math:`(2 N_p C, 2 N_p C)`: rows and columns :math:`[0, N_p C)` for the
        first view, :math:`[N_p C, 2 N_p C)` for the second one. Block diagonal for ZNSSD.

    """
    scale = _check_inputs(result_1, result_2, groups, scale)
    blocks = _group_blocks(groups)
    N_p, C = result_1.x.shape
    corrected, channel_blocks = [], []
    for result in (result_1, result_2):
        n, m, s = _statistics(result.x, groups)
        point_m, point_s = m[groups.point_groups], s[groups.point_groups]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            z = (result.x - point_m) / point_s  # (N_p, C)
            corrected.append(_values_result(result, scale * z))
            coefficient = -scale / (s * n)  # (N_g, C)
            diagonal = scale / point_s
            channel_blocks.append(
                [
                    _derivative_block(
                        blocks, N_p, diagonal[:, c], coefficient[:, c], z[:, c], result.x[:, c], m[:, c], s[:, c],
                        groups.point_groups,
                    )
                    for c in range(C)
                ]
            )
    derivative = scipy.sparse.block_diag(channel_blocks[0] + channel_blocks[1], format="csr")
    return corrected[0], corrected[1], derivative


# =============================================================================
# Matching of the second view onto the first one
# =============================================================================
def matching_correction(
    result_1: ImageProjectionResult,
    result_2: ImageProjectionResult,
    groups: GrayLevelGroups,
    *,
    scale: Real = 1.0,
) -> Tuple[ImageProjectionResult, ImageProjectionResult]:
    r"""
    Map the gray levels of the second view onto the statistics of the first one, with the
    approximate derivative (constant statistics):

    .. math::

        g_1' = \text{scale} \cdot g_1,
        \qquad
        g_2' = \text{scale} \cdot \left( (g_2 - m_2) \frac{s_1}{s_2} + m_1 \right)

    .. warning::

        With several pairings, each pairing keeps the contrast of its first view: the
        pairings no longer have the same weight in the optimization. Prefer
        :func:`znssd_correction` when several pairings are combined.

    Parameters
    ----------
    result_1, result_2, groups, scale
        See :func:`znssd_correction`.

    Returns
    -------
    Tuple[:class:`pysdic.ImageProjectionResult`, :class:`pysdic.ImageProjectionResult`]
        The corrected projections (new objects).

    """
    scale = _check_inputs(result_1, result_2, groups, scale)
    _, m_1, s_1 = _statistics(result_1.x, groups)
    _, m_2, s_2 = _statistics(result_2.x, groups)
    g = groups.point_groups
    with numpy.errstate(invalid="ignore", divide="ignore"):
        ratio = s_1[g] / s_2[g]
        values_2 = scale * ((result_2.x - m_2[g]) * ratio + m_1[g])
    return (
        _scaled_result(result_1, scale * result_1.x, numpy.full_like(result_1.x, scale)),
        _scaled_result(result_2, values_2, scale * ratio),
    )


def matching_correction_exact(
    result_1: ImageProjectionResult,
    result_2: ImageProjectionResult,
    groups: GrayLevelGroups,
    *,
    scale: Real = 1.0,
) -> Tuple[ImageProjectionResult, ImageProjectionResult, scipy.sparse.csr_matrix]:
    r"""
    Map the gray levels of the second view onto the statistics of the first one, with the
    exact derivative. For a point :math:`i` of the group :math:`G`, with
    :math:`z_k = (g_k - m_k) / s_k` (statistics of the group :math:`G`):

    .. math::

        \frac{\partial g'_{2,i}}{\partial g_{2,j}} = \text{scale} \frac{s_1}{s_2}
        \left( \delta_{ij} - \frac{1 + z_{2,i} z_{2,j}}{n_2} \right),
        \qquad
        \frac{\partial g'_{2,i}}{\partial g_{1,j}} = \text{scale} \frac{1 + z_{2,i} z_{1,j}}{n_1}

    .. warning::

        With several pairings, each pairing keeps the contrast of its first view: the
        pairings no longer have the same weight in the optimization.

    Parameters
    ----------
    result_1, result_2, groups, scale
        See :func:`znssd_correction`.

    Returns
    -------
    result_1, result_2, derivative
        See :func:`znssd_correction_exact`. The derivative has a block coupling the second
        view with the first one.

    """
    scale = _check_inputs(result_1, result_2, groups, scale)
    blocks = _group_blocks(groups)
    N_p, C = result_1.x.shape
    g = groups.point_groups
    n_1, m_1, s_1 = _statistics(result_1.x, groups)
    n_2, m_2, s_2 = _statistics(result_2.x, groups)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        z_2 = (result_2.x - m_2[g]) / s_2[g]
        values_2 = scale * (z_2 * s_1[g] + m_1[g])
        coefficient_22 = -scale * s_1 / (s_2 * n_2)
        coefficient_21 = scale / n_1
        diagonal_22 = scale * s_1[g] / s_2[g]

    NC = N_p * C
    identity = scipy.sparse.identity(NC, format="csr") * scale
    block_22 = scipy.sparse.block_diag(
        [
            _derivative_block(
                blocks, N_p, diagonal_22[:, c], coefficient_22[:, c], z_2[:, c], result_2.x[:, c], m_2[:, c], s_2[:, c], g
            )
            for c in range(C)
        ],
        format="csr",
    )
    block_21 = scipy.sparse.block_diag(
        [
            _derivative_block(
                blocks, N_p, None, coefficient_21[:, c], z_2[:, c], result_1.x[:, c], m_1[:, c], s_1[:, c], g
            )
            for c in range(C)
        ],
        format="csr",
    )
    derivative = scipy.sparse.bmat([[identity, None], [block_21, block_22]], format="csr")
    return _values_result(result_1, scale * result_1.x), _values_result(result_2, values_2), derivative