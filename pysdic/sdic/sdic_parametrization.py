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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy
import pysolvegn
import scipy.sparse
from numpy.typing import ArrayLike

from ..geometry.integration_points import IntegrationPoints
from ..geometry.mesh import Mesh
from .._validation import _as_array, _validate_bool
from .sdic_analysis import SDICAnalysis
from .sdic_unknowns import SDICUnknowns


# =============================================================================
# Internal helpers
# =============================================================================
def _build_solver_parametrization(p_func: Callable, jacobian_func: Callable) -> Any:
    # The pysolvegn.Parametrization q -> p of the functions (the only call to its constructor)
    return pysolvegn.Parametrization(parametric_func=p_func, jacobian_func=jacobian_func)


# =============================================================================
# Supported vertices
# =============================================================================
def compute_supported_vertices(mesh: Mesh, integration_points: IntegrationPoints) -> numpy.ndarray:
    r"""
    Compute the vertices of a mesh that support integration points: the vertices of the
    elements containing at least one located integration point.

    In the correlation term (see :class:`pysdic.SDICCorrelationTerm`), the displacement of a vertex only
    appears through the integration points of its elements: the displacement of a vertex
    that supports no integration point has a zero column in the Jacobian. Such vertices
    make the problem singular, unless another term constrains them (e.g. a temporal
    regularization). Use this function to select the displacements to optimize with
    :class:`pysdic.SDICFixedParametrization` (or :meth:`pysdic.SDICAnalysis.compute_unknowns_indices`).

    .. note::

        To exclude elements from the SDIC operators, remove their integration points (the
        operators are built from the integration points only). Do not modify the mesh of
        the analysis: its connectivity defines the element indices of the integration
        points and the neighborhoods of the gray level corrections.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh, with an element type.

    integration_points : :class:`pysdic.IntegrationPoints`
        The integration points, consistent with the mesh (the unlocated points are
        ignored).

    Returns
    -------
    numpy.ndarray
        The sorted indices of the supported vertices.

    Examples
    --------
    .. code-block:: python

        vertices = compute_supported_vertices(analysis.mesh, integration_points)
        print(f"{vertices.size} / {analysis.mesh.n_vertices} vertices support integration points")

    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    mesh.validate_integration_points(integration_points)
    elements = numpy.unique(integration_points.element_indices[integration_points.is_located()])
    return numpy.unique(mesh.elements[elements])


# =============================================================================
# Fixed parametrization
# =============================================================================
class SDICFixedParametrization:
    r"""
    A parametrization of the unknowns of a SDIC problem optimizing only the displacements of
    selected vertices and components of the optimized frames (and all the parameters of the
    optimized transformations), the other unknowns keeping their value in the analysis. The
    ``pysolvegn.Parametrization`` to give to ``pysolvegn.solve`` is the property
    :attr:`parametrization`.

    .. important::

        ``pysolvegn.Parametrization`` is **not** :class:`pysdic.Parametrization`, despite the
        same name and a close purpose:

        - :class:`pysdic.Parametrization` re-expresses the parameters of a
          :class:`pysdic.Transformation` (``encode`` / ``decode``, see
          :class:`pysdic.ReparametrizedTransformation`).
        - ``pysolvegn.Parametrization`` maps the unknowns :math:`\vec{q}` optimized by the
          solver to the parameters :math:`\vec{p} = P(\vec{q})` given to the terms of a
          ``pysolvegn`` problem. :class:`SDICFixedParametrization` builds one.

    The terms of the problem (:class:`pysdic.SDICCorrelationTerm`,
    :class:`pysdic.SDICTemporalRegularizationTerm`) keep working on the complete vector
    :math:`\vec{p} = [\vec{p}_U, \vec{p}_T]` of ``unknowns``. The parametrization is the
    affine map

    .. math::

        \vec{p} = P(\vec{q}) = \vec{p}_{current} \text{ with } \vec{p}[I] = \vec{q},
        \qquad \frac{\partial \vec{p}}{\partial \vec{q}} = S

    where :math:`I` are the selected indices (see :meth:`pysdic.SDICAnalysis.compute_unknowns_indices`) and
    :math:`S` the constant selection matrix. The values of the unknowns that are not selected
    are **read from the analysis at each call** (:meth:`pysdic.SDICAnalysis.pack_unknowns`): after a
    convergence written in the analysis (:meth:`reparametrized_apply`), the same parametrization can be used
    again for a new resolution, without building it again.

    **Initial vector.** With a parametrization, the solver starts from the reduced vector
    :math:`\vec{q}_0` given by :meth:`reparametrized_pack`, not from
    ``analysis.pack_unknowns(unknowns)`` (the complete vector :math:`\vec{p}_0`, used without
    parametrization). The three methods mirror those of :class:`pysdic.SDICAnalysis` on the
    reduced vector :math:`\vec{q}`:

    ==========================================  ===================================================
    Without parametrization (:math:`\vec{p}`)   With this parametrization (:math:`\vec{q}`)
    ==========================================  ===================================================
    ``analysis.pack_unknowns(unknowns)``        :meth:`reparametrized_pack`
    ``analysis.unpack_unknowns(unknowns, p)``   :meth:`reparametrized_unpack` (q)
    ``analysis.apply_unknowns(unknowns, p)``    :meth:`reparametrized_apply` (q)
    ==========================================  ===================================================

    As for the unknowns, :math:`\vec{q}` holds **total** values (displacements from the
    reference mesh, parameters of the transformations), not increments.

    .. warning::

        **With a parametrization, always use** :meth:`reparametrized_pack`,
        :meth:`reparametrized_unpack` **and** :meth:`reparametrized_apply` **on the vectors of
        the solver** (``p0`` and ``result.parameters`` are reduced vectors :math:`\vec{q}`).
        ``analysis.pack_unknowns``, ``analysis.unpack_unknowns`` and
        ``analysis.apply_unknowns`` (and ``pack``, ``unpack``, ``apply`` of the terms) work on
        the complete vector :math:`\vec{p}`: giving them :math:`\vec{q}` raises an error (wrong
        size) or, worse, misplaces the values if the sizes happen to match.

        .. code-block:: python

            q0 = parametrization.reparametrized_pack()               # NOT analysis.pack_unknowns(unknowns)
            result = pysolvegn.solve(terms, q0, parametrization=parametrization.parametrization)
            parametrization.reparametrized_apply(result.parameters)  # NOT analysis.apply_unknowns(unknowns, ...)

    Typical uses:

    - optimize only the vertices supporting integration points (see
      :func:`pysdic.compute_supported_vertices`), the other ones keeping their displacement;
    - optimize only some components, e.g. the in-plane components :math:`x` and :math:`y`.

    The object is immutable: :meth:`with_unknowns` and :meth:`with_options` create a new
    parametrization.

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis of the problem, shared with its terms.

    unknowns : :class:`pysdic.SDICUnknowns`
        The unknowns of the problem, shared with its terms (validated by the analysis).

    vertices : Optional[ArrayLike], optional
        The vertices whose displacement is optimized: indices or a boolean mask of shape
        :math:`(N_v,)`. By default, all the vertices.

    components : Optional[Sequence[Integral]], optional
        The components of the displacement that are optimized. By default, all the
        components.

    sparse : bool, optional (default: ``True``)
        If True, the Jacobian :math:`S` is a ``scipy.sparse.csr_matrix`` (recommended: the
        dense matrix has :math:`\dim(\vec{p}) \times \dim(\vec{q})` entries).

    Raises
    ------
    TypeError
        If ``analysis`` is not a :class:`pysdic.SDICAnalysis` or ``unknowns`` not a
        :class:`pysdic.SDICUnknowns`.

    ValueError
        If no unknown is selected, or an input has a wrong value.

    Examples
    --------
    Measure the displacements of several frames, only for the vertices that support
    integration points and only for the in-plane components:

    .. code-block:: python

        import pysolvegn
        from pysdic import (SDICCorrelationTerm, SDICFixedParametrization, SDICUnknowns,
                            compute_supported_vertices)

        unknowns = SDICUnknowns([1], [])
        correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)
        vertices = compute_supported_vertices(analysis.mesh, integration_points)
        fixed = SDICFixedParametrization(analysis, unknowns, vertices=vertices, components=[0, 1])

        for loop in range(3):  # The same parametrization: the fixed values are read at each call
            result = pysolvegn.solve([correlation.term], fixed.reparametrized_pack(),
                                     parametrization=fixed.parametrization, max_iteration=20)
            fixed.reparametrized_apply(result.parameters)

    """

    __slots__ = ("_analysis", "_unknowns", "_vertices", "_components", "_sparse", "_indices", "_selection", "_parametrization")

    def __init__(
        self,
        analysis: SDICAnalysis,
        unknowns: SDICUnknowns,
        *,
        vertices: Optional[ArrayLike] = None,
        components: Optional[Sequence[Integral]] = None,
        sparse: bool = True,
    ) -> None:
        if not isinstance(analysis, SDICAnalysis):
            raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
        if not isinstance(unknowns, SDICUnknowns):
            raise TypeError(f"Input 'unknowns' must be a SDICUnknowns. Got type {type(unknowns)}.")
        analysis.validate_unknowns(unknowns)
        self._analysis = analysis
        self._unknowns = unknowns
        self._sparse = _validate_bool(sparse, "sparse")
        indices = analysis.compute_unknowns_indices(unknowns, vertices=vertices, components=components)
        if indices.size == 0:
            raise ValueError("No unknown is selected: select at least one vertex and one component, or a transformation.")
        self._vertices = None if vertices is None else numpy.array(vertices)
        self._components = None if components is None else tuple(int(c) for c in numpy.atleast_1d(components))
        indices.setflags(write=False)
        self._indices = indices
        n_p, n_q = analysis.n_unknowns(unknowns), indices.size
        selection = scipy.sparse.csr_matrix(
            (numpy.ones(n_q), (indices, numpy.arange(n_q))), shape=(n_p, n_q)
        )
        self._selection = selection if self._sparse else selection.toarray()
        self._parametrization = _build_solver_parametrization(self.p_func, lambda q: self._selection)

    # ------------------------------------------------------------------ Properties
    @property
    def unknowns(self) -> SDICUnknowns:
        r"""The unknowns of the problem (the complete vector :math:`\vec{p}`)."""
        return self._unknowns

    @property
    def analysis(self) -> Any:
        r"""The analysis of the problem (shared with its terms)."""
        return self._analysis

    @property
    def parametrization(self) -> Any:
        r"""The ``pysolvegn.Parametrization`` :math:`\vec{q} \mapsto \vec{p}`, to give to ``pysolvegn.solve``."""
        return self._parametrization

    @property
    def indices(self) -> numpy.ndarray:
        r"""The indices :math:`I` of the selected unknowns in :math:`\vec{p}` (read-only)."""
        return self._indices

    @property
    def n_reparametrized(self) -> int:
        r"""The number of optimized unknowns :math:`\dim(\vec{q})`."""
        return int(self._indices.size)

    @property
    def sparse(self) -> bool:
        r"""Whether the Jacobian :math:`S` is sparse."""
        return self._sparse

    # ------------------------------------------------------------------ Vectors
    def p_func(self, q: ArrayLike) -> numpy.ndarray:
        r"""
        The complete vector :math:`\vec{p} = P(\vec{q})`: the current values of the unknowns
        in the analysis, with the selected ones replaced by :math:`\vec{q}`.
        """
        q = _as_array(q, "q", numpy.float64, shape=(self._indices.size,), finite=False)
        p = self._analysis.pack_unknowns(self._unknowns)
        p[self._indices] = q
        return p

    def reparametrized_pack(self) -> numpy.ndarray:
        r"""
        The reduced vector :math:`\vec{q}` of the current state of the analysis: the initial
        parameters ``q0`` of a resolution with :attr:`parametrization` (``analysis.pack_unknowns(unknowns)``
        gives the complete vector, used without parametrization).

        .. danger::

            With this parametrization, always use this method for ``p0``, never ``analysis.pack_unknowns(unknowns)``.
        """
        return self._analysis.pack_unknowns(self._unknowns)[self._indices]

    def reparametrized_unpack(
        self, q: ArrayLike
    ) -> Tuple[Dict[int, numpy.ndarray], Dict[Tuple[str, int], numpy.ndarray]]:
        r"""
        Split a reduced vector :math:`\vec{q}` (e.g. ``result.parameters``) into the complete
        displacements of the optimized frames and the parameters of the optimized
        transformations, the values not selected being those of the analysis:
        ``analysis.unpack_unknowns(unknowns, p_func(q))``. The analysis is not modified.

        Returns
        -------
        displacements : Dict[int, numpy.ndarray]
            For each optimized frame, its displacement of shape :math:`(N_v, E)` (a copy).

        transformations : Dict[Tuple[str, int], numpy.ndarray]
            For each optimized transformation ``(kind, index)``, its parameters (a copy).

        """
        return self._analysis.unpack_unknowns(self._unknowns, self.p_func(q))

    def reparametrized_apply(self, q: ArrayLike) -> None:
        r"""
        Write a reduced vector :math:`\vec{q}` (e.g. ``result.parameters``) in the analysis:
        ``analysis.apply_unknowns(unknowns, p_func(q))`` (the values not selected are unchanged).
        """
        self._analysis.apply_unknowns(self._unknowns, self.p_func(q))

    # ------------------------------------------------------------------ New parametrizations
    def with_unknowns(self, unknowns: SDICUnknowns) -> SDICFixedParametrization:
        r"""A new parametrization with other unknowns and the same selection (vertices, components)."""
        return SDICFixedParametrization(
            self._analysis, unknowns, vertices=self._vertices, components=self._components, sparse=self._sparse
        )

    def with_options(self, **changes: Any) -> SDICFixedParametrization:
        r"""A new parametrization with other options (``vertices``, ``components``, ``sparse``)."""
        options = {"vertices": self._vertices, "components": self._components, "sparse": self._sparse}
        unknown = sorted(set(changes) - set(options))
        if unknown:
            raise TypeError(f"Unknown options {unknown}. Valid options: {sorted(options)}.")
        return SDICFixedParametrization(self._analysis, self._unknowns, **{**options, **changes})

    def __repr__(self) -> str:
        return f"SDICFixedParametrization({self.n_reparametrized} / {self._analysis.n_unknowns(self._unknowns)} unknowns optimized)"