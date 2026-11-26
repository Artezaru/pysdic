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

# Measure of the shape of the specimen at a reference frame, with the stereo pairings only.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy
import pysolvegn
import scipy.sparse

from ..geometry.integration_points import IntegrationPoints
from .._validation import _validate_bool, _validate_integer
from ..sdic.gl_correction import GrayLevelGroups
from ..sdic.pairing import Pairing
from ..sdic.sdic_analysis import SDICAnalysis
from ..sdic.sdic_parametrization import SDICFixedParametrization, compute_supported_vertices
from ..sdic.sdic_term import SDICCorrelationTerm
from ..sdic.sdic_unknowns import SDICUnknowns


@dataclass(frozen=True)
class ReferenceShapeResult:
    r"""
    The shape measured at a reference frame, returned by :func:`measure_reference_shape`.

    Attributes
    ----------
    frame : int
        The reference frame.

    displacement : numpy.ndarray
        The measured displacement :math:`(N_v, E)` of the frame, from the vertices of the mesh
        to the measured surface. Not written in the analysis.

    normal_displacement : Optional[numpy.ndarray]
        With ``normal_only=True``, the displacement :math:`(N_v,)` of each vertex along its
        normal (0 for the vertices that support no integration point). None otherwise.

    pairings : Tuple[:class:`pysdic.Pairing`, ...]
        The stereo pairings of the term.

    solver_result : Any
        The result of ``pysolvegn.solve`` (in the reduced unknowns of the problem).

    """

    frame: int
    displacement: numpy.ndarray
    normal_displacement: Optional[numpy.ndarray]
    pairings: Tuple[Pairing, ...]
    solver_result: Any


def _normal_term(
    correlation: SDICCorrelationTerm, frame: int, vertices: numpy.ndarray, normals: numpy.ndarray
) -> Tuple[Any, numpy.ndarray, scipy.sparse.csr_matrix]:
    # Term of the correlation reduced to one unknown per vertex: p = p_0 + S q, with S moving the
    # vertex k of the frame along its normal n_k (displacement in Fortran order, row c N_v + k)
    unknowns, analysis = correlation.unknowns, correlation.analysis
    N_v = analysis.mesh.n_vertices
    p0 = analysis.pack_unknowns(unknowns)
    start = analysis.get_unknowns_frame_slice(unknowns, frame).start
    E = normals.shape[1]
    rows = (start + numpy.arange(E)[:, numpy.newaxis] * N_v + vertices[numpy.newaxis, :]).ravel()
    columns = numpy.tile(numpy.arange(vertices.size), E)
    selection = scipy.sparse.csr_matrix((normals.T.ravel(), (rows, columns)), shape=(p0.size, vertices.size))
    base = correlation.term

    def residual_func(q: numpy.ndarray) -> numpy.ndarray:
        return base.residual_func(p0 + selection @ q)

    def jacobian_func(q: numpy.ndarray) -> Any:
        return base.jacobian_func(p0 + selection @ q) @ selection

    term = pysolvegn.Term.from_rJ(residual_func=residual_func, jacobian_func=jacobian_func)
    return term, p0, selection


def measure_reference_shape(
    analysis: SDICAnalysis,
    integration_points: IntegrationPoints,
    *,
    frame: Integral = 0,
    pairings: Optional[Sequence[Pairing]] = None,
    normal_only: bool = True,
    gl_correction: Optional[Mapping[str, Any]] = None,
    gray_level_groups: Optional[GrayLevelGroups] = None,
    **solve_options: Any,
) -> ReferenceShapeResult:
    r"""
    Measure the shape of the specimen at a reference frame: the displacement of the frame
    bringing the mesh (e.g. a CAD mesh, never exactly at the position of the specimen) onto
    the observed surface, with the stereo pairings of the frame only.

    The displacement of the frame (``SDICUnknowns([frame], [])``) is optimized with
    a :class:`pysdic.SDICCorrelationTerm` on the stereo pairings ``(frame, i) - (frame, j)``: the views of the same instant must see
    the same gray levels at each material point. Only the vertices supporting integration
    points are optimized (see :func:`pysdic.compute_supported_vertices`), the other ones keep
    their current displacement (:class:`pysdic.SDICFixedParametrization`).

    **Normal displacement.** With the stereo pairings only, a displacement tangent to the
    surface is not observable: the material point slides on the surface, but both views still
    see the same speckle at its new position, so the residuals do not change (the Hessian is
    singular along these directions). With ``normal_only=True`` (default), each vertex only
    moves along its normal :math:`\vec{n}_k` (normals of the mesh deformed by the current
    displacement), with one unknown :math:`q_k` per vertex:

    .. math::

        \vec{u}_k = \vec{u}_k^0 + q_k \, \vec{n}_k

    With ``normal_only=False``, the three components are optimized: use it only with another
    constraint of the tangential motion.

    The analysis is not modified: write the result with
    ``analysis.set_displacement(frame, result.displacement)``. This measure comes before any
    measure of displacement against this frame.

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with at least two views at ``frame`` and its displacement set (the initial
        guess, e.g. zero).

    integration_points : :class:`pysdic.IntegrationPoints`
        The material points, all located and visible in the views of the frame (see
        :func:`select_visible_integration_points`).

    frame : Integral, optional (default: ``0``)
        The reference frame.

    pairings : Optional[Sequence[:class:`pysdic.Pairing`]], optional
        The stereo pairings of the frame. By default, all the pairs of views ``(i, j)`` with
        :math:`i < j`.

    normal_only : bool, optional (default: ``True``)
        If True, only the normal displacement of the vertices is optimized (3D mesh only).

    gl_correction : Optional[Mapping[str, Any]], optional
        The gray level correction of the term (see :class:`pysdic.SDICCorrelationTerm`).

    gray_level_groups : Optional[:class:`pysdic.GrayLevelGroups`], optional
        The groups of the correction (see :class:`pysdic.SDICCorrelationTerm`).

    **solve_options
        The options of ``pysolvegn.solve`` (``max_iteration``, ``ftol``, ``verbosity``, ...).

    Returns
    -------
    :class:`ReferenceShapeResult`
        The measured displacement of the frame.

    Raises
    ------
    ValueError
        If the frame has less than two views, its displacement is not set, a pairing is not a
        stereo pairing of the frame, or ``normal_only`` is used with a 2D mesh.

    Examples
    --------
    .. code-block:: python

        from pysdic.workflows import measure_reference_shape

        analysis.set_zero_displacement(0)
        shape = measure_reference_shape(analysis, integration_points, gl_correction={"mode": "ZNSSD"},
                                        max_iteration=30)
        analysis.set_displacement(0, shape.displacement)
        print(f"Normal gap: {numpy.abs(shape.normal_displacement).max():.3f} mm")

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    frame = _validate_integer(frame, "frame", min_value=0)
    normal_only = _validate_bool(normal_only, "normal_only")
    mesh = analysis.mesh
    if analysis.n_views(frame) < 2:
        raise ValueError(f"The frame {frame} must have at least two views. Got {analysis.n_views(frame)}.")
    if not analysis.has_displacement(frame):
        raise ValueError(
            f"The displacement of the frame {frame} is not set: set the initial guess before (e.g. "
            f"analysis.set_zero_displacement({frame}))."
        )
    if normal_only and mesh.n_dimensions != 3:
        raise ValueError("Input 'normal_only' requires a 3D mesh (the normal of a 2D mesh is out of its plane).")
    if pairings is None:
        n_views = analysis.n_views(frame)
        pairings = [Pairing.stereo(frame, i, j) for i in range(n_views) for j in range(i + 1, n_views)]
    else:
        pairings = list(pairings)
        wrong = [p for p in pairings if not (p.is_stereo and p.frame_index_1 == frame)]
        if wrong:
            raise ValueError(f"The pairings must be stereo pairings of the frame {frame}. Got {wrong}.")

    unknowns = SDICUnknowns([frame], [])
    correlation = SDICCorrelationTerm(
        analysis,
        unknowns,
        integration_points,
        pairings,
        gl_correction=gl_correction,
        gray_level_groups=gray_level_groups,
        sparse=True,
    )
    vertices = compute_supported_vertices(mesh, integration_points)

    normal_displacement = None
    if normal_only:
        normals = analysis.get_deformed_mesh(frame).compute_vertices_normals()[vertices]  # (n, 3)
        term, p0, selection = _normal_term(correlation, frame, vertices, normals)
        result = pysolvegn.solve(terms=[term], p0=numpy.zeros(vertices.size), **solve_options)
        displacements, _ = analysis.unpack_unknowns(unknowns, p0 + selection @ result.parameters)
        normal_displacement = numpy.zeros(mesh.n_vertices)
        normal_displacement[vertices] = result.parameters
    else:
        parametrization = SDICFixedParametrization(analysis, unknowns, vertices=vertices)
        result = pysolvegn.solve(
            terms=[correlation.term],
            p0=parametrization.reparametrized_pack(),
            parametrization=parametrization.parametrization,
            **solve_options,
        )
        displacements, _ = parametrization.reparametrized_unpack(result.parameters)  # Analysis not modified

    return ReferenceShapeResult(
        frame=frame,
        displacement=displacements[frame],
        normal_displacement=normal_displacement,
        pairings=tuple(pairings),
        solver_result=result,
    )