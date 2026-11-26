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
from typing import Optional, Tuple, Union
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy.sparse
import scipy.sparse.linalg

from .._validation import (
    _validate_bool,
    _validate_choice,
    _validate_real,
    _as_array,
    _as_scalar_or_1d_array,
    _check_matching_size,
)
from ._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ._validation import (
    _check_finite_rows,
    _validate_connectivity,
    _validate_element_indices,
    _validate_element_type,
)
from .shape_functions import (
    _shape_functions_impl,
    _validate_natural_coordinates,
)


# Relative tolerance used by ``scipy.sparse.linalg.lsqr`` for sparse projections.
_LSQR_TOLERANCE = 1e-12

# Minimum ratio between the smallest and the largest singular values of a Jacobian
# matrix. Below this ratio, the element is considered as degenerated.
_JACOBIAN_RCOND = 1e-10


# =============================================================================
# Internal helpers: points handling
# =============================================================================
def _gather_nodal_values(
    nodal_values: numpy.ndarray,
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Gather the nodal values of the element of each point.

    .. important::

        Internal use only. All the ``element_indices`` must be valid (no ``-1``).


    Parameters
    ----------
    nodal_values: numpy.ndarray
        Values at the nodes with shape :math:`(N_{v}, ...)`.

    connectivity_array: numpy.ndarray
        Connectivity with shape :math:`(N_{e}, N_{vpe})`.

    element_indices: numpy.ndarray
        Element indices of the points with shape :math:`(N_{p},)`.


    Returns
    -------
    numpy.ndarray
        Gathered values with shape :math:`(N_{p}, N_{vpe}, ...)`.

    """
    return nodal_values[connectivity_array[element_indices]]


def _prepare_points_inputs(
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    allow_unlocated_points: bool,
    tolerance: Real,
    n_vertices: Optional[Integral] = None,
) -> Tuple[str, numpy.ndarray, int, numpy.ndarray, float, numpy.ndarray, numpy.ndarray]:
    r"""
    Validate once the inputs shared by the ``compute_*`` functions.

    .. important::

        Internal use only.


    Returns
    -------
    element_type: str
        Validated element type (lower case).

    connectivity_array: numpy.ndarray
        Validated connectivity with shape :math:`(N_{e}, N_{vpe})`.

    n_vertices: int
        Validated (or inferred) number of vertices.

    natural_coordinates: numpy.ndarray
        Validated natural coordinates with shape :math:`(N_{p}, K)`.

    tolerance: float
        Validated tolerance.

    element_indices: numpy.ndarray
        Validated element indices with shape :math:`(N_{p},)`.

    located_mask: numpy.ndarray
        Boolean mask with shape :math:`(N_{p},)` of the points located in an element.

    """
    element_type = _validate_element_type(element_type)
    properties = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[element_type]
    connectivity_array, n_vertices = _validate_connectivity(
        connectivity_array, n_vertices, properties["expected_N_vpe"]
    )
    natural_coordinates = _validate_natural_coordinates(
        natural_coordinates, properties["expected_K"]
    )
    tolerance = _validate_real(tolerance, "tolerance", non_negative=True, finite=True)
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    _check_matching_size(
        natural_coordinates.shape[0], element_indices.shape[0],
        "natural_coordinates", "element_indices", "N_p",
    )
    return (
        element_type, connectivity_array, n_vertices, natural_coordinates, tolerance,
        element_indices, located_mask,
    )


def _check_located_points_inside(
    inside_mask: numpy.ndarray,
    located_mask: numpy.ndarray,
    element_type: str,
    tolerance: float,
) -> None:
    r"""
    Raise a ``ValueError`` if a located point has natural coordinates outside the
    reference element (enlarged by ``tolerance``).

    .. important::

        Internal use only.

    """
    outside = located_mask & ~inside_mask
    if numpy.any(outside):
        first = int(numpy.flatnonzero(outside)[0])
        raise ValueError(
            f"{int(numpy.count_nonzero(outside))} located points have natural "
            f"coordinates outside the reference element of '{element_type}' "
            f"(tolerance={tolerance}), e.g. point {first}. "
            f"Check 'element_indices' and 'natural_coordinates', or increase "
            f"'tolerance' to accept the points on the boundary of their element up to "
            f"rounding errors."
        )


def _assemble_shape_function_matrix_impl(
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
    shape_functions: numpy.ndarray,
    valid_mask: numpy.ndarray,
    n_vertices: int,
    sparse: bool,
    sparse_format: str,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
    r"""
    Assemble the shape function matrix :math:`\mathcal{N}` of shape
    :math:`(N_{p}, N_{v})` without any validation.

    .. important::

        Internal use only. All the inputs are assumed to be validated (distinct
        vertices in each element, finite shape functions for the valid points). The
        rows of the points with ``valid_mask`` False are zeros (no entry in sparse
        matrices).

    """
    N_p, N_vpe = shape_functions.shape
    rows = numpy.flatnonzero(valid_mask)  # (N_valid,)
    columns = connectivity_array[element_indices[rows]]  # (N_valid, N_vpe)
    values = shape_functions[rows]  # (N_valid, N_vpe)

    if not sparse:
        shape_function_matrix = numpy.zeros((N_p, n_vertices), dtype=numpy.float64)
        shape_function_matrix[rows[:, numpy.newaxis], columns] = values
        return shape_function_matrix

    shape_function_matrix = scipy.sparse.coo_matrix(
        (values.ravel(), (numpy.repeat(rows, N_vpe), columns.ravel())),
        shape=(N_p, n_vertices),
    )
    if sparse_format == "csr":
        return shape_function_matrix.tocsr()
    return shape_function_matrix


def _remap_vertices_coordinates_impl(
    vertices_coordinates: numpy.ndarray,
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
    located_mask: numpy.ndarray,
    default: float,
) -> numpy.ndarray:
    r"""
    Remap the vertices coordinates to the points without any validation. The points
    with ``located_mask`` False are filled with ``default``.

    .. important::

        Internal use only.

    """
    N_p = element_indices.shape[0]
    N_vpe = connectivity_array.shape[1]
    E = vertices_coordinates.shape[1]
    remapped_coordinates = numpy.full((N_p, N_vpe, E), default, dtype=numpy.float64)
    remapped_coordinates[located_mask] = _gather_nodal_values(
        vertices_coordinates, connectivity_array, element_indices[located_mask]
    )  # (N_located, N_vpe, E)
    return remapped_coordinates


def _assemble_jacobian_matrix_impl(
    shape_function_derivatives: numpy.ndarray, remapped_coordinates: numpy.ndarray
) -> numpy.ndarray:
    r"""
    Assemble the Jacobian matrices :math:`(N_{p}, E, K)` without any validation.

    .. important::

        Internal use only.

    """
    # J[p, e, k] = sum_j dN_j/dxi_k (p) * x_{j, e}
    return numpy.einsum(
        "pjk,pje->pek", shape_function_derivatives, remapped_coordinates
    )  # (N_p, E, K)


def _assemble_property_derivative_impl(
    nodal_values: numpy.ndarray,
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
    shape_function_derivatives: numpy.ndarray,
    jacobian_matrix: numpy.ndarray,
    located_mask: numpy.ndarray,
    default: float,
) -> numpy.ndarray:
    r"""
    Assemble the derivatives of a property :math:`(N_{p}, P, E)` without any
    validation of the inputs. The points with ``located_mask`` False are set to
    ``default``.

    .. important::

        Internal use only. The shape function derivatives and the Jacobian matrices
        of the located points are assumed to be finite. A ``ValueError`` is raised if
        the Jacobian matrix of a located point is degenerated.

    """
    N_p = shape_function_derivatives.shape[0]
    E = jacobian_matrix.shape[1]
    P = nodal_values.shape[1]
    gradients = numpy.full((N_p, P, E), default, dtype=numpy.float64)
    indices = numpy.flatnonzero(located_mask)
    if indices.size == 0:
        return gradients

    # A single SVD gives both the conditioning and the pseudo-inverse J^+ = V S^-1 U^T
    u, s, vt = numpy.linalg.svd(jacobian_matrix[indices], full_matrices=False)
    # u: (N_located, E, K), s: (N_located, K), vt: (N_located, K, K)
    degenerated = ~(s[:, -1] > _JACOBIAN_RCOND * s[:, 0])  # Also True if s_max = 0
    if numpy.any(degenerated):
        raise ValueError(
            f"{int(numpy.count_nonzero(degenerated))} located points have a "
            f"degenerated Jacobian matrix (ratio of the smallest to the largest "
            f"singular value below {_JACOBIAN_RCOND}), e.g. point "
            f"{int(indices[numpy.flatnonzero(degenerated)[0]])}. Check the mesh for "
            f"flat elements."
        )
    jacobians_pinv = numpy.einsum(
        "nlk,nel->nke", vt, u / s[:, numpy.newaxis, :]
    )  # (N_located, K, E): J^+[k, e] = sum_l V[k, l] / s_l * U[e, l]

    # dP/dxi_k = sum_j dN_j/dxi_k * P_j
    values = _gather_nodal_values(
        nodal_values, connectivity_array, element_indices[indices]
    )  # (N_located, N_vpe, P)
    natural_derivatives = numpy.einsum(
        "njp,njk->npk", values, shape_function_derivatives[indices]
    )  # (N_located, P, K)

    # dP/dx_e = sum_k dP/dxi_k * J^+[k, e]
    gradients[indices] = numpy.einsum(
        "npk,nke->npe", natural_derivatives, jacobians_pinv
    )  # (N_located, P, E)
    return gradients


def _assemble_property_interpolation_impl(
    nodal_values: numpy.ndarray,
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
    shape_functions: numpy.ndarray,
    located_mask: numpy.ndarray,
    default: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Interpolate a property :math:`(N_{p}, P)` without any validation. The points with
    ``located_mask`` False are set to ``default`` (array of shape :math:`(P,)`).

    .. important::

        Internal use only.

    """
    N_p = shape_functions.shape[0]
    indices = numpy.flatnonzero(located_mask)

    # P(xi) = sum_j N_j(xi) P_j
    values = _gather_nodal_values(
        nodal_values, connectivity_array, element_indices[indices]
    )  # (N_located, N_vpe, P)
    interpolated_values = numpy.tile(default, (N_p, 1))  # (N_p, P)
    interpolated_values[indices] = numpy.einsum(
        "nj,njp->np", shape_functions[indices], values
    )  # (N_located, P)
    return interpolated_values


def _validate_projection_inputs(
    point_values: ArrayLike,
    point_weights: Optional[ArrayLike],
    n_points: int,
    default: Real,
    return_unaffected: bool,
) -> Tuple[numpy.ndarray, numpy.ndarray, float, bool]:
    r"""
    Validate the inputs of the property projection which do not depend on the shape
    function matrix.

    .. important::

        Internal use only.

    """
    point_values = _as_array(
        point_values, "point_values", numpy.float64,
        shape=(None, None), shape_hint="(N_p, P)", accept_1d_as_column=True, finite=False,
    )
    _check_matching_size(
        point_values.shape[0], n_points,
        "point_values", "shape_function_matrix", "N_p",
    )
    if point_weights is None:
        point_weights = numpy.ones((n_points,), dtype=numpy.float64)
    else:
        point_weights = _as_array(
            point_weights, "point_weights", numpy.float64,
            shape=(None,), shape_hint="(N_p,)", finite=True,
        )
        _check_matching_size(
            point_weights.shape[0], n_points, "point_weights", "point_values", "N_p"
        )
        if numpy.any(point_weights < 0):
            raise ValueError("Input 'point_weights' must be non-negative.")
    default = _validate_real(default, "default")
    return_unaffected = _validate_bool(return_unaffected, "return_unaffected")
    return point_values, point_weights, default, return_unaffected


def _assemble_property_projection_impl(
    point_values: numpy.ndarray,
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix],
    point_weights: numpy.ndarray,
    default: float,
    return_unaffected: bool,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property to the nodes without any validation.

    .. important::

        Internal use only. ``shape_function_matrix`` is a ``numpy.float64`` array or a
        ``scipy.sparse.csr_matrix``.

    """
    is_sparse = scipy.sparse.issparse(shape_function_matrix)
    N_p, N_v = shape_function_matrix.shape
    P = point_values.shape[1]

    # Kept points: positive weight, finite values and finite shape functions
    keep_mask = (point_weights > 0) & numpy.all(numpy.isfinite(point_values), axis=1)
    if is_sparse:
        coo = shape_function_matrix.tocoo()
        keep_mask[coo.row[~numpy.isfinite(coo.data)]] = False
    else:
        keep_mask &= numpy.all(numpy.isfinite(shape_function_matrix), axis=1)
    rows = numpy.flatnonzero(keep_mask)
    matrix = shape_function_matrix[rows]  # (N_kept, N_v)

    # Affected nodes: at least one non-zero shape function among the kept points
    if is_sparse:
        coo = matrix.tocoo()
        affected_mask = numpy.zeros((N_v,), dtype=bool)
        affected_mask[coo.col[coo.data != 0]] = True
    else:
        affected_mask = numpy.any(matrix != 0, axis=0)
    columns = numpy.flatnonzero(affected_mask)

    # Weighted least squares on the reduced system: W^1/2 N x = W^1/2 p
    projected_values = numpy.full((N_v, P), default, dtype=numpy.float64)
    if columns.size > 0:
        sqrt_weights = numpy.sqrt(point_weights[rows])  # (N_kept,)
        rhs = sqrt_weights[:, numpy.newaxis] * point_values[rows]  # (N_kept, P)

        if is_sparse:
            lhs = scipy.sparse.diags(sqrt_weights) @ matrix[:, columns]
            lhs = scipy.sparse.csr_matrix(lhs)  # (N_kept, N_affected)
            for p in range(P):
                projected_values[columns, p] = scipy.sparse.linalg.lsqr(
                    lhs, rhs[:, p], atol=_LSQR_TOLERANCE, btol=_LSQR_TOLERANCE
                )[0]
        else:
            lhs = sqrt_weights[:, numpy.newaxis] * matrix[:, columns]
            projected_values[columns] = numpy.linalg.lstsq(lhs, rhs, rcond=None)[0]

    if return_unaffected:
        return projected_values, ~affected_mask
    return projected_values


# =============================================================================
# Shape function matrix
# =============================================================================
def assemble_shape_function_matrix(
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    shape_functions: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
    sparse: bool = False,
    sparse_format: str = "csr",
    allow_unlocated_points: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
    r"""
    Assemble the shape function matrix :math:`\mathcal{N}` at given integration
    points within elements with shape :math:`(N_{p}, N_{v})`, from precomputed shape
    functions values with zero-filling for non-associated nodes/vertices.

    ``shape_function_matrix[i, j]`` contains the shape function value at integration
    point :math:`i` for vertex :math:`j` and a non-zero value only if vertex
    :math:`j` is part of the element containing integration point :math:`i`.

    .. note::

        - Input ``shape_functions`` will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be
          converted to ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The row of a point with element index ``-1`` (only accepted with
        ``allow_unlocated_points=True``) is filled with zeros (no entry in sparse
        matrices), i.e. the point has a zero weight. A non-finite shape function value
        of a located point raises a ``ValueError``.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    shape_functions: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum node index in
        ``connectivity_array`` plus one.

    sparse: bool, optional (default: ``False``)
        If set to True, the function will use ``scipy.sparse`` to create a
        sparse matrix representation of the shape function matrix.

    sparse_format: str, optional (default: ``'csr'``)
        The format of the sparse matrix to create when ``sparse`` is True.
        Supported formats are ``'csr'`` for Compressed Sparse Row and ``'coo'`` for
        Coordinate format. Ignored if ``sparse`` is False.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.


    Returns
    -------
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]
        An array of shape :math:`(N_{p}, N_{v})` or sparse matrix containing the shape
        function values at each of the :math:`N_{p}` integration points for all
        :math:`N_{v}` nodes in the mesh.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements
        with shape :math:`(N_{p}, N_{vpe})`.

    pysdic.fem.compute_shape_function_matrix:
        To compute the shape function matrix directly from natural coordinates
        and mesh connectivity.


    Examples
    --------
    Lets construct a simple 2D mesh and build the shape function matrix
    at given integration points for triangular elements. The last point is outside
    the mesh (element index ``-1``).

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import assemble_shape_function_matrix
        from pysdic.fem import compute_triangle_3_shape_functions

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2],
             [0.0, 0.0]]
        )

        element_indices = numpy.array([0, 1, -1])

        shape_functions = compute_triangle_3_shape_functions(natural_coordinates)

        shape_func_matrix = assemble_shape_function_matrix(
            connectivity_array=connectivity_array,
            element_indices=element_indices,
            shape_functions=shape_functions,
            allow_unlocated_points=True,  # Accept the -1 of the last point
        )

        print(f"shape function matrix (shape={shape_func_matrix.shape}):")
        print(shape_func_matrix)

    .. code-block:: console

        shape function matrix (shape=(3, 4)):
        [[0.5 0.2 0.3 0. ]
         [0.2 0.  0.6 0.2]
         [0.  0.  0.  0. ]]

    """
    # Input validation and conversion
    shape_functions = _as_array(
        shape_functions, "shape_functions", numpy.float64,
        shape=(None, None), shape_hint="(N_p, N_vpe)", finite=False,
    )  # Finite values checked for the located points only
    connectivity_array, n_vertices = _validate_connectivity(
        connectivity_array, n_vertices
    )
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    _check_matching_size(
        shape_functions.shape[0], element_indices.shape[0],
        "shape_functions", "element_indices", "N_p",
    )
    _check_matching_size(
        shape_functions.shape[1], connectivity_array.shape[1],
        "shape_functions", "connectivity_array", "N_vpe",
    )
    sparse = _validate_bool(sparse, "sparse")
    sparse_format = _validate_choice(sparse_format, "sparse_format", ("csr", "coo"))
    _check_finite_rows(shape_functions, located_mask, "shape_functions")

    return _assemble_shape_function_matrix_impl(
        connectivity_array, element_indices, shape_functions, located_mask,
        n_vertices, sparse, sparse_format,
    )


def compute_shape_function_matrix(
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
    sparse: bool = False,
    sparse_format: str = "csr",
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
    r"""
    Compute and assemble the shape function matrix :math:`\mathcal{N}` at given
    integration points within specified elements with shape :math:`(N_{p}, N_{v})`.

    This function combines the computation of shape functions at given natural
    coordinates with the assembly of the shape function matrix for the entire mesh.

    ``shape_function_matrix[i, j]`` contains the shape function value at integration
    point :math:`i` for vertex :math:`j` and a non-zero value only if vertex
    :math:`j` is part of the element containing integration point :math:`i`.

    .. note::

        - Input ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be
          converted to ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The row of a point with element index ``-1`` (only accepted with
        ``allow_unlocated_points=True``) or with natural coordinates outside the
        reference element (enlarged by ``tolerance``) is filled with zeros (no entry in
        sparse matrices), i.e. the point has a zero weight.


    Parameters
    ----------
    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of
        the elements in the mesh, where each row contains the indices of the nodes
        that form an element.

    element_type: str
        A string specifying the type of element (e.g., ``'segment_2'``,
        ``'triangle_3'``, etc.) to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum node index in
        ``connectivity_array`` plus one.

    sparse: bool, optional (default: ``False``)
        If set to True, the function will use ``scipy.sparse`` to create a
        sparse matrix representation of the shape function matrix.

    sparse_format: str, optional (default: ``'csr'``)
        The format of the sparse matrix to create when ``sparse`` is True.
        Supported formats are ``'csr'`` for Compressed Sparse Row and ``'coo'`` for
        Coordinate format. Ignored if ``sparse`` is False.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the reference element when checking whether
        the natural coordinates of a point are inside its element (see
        :func:`pysdic.fem.compute_shape_functions`). Useful for points located on the
        boundary of their element up to rounding errors. With the default ``0.0``, only
        the points exactly inside the reference element are accepted.


    Returns
    -------
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]
        An array of shape :math:`(N_{p}, N_{v})` or sparse matrix containing the shape
        function values at each of the :math:`N_{p}` integration points for all
        :math:`N_{v}` nodes in the mesh.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements
        with shape :math:`(N_{p}, N_{vpe})`.

    pysdic.fem.assemble_shape_function_matrix:
        To assemble the shape function matrix from precomputed shape function values.


    Examples
    --------
    Lets construct a simple 2D mesh and build the shape function matrix at given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_shape_function_matrix

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        shape_func_matrix = compute_shape_function_matrix(
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        print(f"shape function matrix (shape={shape_func_matrix.shape}):")
        print(shape_func_matrix)

    .. code-block:: console

        shape function matrix (shape=(2, 4)):
        [[0.5 0.2 0.3 0. ]
         [0.2 0.  0.6 0.2]]

    """
    # Input validation and conversion (once for the whole computation)
    (
        element_type, connectivity_array, n_vertices, natural_coordinates, tolerance,
        element_indices, located_mask,
    ) = _prepare_points_inputs(
        connectivity_array, element_type, natural_coordinates, element_indices,
        allow_unlocated_points, tolerance, n_vertices,
    )
    sparse = _validate_bool(sparse, "sparse")
    sparse_format = _validate_choice(sparse_format, "sparse_format", ("csr", "coo"))

    # Points outside their element get a zero weight (empty row)
    shape_functions, _, inside_mask = _shape_functions_impl(
        natural_coordinates, element_type, False, tolerance
    )
    return _assemble_shape_function_matrix_impl(
        connectivity_array, element_indices, shape_functions,
        located_mask & inside_mask, n_vertices, sparse, sparse_format,
    )


# =============================================================================
# Vertices remapping and Jacobian
# =============================================================================
def remap_vertices_coordinates(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Remap the global coordinates of the vertices to given integration points within
    elements based on the element connectivity, and the remapped coordinates
    will have shape :math:`(N_{p}, N_{vpe}, E)`.

    ``remapped_coordinates[i]`` contains the global coordinates of the vertices for
    the element containing integration point :math:`i`.

    .. note::

        - Input ``vertices_coordinates`` will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        With ``allow_unlocated_points=True``, the points with element index ``-1`` are
        filled with ``default``. With ``allow_unlocated_points=False`` (default),
        negative element indices raise an ``IndexError``.


    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates of the
        vertices in the mesh.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    default: Real, optional (default: ``numpy.nan``)
        The default value to assign to remapped coordinates for integration points
        associated with an element index of -1 when ``allow_unlocated_points`` is True.


    Returns
    -------
    remapped_coordinates: numpy.ndarray
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global
        coordinates of the vertices for each integration point within the elements.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes. The mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the
    remapped coordinates array has shape :math:`(N_{p}, N_{vpe}, E)` where each entry
    corresponds to the global coordinates of the nodes associated with the element
    containing the integration point.


    See Also
    --------
    pysdic.fem.assemble_jacobian_matrix:
        To compute the Jacobian matrices for the transformation between natural and
        global coordinates.


    Examples
    --------
    Lets construct a simple 2D mesh and remap the vertex coordinates to 3 integration
    points (the last one is outside the mesh).

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import remap_vertices_coordinates

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        element_indices = numpy.array([0, 1, -1])

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity_array=connectivity_array,
            element_indices=element_indices,
            allow_unlocated_points=True,  # Accept the -1 of the last point
        )

        print(f"remapped coordinates (shape={remapped_coords.shape}):")
        print(remapped_coords)

    .. code-block:: console

        remapped coordinates (shape=(3, 3, 2)):
        [[[ 0.  0.]
          [ 1.  0.]
          [ 1.  1.]]

         [[ 0.  0.]
          [ 1.  1.]
          [ 0.  1.]]

         [[nan nan]
          [nan nan]
          [nan nan]]]

    """
    # Input validation and conversion
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, None), shape_hint="(N_v, E)", finite=False,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, n_vertices=vertices_coordinates.shape[0]
    )
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    default = _validate_real(default, "default")

    return _remap_vertices_coordinates_impl(
        vertices_coordinates, connectivity_array, element_indices, located_mask, default
    )


def assemble_jacobian_matrix(
    shape_function_derivatives: ArrayLike,
    remapped_coordinates: ArrayLike,
) -> numpy.ndarray:
    r"""
    Assemble the Jacobian (derivative) matrix :math:`J` of the transformation from
    natural coordinates :math:`(\xi, \eta, \zeta, ...)` to global coordinates
    :math:`(x, y, z, ...)` with shape :math:`(N_{p}, E, K)` from precomputed shape
    function derivatives and remapped vertex coordinates.

    ``jacobians[i]`` contains the Jacobian matrix for the integration point :math:`i`.

    .. note::

        - Inputs ``shape_function_derivatives`` and ``remapped_coordinates`` will be
          converted to ``numpy.float64``.

    .. warning::

        No tests are performed to check if the inputs array are consistent.
        Only shapes are validated. Non-finite inputs (e.g. NaN for invalid points)
        propagate to the corresponding Jacobian matrices.


    Parameters
    ----------
    shape_function_derivatives: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of
        the shape functions with respect to the local coordinates,
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    remapped_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global
        coordinates of the vertices for each integration point within the elements.


    Returns
    -------
    jacobians: numpy.ndarray
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions
        :math:`(E \times K)`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    The Jacobian matrix has dimensions :math:`(E \times K)` and constructed as follows:

    .. math::

        J = J_{X/\Xi} = \frac{dX}{d\Xi} = \begin{bmatrix}
        \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} & \frac{\partial x}{\partial \zeta} & \ldots \\
        \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} & \frac{\partial y}{\partial \zeta} & \ldots \\
        \frac{\partial z}{\partial \xi} & \frac{\partial z}{\partial \eta} & \frac{\partial z}{\partial \zeta} & \ldots \\
        \vdots & \vdots & \vdots & \ddots
        \end{bmatrix}

    where each entry is computed as:

    .. math::

        \frac{\partial x_i}{\partial \xi_j} = \sum_{a=1}^{N_{vpe}} \frac{\partial N_a}{\partial \xi_j} x_{i,a}

    Here, :math:`N_a` are the shape functions associated with each node, and
    :math:`x_{i,a}` are the global coordinates of the nodes.

    .. math::

        \nabla_{\Xi} N = J^{T} \nabla_{X} N = J_{X/\Xi}^{T} \nabla_{X} N

    To obtain the jacobian matrix for the transformation from global coordinates to
    natural coordinates noted :math:`J_{\Xi/X}` such as:

    .. math::

        \nabla_{X} N = J_{\Xi/X}^{T} \nabla_{\Xi} N

    we get :

    .. math::

        \nabla_{X} N = (J^{T})^{-1} \nabla_{\Xi} N = (J^{-1})^{T} \nabla_{\Xi} N

    So the jacobian matrix for the transformation from global coordinates to natural
    coordinates is the inverse of the jacobian matrix for the transformation from
    natural coordinates to global coordinates.

    .. math::

        J_{\Xi/X} = J^{-1} = J_{X/\Xi}^{-1}

    For :math:`K < E`, the Jacobian matrix is not square and thus not invertible.
    In this case, the pseudo-inverse can be used to compute the transformation from
    global coordinates to natural coordinates. The Moore-Penrose pseudo-inverse of the
    Jacobian matrix :math:`J \in \mathbb{R}^{E \times K}` is given by
    :math:`J^{+} = (J^{T} J)^{-1} J^{T}` because :math:`J` has full column rank
    (i.e., the columns of :math:`J` are linearly independent).


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.fem.remap_vertices_coordinates:
        To remap the vertices coordinates for each element at given integration
        points.

    pysdic.fem.compute_jacobian_matrix:
        To compute the Jacobian matrices directly from natural coordinates and mesh
        connectivity.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the Jacobian at given integration points
    for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            remap_vertices_coordinates,
            assemble_jacobian_matrix,
            compute_triangle_3_shape_functions
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity_array=connectivity_array,
            element_indices=element_indices
        )

        _, shape_func_derivs = compute_triangle_3_shape_functions(
            natural_coordinates=natural_coordinates,
            return_derivatives=True
        )

        jacobians = assemble_jacobian_matrix(
            shape_function_derivatives=shape_func_derivs,
            remapped_coordinates=remapped_coords
        )

        print(f"jacobians (shape={jacobians.shape}):")
        print(jacobians)

    .. code-block:: console

        jacobians (shape=(2, 2, 2)):
        [[[1. 1.]
          [0. 1.]]

         [[1. 0.]
          [1. 1.]]]

    """
    # Input validation and conversion
    shape_function_derivatives = _as_array(
        shape_function_derivatives, "shape_function_derivatives", numpy.float64,
        shape=(None, None, None), shape_hint="(N_p, N_vpe, K)", finite=False,
    )
    remapped_coordinates = _as_array(
        remapped_coordinates, "remapped_coordinates", numpy.float64,
        shape=(None, None, None), shape_hint="(N_p, N_vpe, E)", finite=False,
    )
    _check_matching_size(
        shape_function_derivatives.shape[0], remapped_coordinates.shape[0],
        "shape_function_derivatives", "remapped_coordinates", "N_p",
    )
    _check_matching_size(
        shape_function_derivatives.shape[1], remapped_coordinates.shape[1],
        "shape_function_derivatives", "remapped_coordinates", "N_vpe",
    )

    return _assemble_jacobian_matrix_impl(
        shape_function_derivatives, remapped_coordinates
    )  # (N_p, E, K)


def compute_jacobian_matrix(
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Compute and assemble the Jacobian (derivative) matrix :math:`J` of the
    transformation from natural coordinates :math:`(\xi, \eta, \zeta, ...)` to
    global coordinates :math:`(x, y, z, ...)` with shape :math:`(N_{p}, E, K)` at
    given integration points within specified elements.

    ``jacobians[i]`` contains the Jacobian matrix for the integration point :math:`i`.

    This function combines the remapping of vertex coordinates, computation of shape
    function derivatives, and assembly of the Jacobian matrices for the entire mesh.

    .. note::

        - Inputs ``vertices_coordinates`` and ``natural_coordinates`` will be
          converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The Jacobian matrices of the points with element index ``-1`` (only accepted
        with ``allow_unlocated_points=True``) are filled with ``default``. A located
        point whose natural coordinates are outside the reference element (enlarged by
        ``tolerance``) raises a ``ValueError``.


    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates of the
        vertices in the mesh.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: str
        A string specifying the type of element (e.g., ``'segment_2'``,
        ``'triangle_3'``, etc.) to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the reference element when checking whether
        the natural coordinates of a point are inside its element (see
        :func:`pysdic.fem.compute_shape_functions`). Useful for points located on the
        boundary of their element up to rounding errors. With the default ``0.0``, only
        the points exactly inside the reference element are accepted.

    default: Real, optional (default: ``numpy.nan``)
        The value assigned to the Jacobian matrices of the points with element index
        ``-1``.


    Returns
    -------
    jacobians: numpy.ndarray
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions
        :math:`(E \times K)`.


    Notes
    -----
    See :func:`pysdic.fem.assemble_jacobian_matrix` for the definition of the Jacobian
    matrix :math:`J` of shape :math:`(E \times K)`, the transformation between natural
    and global coordinates and the use of the pseudo-inverse for non-square Jacobian
    matrices (:math:`K < E`).


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.fem.remap_vertices_coordinates:
        To remap the vertices coordinates for each element at given integration points.

    pysdic.fem.assemble_jacobian_matrix:
        To assemble the Jacobian matrices from shape function derivatives and remapped
        coordinates.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the Jacobian at given integration points
    for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_jacobian_matrix

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        jacobians = compute_jacobian_matrix(
            vertices_coordinates=vertices_coordinates,
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
        )

        print(f"jacobians (shape={jacobians.shape}):")
        print(jacobians)

    .. code-block:: console

        jacobians (shape=(2, 2, 2)):
        [[[1. 1.]
          [0. 1.]]

         [[1. 0.]
          [1. 1.]]]

    """
    # Input validation and conversion (once for the whole computation)
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, None), shape_hint="(N_v, E)", finite=False,
    )
    (
        element_type, connectivity_array, _, natural_coordinates, tolerance,
        element_indices, located_mask,
    ) = _prepare_points_inputs(
        connectivity_array, element_type, natural_coordinates, element_indices,
        allow_unlocated_points, tolerance, vertices_coordinates.shape[0],
    )
    default = _validate_real(default, "default")

    _, shape_function_derivatives, inside_mask = _shape_functions_impl(
        natural_coordinates, element_type, True, tolerance
    )  # (N_p, N_vpe, K)
    _check_located_points_inside(inside_mask, located_mask, element_type, tolerance)

    N_p, _, K = shape_function_derivatives.shape
    E = vertices_coordinates.shape[1]
    jacobians = numpy.full((N_p, E, K), default, dtype=numpy.float64)
    indices = numpy.flatnonzero(located_mask)
    remapped_coordinates = _gather_nodal_values(
        vertices_coordinates, connectivity_array, element_indices[indices]
    )  # (N_located, N_vpe, E)
    jacobians[indices] = _assemble_jacobian_matrix_impl(
        shape_function_derivatives[indices], remapped_coordinates
    )  # (N_located, E, K)
    return jacobians


# =============================================================================
# Property derivative
# =============================================================================
def assemble_property_derivative(
    nodal_values: ArrayLike,
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    shape_function_derivatives: ArrayLike,
    jacobian_matrix: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Assemble the derivative of a property defined at the nodes of a mesh to given
    integration points within elements with respect to global coordinates
    :math:`(x,y,z,...)` from precomputed shape function derivatives and Jacobian
    matrices.

    ``gradients[i]`` contains the derivative of the property at the integration
    point :math:`i`.

    .. note::

        - Inputs ``nodal_values``, ``shape_function_derivatives`` and
          ``jacobian_matrix`` will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted
          to ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The derivatives of the points with element index ``-1`` (only accepted with
        ``allow_unlocated_points=True``) are set to ``default``. A ``ValueError`` is
        raised if, for a located point:

        - its shape function derivatives or its Jacobian matrix are not finite,
        - its Jacobian matrix is degenerated, i.e. the ratio of its smallest to its
          largest singular value is below ``1e-10`` (e.g. flat element).


    Parameters
    ----------
    nodal_values: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    shape_function_derivatives: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of
        the shape functions evaluated at :math:`N_{p}` points for the :math:`N_{vpe}`
        nodes of the element.

    jacobian_matrix: ArrayLike
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions
        :math:`(E \times K)`.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    default: Real, optional (default: ``numpy.nan``)
        The value assigned to the derivatives of the points with element index ``-1``.


    Returns
    -------
    gradients: numpy.ndarray
        An array of shape :math:`(N_{p}, P, E)` containing the derivatives of the
        property at each of the :math:`N_{p}` integration points with respect to the
        global coordinates :math:`(x, y, z, ...)`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_p` integration points located within elements, the
    gradients array has shape :math:`(N_p, P, E)` where :math:`P` is the number
    of property components (e.g., 1 for scalar properties, 3 for vector properties),
    and :math:`E` is the dimension of the space.

    The derivative of the property at each integration point is computed as:

    If the dimension of the elements :math:`K` is equal to the space dimension
    :math:`E`:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J^{-T} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\xi, \eta, \zeta, ...) P_i

    If the dimension of the elements :math:`K` is less than the space dimension
    :math:`E`:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, :math:`P_i` are
    the property values at the nodes of the element, and :math:`J` is the Jacobian
    matrix of the transformation from natural to global coordinates.

    Both cases are computed with the Moore-Penrose pseudo-inverse :math:`J^{+}`
    (:math:`(J^{+})^{T} = J^{-T}` when :math:`J` is square and invertible), obtained
    from the singular value decomposition of :math:`J`.
    For :math:`K < E`, the result is the tangential gradient: its component normal to
    the element is zero.


    Demonstration
    --------------
    The property field is defined at the nodes of the mesh and can be interpolated to
    any point within the elements using shape functions.

    .. math::

        P(\Xi) = \sum_{i=1}^{N_{vpe}} N_i(\Xi) P_i

    The spatial derivative of the property field with respect to global coordinates is
    defined as:

    .. math::

        \nabla_X P(\Xi) = \sum_{i=1}^{N_{vpe}} \nabla_{X} N_i(\Xi) P_i

    The derivative of the shape functions with respect to global coordinates is obtained
    via the chain rule:

    .. math::

        \nabla_{X} N = J_{\Xi/X}^T \nabla_{\Xi} N

    where :math:`J_{\Xi/X}` is the Jacobian matrix of the transformation from global
    coordinates to natural coordinates, which is the inverse of the Jacobian matrix
    :math:`J = J_{X/\Xi}` of the transformation from natural coordinates to global
    coordinates.

    For :math:`K = E`, the Jacobian matrix is square and invertible :

    .. math::

        J_{\Xi/X} = J^{-1} \rightarrow \nabla_{X} N = J^{-T} \nabla_{\Xi} N

    And we obtain the formula as mentioned above:

    .. math::

        \nabla_X P(\Xi) = J^{-T} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\Xi) P_i

    For :math:`K < E`, the Jacobian matrix is not square and thus not invertible.
    The pseudo-inverse is :

    .. math::

        J_{\Xi/X} = (J^T J)^{-1} J^T \rightarrow \nabla_{X} N = J (J^T J)^{-1} \nabla_{\Xi} N

    And we obtain the formula as mentioned above:

    .. math::

        \nabla_X P(\Xi) = J (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\Xi) P_i


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.fem.assemble_jacobian_matrix:
        To assemble the Jacobian matrices from shape function derivatives and
        remapped coordinates.

    pysdic.fem.compute_property_derivative:
        To compute the property derivatives directly from mesh and integration point
        information.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the derivative of a scalar property at
    given integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            remap_vertices_coordinates,
            assemble_jacobian_matrix,
            assemble_property_derivative,
            compute_triangle_3_shape_functions
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        nodal_values = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity_array=connectivity_array,
            element_indices=element_indices
        )

        _, shape_func_derivs = compute_triangle_3_shape_functions(
            natural_coordinates=natural_coordinates,
            return_derivatives=True
        )

        jacobians = assemble_jacobian_matrix(
            shape_function_derivatives=shape_func_derivs,
            remapped_coordinates=remapped_coords
        )

        gradients = assemble_property_derivative(
            nodal_values=nodal_values,
            connectivity_array=connectivity_array,
            element_indices=element_indices,
            shape_function_derivatives=shape_func_derivs,
            jacobian_matrix=jacobians
        )

        print(f"gradients (shape={gradients.shape}):")
        print(gradients)

    .. code-block:: console

        gradients (shape=(2, 1, 2)):
        [[[ 10.  10.]]

         [[-10.  30.]]]

    """
    # Input validation and conversion
    nodal_values = _as_array(
        nodal_values, "nodal_values", numpy.float64,
        shape=(None, None), shape_hint="(N_v, P)", accept_1d_as_column=True, finite=False,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, n_vertices=nodal_values.shape[0]
    )
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    shape_function_derivatives = _as_array(
        shape_function_derivatives, "shape_function_derivatives", numpy.float64,
        shape=(None, None, None), shape_hint="(N_p, N_vpe, K)", finite=False,
    )
    jacobian_matrix = _as_array(
        jacobian_matrix, "jacobian_matrix", numpy.float64,
        shape=(None, None, None), shape_hint="(N_p, E, K)", finite=False,
    )
    _check_matching_size(
        element_indices.shape[0], shape_function_derivatives.shape[0],
        "element_indices", "shape_function_derivatives", "N_p",
    )
    _check_matching_size(
        jacobian_matrix.shape[0], shape_function_derivatives.shape[0],
        "jacobian_matrix", "shape_function_derivatives", "N_p",
    )
    _check_matching_size(
        connectivity_array.shape[1], shape_function_derivatives.shape[1],
        "connectivity_array", "shape_function_derivatives", "N_vpe",
    )
    _check_matching_size(
        jacobian_matrix.shape[2], shape_function_derivatives.shape[2],
        "jacobian_matrix", "shape_function_derivatives", "K",
    )
    K, E = jacobian_matrix.shape[2], jacobian_matrix.shape[1]
    if K > E:
        raise ValueError(
            f"The topological dimension K ({K}) of the elements cannot be greater than "
            f"the dimension E ({E}) of the space."
        )
    _check_finite_rows(
        shape_function_derivatives, located_mask, "shape_function_derivatives"
    )
    _check_finite_rows(jacobian_matrix, located_mask, "jacobian_matrix")
    default = _validate_real(default, "default")

    return _assemble_property_derivative_impl(
        nodal_values, connectivity_array, element_indices,
        shape_function_derivatives, jacobian_matrix, located_mask, default,
    )


def compute_property_derivative(
    nodal_values: ArrayLike,
    vertices_coordinates: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Compute and assemble the derivative of a property defined at the nodes of a mesh to
    given integration points within elements with respect to global coordinates
    :math:`(x,y,z,...)` from mesh and integration point information.

    ``gradients[i]`` contains the derivative of the property at the integration point
    :math:`i`.

    This function combines the remapping of vertex coordinates, computation of shape
    function derivatives, assembly of the Jacobian matrices, and assembly of the
    property derivatives for the entire mesh.

    .. note::

        - Inputs ``nodal_values``, ``vertices_coordinates`` and
          ``natural_coordinates`` will be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be
          converted to ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The derivatives of the points with element index ``-1`` (only accepted with
        ``allow_unlocated_points=True``) are set to ``default``. A ``ValueError`` is
        raised if a located point has natural coordinates outside the reference element
        (enlarged by ``tolerance``) or a degenerated Jacobian matrix (see
        :func:`pysdic.fem.assemble_property_derivative`).


    Parameters
    ----------
    nodal_values: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates
        :math:`(x, y, z, ...)` of the vertices in the mesh.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: str
        A string specifying the type of element (e.g., ``'segment_2'``,
        ``'triangle_3'``, etc.) to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the reference element when checking whether
        the natural coordinates of a point are inside its element (see
        :func:`pysdic.fem.compute_shape_functions`). Useful for points located on the
        boundary of their element up to rounding errors. With the default ``0.0``, only
        the points exactly inside the reference element are accepted.

    default: Real, optional (default: ``numpy.nan``)
        The value assigned to the derivatives of the points with element index ``-1``.


    Returns
    -------
    gradients: numpy.ndarray
        An array of shape :math:`(N_{p}, P, E)` containing the derivatives of the
        property at each of the :math:`N_{p}` integration points with respect to the
        global coordinates :math:`(x, y, z, ...)`.


    Notes
    -----
    See :func:`pysdic.fem.assemble_property_derivative` for the computation of the
    derivatives with respect to the global coordinates (inverse or pseudo-inverse of the
    Jacobian matrix for :math:`K = E` or :math:`K < E`) and its demonstration.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.fem.compute_jacobian_matrix:
        To compute the Jacobian matrices from mesh and integration point information.

    pysdic.fem.assemble_property_derivative:
        To assemble the property derivatives from precomputed shape function derivatives
        and Jacobian matrices.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the derivative of a scalar property at
    given integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_property_derivative

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        nodal_values = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        gradients = compute_property_derivative(
            nodal_values=nodal_values,
            vertices_coordinates=vertices_coordinates,
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        print(f"gradients (shape={gradients.shape}):")
        print(gradients)

    .. code-block:: console

        gradients (shape=(2, 1, 2)):
        [[[ 10.  10.]]

         [[-10.  30.]]]

    """
    # Input validation and conversion (once for the whole computation)
    nodal_values = _as_array(
        nodal_values, "nodal_values", numpy.float64,
        shape=(None, None), shape_hint="(N_v, P)", accept_1d_as_column=True, finite=False,
    )
    vertices_coordinates = _as_array(
        vertices_coordinates, "vertices_coordinates", numpy.float64,
        shape=(None, None), shape_hint="(N_v, E)", finite=False,
    )
    _check_matching_size(
        nodal_values.shape[0], vertices_coordinates.shape[0],
        "nodal_values", "vertices_coordinates", "N_v",
    )
    (
        element_type, connectivity_array, _, natural_coordinates, tolerance,
        element_indices, located_mask,
    ) = _prepare_points_inputs(
        connectivity_array, element_type, natural_coordinates, element_indices,
        allow_unlocated_points, tolerance, vertices_coordinates.shape[0],
    )
    default = _validate_real(default, "default")
    K = natural_coordinates.shape[1]
    E = vertices_coordinates.shape[1]
    if K > E:
        raise ValueError(
            f"The topological dimension K ({K}) of the elements cannot be greater than "
            f"the dimension E ({E}) of the space."
        )

    _, shape_function_derivatives, inside_mask = _shape_functions_impl(
        natural_coordinates, element_type, True, tolerance
    )  # (N_p, N_vpe, K)
    _check_located_points_inside(inside_mask, located_mask, element_type, tolerance)

    # Jacobian matrices of the located points (the other rows are not used)
    jacobians = numpy.zeros((natural_coordinates.shape[0], E, K), dtype=numpy.float64)
    indices = numpy.flatnonzero(located_mask)
    jacobians[indices] = _assemble_jacobian_matrix_impl(
        shape_function_derivatives[indices],
        _gather_nodal_values(
            vertices_coordinates, connectivity_array, element_indices[indices]
        ),
    )  # (N_located, E, K)

    return _assemble_property_derivative_impl(
        nodal_values, connectivity_array, element_indices,
        shape_function_derivatives, jacobians, located_mask, default,
    )


# =============================================================================
# Property interpolation
# =============================================================================
def assemble_property_interpolation(
    nodal_values: ArrayLike,
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    shape_functions: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    default: Union[Real, ArrayLike] = numpy.nan,
) -> numpy.ndarray:
    r"""
    Interpolate a property defined at the nodes of a mesh to given integration points
    within elements using precomputed shape functions.

    ``interpolated_values[i]`` contains the interpolated property at the integration
    point :math:`i`.

    .. note::

        - Inputs ``nodal_values`` and ``shape_functions`` will be converted to
          ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The interpolated values of the points with element index ``-1`` (only accepted
        with ``allow_unlocated_points=True``) are set to ``default``. A non-finite
        shape function value of a located point raises a ``ValueError``.


    Parameters
    ----------
    nodal_values: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    shape_functions: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    default: Union[Real, ArrayLike], optional (default: ``numpy.nan``)
        The value assigned to the interpolated values of the points with element
        index ``-1``.
        The input can also be an array of shape :math:`(P,)` to assign different
        default values for each property component.


    Returns
    -------
    interpolated_values: numpy.ndarray
        An array of shape :math:`(N_{p}, P)` containing the interpolated property values
        at each of the :math:`N_{p}` integration points.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the
    interpolated values array has shape :math:`(N_{p}, P)` where :math:`P` is the
    number of property components (e.g., 1 for scalar properties, 3 for vector
    properties).

    The property at each integration point is computed as:

    .. math::

        P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, and :math:`P_i`
    are the property values at the nodes of the element.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.fem.compute_property_interpolation:
        To compute the property interpolation directly from mesh and integration point
        information.

    pysdic.fem.assemble_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh.


    Examples
    --------
    Lets construct a simple 2D mesh and interpolate a scalar property at given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            compute_shape_functions,
            assemble_property_interpolation
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        nodal_values = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        shape_functions = compute_shape_functions(
            natural_coordinates=natural_coordinates,
            element_type='triangle_3',
        )

        interpolated_values = assemble_property_interpolation(
            nodal_values=nodal_values,
            connectivity_array=connectivity_array,
            element_indices=element_indices,
            shape_functions=shape_functions
        )

        print(f"interpolated values (shape={interpolated_values.shape}):")
        print(interpolated_values)

    .. code-block:: console

        interpolated values (shape=(2, 1)):
        [[18.]
         [28.]]

    """
    # Input validation and conversion
    nodal_values = _as_array(
        nodal_values, "nodal_values", numpy.float64,
        shape=(None, None), shape_hint="(N_v, P)", accept_1d_as_column=True, finite=False,
    )
    connectivity_array, _ = _validate_connectivity(
        connectivity_array, n_vertices=nodal_values.shape[0]
    )
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    shape_functions = _as_array(
        shape_functions, "shape_functions", numpy.float64,
        shape=(None, None), shape_hint="(N_p, N_vpe)", finite=False,
    )  # Finite values checked for the located points only
    _check_matching_size(
        shape_functions.shape[0], element_indices.shape[0],
        "shape_functions", "element_indices", "N_p",
    )
    _check_matching_size(
        shape_functions.shape[1], connectivity_array.shape[1],
        "shape_functions", "connectivity_array", "N_vpe",
    )
    _check_finite_rows(shape_functions, located_mask, "shape_functions")
    default = _as_scalar_or_1d_array(
        default, "default", nodal_values.shape[1], numpy.float64, finite=False
    )  # (P,)

    return _assemble_property_interpolation_impl(
        nodal_values, connectivity_array, element_indices, shape_functions,
        located_mask, default,
    )


def compute_property_interpolation(
    nodal_values: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
    default: Union[Real, ArrayLike] = numpy.nan,
) -> numpy.ndarray:
    r"""
    Interpolate a property defined at the nodes of a mesh to given integration points
    within elements using mesh and integration point information.

    This function combines the computation of shape functions and the assembly of the
    property interpolation for the entire mesh.

    ``interpolated_values[i]`` contains the interpolated property at the integration
    point :math:`i`.

    .. note::

        - Inputs ``nodal_values`` and ``natural_coordinates`` will be converted to
          ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The interpolated values of the points with element index ``-1`` (only accepted
        with ``allow_unlocated_points=True``) are set to ``default``. A located point
        whose natural coordinates are outside the reference element (enlarged by
        ``tolerance``) raises a ``ValueError``.


    Parameters
    ----------
    nodal_values: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: str
        A string specifying the type of element (e.g., ``'segment_2'``,
        ``'triangle_3'``, etc.) to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the reference element when checking whether
        the natural coordinates of a point are inside its element (see
        :func:`pysdic.fem.compute_shape_functions`). Useful for points located on the
        boundary of their element up to rounding errors. With the default ``0.0``, only
        the points exactly inside the reference element are accepted.

    default: Union[Real, ArrayLike], optional (default: ``numpy.nan``)
        The value assigned to the interpolated values of the points with element
        index ``-1``.
        The input can also be an array of shape :math:`(P,)` to assign different
        default values for each property component.


    Returns
    -------
    interpolated_values: numpy.ndarray
        An array of shape :math:`(N_{p}, P)` containing the interpolated property values
        at each of the :math:`N_{p}` integration points.


    Notes
    -----
    See :func:`pysdic.fem.assemble_property_interpolation` for the interpolation
    formula. To interpolate with precomputed shape functions, use
    :func:`pysdic.fem.assemble_property_interpolation` directly.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.fem.assemble_property_interpolation:
        To assemble the property interpolation from precomputed shape functions.

    pysdic.fem.compute_property_derivative:
        To compute the derivative of a property at integration points within elements.

    pysdic.fem.compute_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh.


    Examples
    --------
    Lets construct a simple 2D mesh and interpolate a 2-components property at given
    integration points for triangular elements. The last point is outside the mesh.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import compute_property_interpolation

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2],
             [0.0, 0.0]]
        )

        element_indices = numpy.array([0, 1, -1])

        nodal_values = numpy.array(
            [[10.0, 1.0],
             [20.0, 2.0],
             [30.0, 3.0],
             [40.0, 4.0]]
        )

        interpolated_values = compute_property_interpolation(
            nodal_values=nodal_values,
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            allow_unlocated_points=True,  # Accept the -1 of the last point
            default=[-1.0, 0.0],
        )

        print(f"interpolated values (shape={interpolated_values.shape}):")
        print(interpolated_values)

    .. code-block:: console

        interpolated values (shape=(3, 2)):
        [[18.   1.8]
         [28.   2.8]
         [-1.   0. ]]

    """
    # Input validation and conversion (once for the whole computation)
    nodal_values = _as_array(
        nodal_values, "nodal_values", numpy.float64,
        shape=(None, None), shape_hint="(N_v, P)", accept_1d_as_column=True, finite=False,
    )
    (
        element_type, connectivity_array, _, natural_coordinates, tolerance,
        element_indices, located_mask,
    ) = _prepare_points_inputs(
        connectivity_array, element_type, natural_coordinates, element_indices,
        allow_unlocated_points, tolerance, nodal_values.shape[0],
    )
    default = _as_scalar_or_1d_array(
        default, "default", nodal_values.shape[1], numpy.float64, finite=False
    )  # (P,)

    shape_functions, _, inside_mask = _shape_functions_impl(
        natural_coordinates, element_type, False, tolerance
    )
    _check_located_points_inside(inside_mask, located_mask, element_type, tolerance)

    return _assemble_property_interpolation_impl(
        nodal_values, connectivity_array, element_indices, shape_functions,
        located_mask, default,
    )


# =============================================================================
# Property projection
# =============================================================================
def assemble_property_projection(
    point_values: ArrayLike,
    shape_function_matrix: Union[
        ArrayLike, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix
    ],
    point_weights: Optional[ArrayLike] = None,
    *,
    default: Real = numpy.nan,
    return_unaffected: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property defined at integration points within elements back to the nodes
    of a mesh using a precomputed shape function matrix.

    ``projected_values[i]`` contains the projected property at the node :math:`i`.

    .. note::

        - Inputs ``point_values``, ``shape_function_matrix`` and ``point_weights``
          will be converted to ``numpy.float64``.
        - Any ``scipy.sparse`` matrix is accepted and converted to the CSR format.

    .. warning::

        No tests are performed to check if the inputs array are consistent.
        Only shapes are validated.

    .. important::

        The points with a zero weight, non-finite values (e.g. NaN for points outside
        the mesh) or a non-finite row in ``shape_function_matrix`` are ignored.
        The nodes that are not affected by any remaining point are set to ``default``
        and can be retrieved with ``return_unaffected=True``.


    Parameters
    ----------
    point_values: ArrayLike
        An array of shape :math:`(N_{p}, P)` containing the property values defined at
        the :math:`N_{p}` integration points. If 1D-array is provided, it will be
        treated as a single-component property of shape :math:`(N_{p}, 1)`.

    shape_function_matrix: Union[ArrayLike, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]
        A precomputed shape functions matrix of shape :math:`(N_{p}, N_{v})` where
        :math:`N_{v}` is the number of nodes in the mesh.

    point_weights: Optional[ArrayLike], optional (default: ``None``)
        An array of shape :math:`(N_{p},)` containing the non-negative weights
        associated with each integration point. If not provided, all weights will be
        assumed to be equal to one.

    default: Real, optional (default: ``numpy.nan``)
        The value assigned to the nodes that are not affected by any point (no
        information to project).

    return_unaffected: bool, optional (default: ``False``)
        If set to True, the function will return a tuple containing the projected
        values and a boolean mask indicating which nodes were unaffected by any
        integration point.


    Returns
    -------
    projected_values: numpy.ndarray
        An array of shape :math:`(N_{v}, P)` containing the projected property values at
        the nodes of the mesh.

    unaffected_mask: numpy.ndarray
        Only returned if ``return_unaffected`` is True. A boolean array of shape
        :math:`(N_{v},)` indicating which nodes were unaffected by any integration
        point (True for unaffected nodes). These nodes are set to ``default``.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    The evaluation of a property at the given integration points is represented by an
    array of shape :math:`(N_{p}, P)` and is performed as:

    .. math::

        P_{points} = N_f \cdot P_{nodes} \Leftrightarrow P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_f` is the shape functions matrix assembled for all :math:`N_{p}`
    integration points, :math:`P_{points}` is the property array at the integration
    points, and :math:`P_{nodes}` is the property array at the mesh nodes.

    To project back to the nodes, we solve the following weighted least squares
    problem:

    .. math::

        \hat{P}_{nodes} = \underset{P_{nodes}}{\text{argmin}} \; \| W^{1/2} (P_{points} - N_f \cdot P_{nodes}) \|^2

    where :math:`W` is a diagonal matrix of weights associated with each integration
    point. When :math:`N_f^T W N_f` is invertible, the solution is
    :math:`\hat{P}_{nodes} = (N_f^T W N_f)^{-1} N_f^T W P_{points}`.

    The problem is solved directly on :math:`W^{1/2} N_f` (without forming the normal
    equations, whose condition number is squared) with ``numpy.linalg.lstsq`` for dense
    matrices and ``scipy.sparse.linalg.lsqr`` for sparse matrices. If some nodes are
    not sufficiently constrained (e.g. fewer points than nodes in an element), the
    minimum-norm solution is returned instead of raising an error.


    Demonstration
    --------------
    Lets consider the following notations:

    - :math:`N_v` the number of nodes in the mesh.
    - :math:`N_{p}` the number of integration points.
    - :math:`P` the number of property components (e.g., 1 for scalar properties, 3 for vector properties).
    - :math:`P_{points}` the property array at the integration points with shape :math:`(N_{p}, P)`.
    - :math:`P_{nodes}` the property array at the mesh nodes with shape :math:`(N_v, P)`.
    - :math:`N_f` the shape functions matrix with shape :math:`(N_{p}, N_v)`.
    - :math:`W` the diagonal matrix of weights with shape :math:`(N_{p}, N_{p})`.

    We know that:

    .. math::

        P_{points} = N_f \cdot P_{nodes}

    To project back to the vertices of the mesh, we want to minimize the following
    weighted least squares problem:

    .. math::

        \hat{P}_{nodes} = \underset{P_{nodes}}{\text{argmin}} \; \frac{1}{2}\sum_{i=1}^{N_p} w_i \| P_{points,i} - N_{f,i} \cdot P_{nodes} \|^2

    .. math::

        \hat{P}_{nodes} = \underset{P_{nodes}}{\text{argmin}} \; \frac{1}{2} (P_{points} - N_f \cdot P_{nodes})^T W (P_{points} - N_f \cdot P_{nodes})

    The gradient of the objective function with respect to :math:`P_{nodes}` is:

    .. math::

        - N_f^T W (P_{points} - N_f \cdot P_{nodes})

    To find the optimal solution, we set the gradient to zero and solve for
    :math:`P_{nodes}`:

    .. math::

        N_f^T W P_{points} = N_f^T W N_f \cdot P_{nodes}

    So if we denote :math:`A = N_f^T W N_f` invertible, we have:

    .. math::

        \hat{P}_{nodes} = A^{-1} N_f^T W P_{points}


    See Also
    --------
    pysdic.fem.compute_shape_function_matrix:
        To compute the shape function matrix from mesh and integration point
        information.

    pysdic.fem.compute_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh using mesh and integration point information.

    pysdic.fem.assemble_property_interpolation:
        To interpolate properties defined at the nodes of the mesh to integration points
        within elements.


    Examples
    --------
    Lets construct a simple 2D mesh, interpolate a scalar property at 4 points per
    element and project it back to the nodes.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            compute_shape_function_matrix,
            assemble_property_projection,
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )
        N_e = connectivity_array.shape[0]

        nodal_values = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        # Define natural coordinates for interpolation (4 points per element)
        natural_coordinates = numpy.array(
            [[0.3, 0.3],
             [0.2, 0.5],
             [0.5, 0.2],
             [0.1, 0.1]]
        )
        natural_coordinates = numpy.vstack([natural_coordinates] * N_e)
        element_indices = numpy.repeat(numpy.arange(N_e), 4)

        shape_function_matrix = compute_shape_function_matrix(
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
        )

        # Values at the points, then projection back to the nodes
        point_values = shape_function_matrix @ nodal_values

        projected_values = assemble_property_projection(
            point_values=point_values,
            shape_function_matrix=shape_function_matrix
        )

        print(f"projected values (shape={projected_values.shape}):")
        print(projected_values)

    .. code-block:: console

        projected values (shape=(4, 1)):
        [[10.]
         [20.]
         [30.]
         [40.]]

    """
    # Input validation and conversion
    if scipy.sparse.issparse(shape_function_matrix):
        shape_function_matrix = scipy.sparse.csr_matrix(
            shape_function_matrix, dtype=numpy.float64
        )
    else:
        shape_function_matrix = _as_array(
            shape_function_matrix, "shape_function_matrix", numpy.float64,
            shape=(None, None), shape_hint="(N_p, N_v)", finite=False,
        )
    point_values, point_weights, default, return_unaffected = (
        _validate_projection_inputs(
            point_values, point_weights, shape_function_matrix.shape[0], default,
            return_unaffected,
        )
    )

    return _assemble_property_projection_impl(
        point_values, shape_function_matrix, point_weights, default, return_unaffected
    )


def compute_property_projection(
    point_values: ArrayLike,
    connectivity_array: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    point_weights: Optional[ArrayLike] = None,
    *,
    n_vertices: Optional[Integral] = None,
    sparse: bool = False,
    allow_unlocated_points: bool = False,
    tolerance: Real = 0.0,
    default: Real = numpy.nan,
    return_unaffected: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property defined at integration points within elements back to the nodes
    of a mesh using mesh and integration point information.

    This function combines the computation of the shape function matrix and the
    assembly of the property projection for the entire mesh.

    ``projected_values[i]`` contains the projected property at the node :math:`i`.

    .. note::

        - Inputs ``point_values``, ``natural_coordinates`` and ``point_weights`` will
          be converted to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be converted to
          ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements. The geometry of the elements (e.g.
        their orientation) is not checked.

    .. important::

        The points with element index ``-1`` (only accepted with
        ``allow_unlocated_points=True``), a zero weight or non-finite values are
        ignored. A located point whose natural coordinates are outside the reference
        element (enlarged by ``tolerance``) raises a ``ValueError``. The nodes that are
        not affected by any remaining point are set to ``default``.


    Parameters
    ----------
    point_values: ArrayLike
        An array of shape :math:`(N_{p}, P)` containing the property values defined at
        the :math:`N_{p}` integration points. If 1D-array is provided, it will be
        treated as a single-component property of shape :math:`(N_{p}, 1)`.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: str
        A string specifying the type of element (e.g., ``'segment_2'``,
        ``'triangle_3'``, etc.) to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    point_weights: Optional[ArrayLike], optional (default: ``None``)
        An array of shape :math:`(N_{p},)` containing the non-negative weights
        associated with each integration point. If not provided, all weights will be
        assumed to be equal to one.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it
        will be inferred as the maximum index in ``connectivity_array`` plus one.

    sparse: bool, optional (default: ``False``)
        If set to True, the shape functions matrix will be constructed as a
        sparse matrix to optimize memory usage for large meshes.

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.

    tolerance: Real, optional (default: ``0.0``)
        Non-negative tolerance enlarging the reference element when checking whether
        the natural coordinates of a point are inside its element (see
        :func:`pysdic.fem.compute_shape_functions`). Useful for points located on the
        boundary of their element up to rounding errors. With the default ``0.0``, only
        the points exactly inside the reference element are accepted.

    default: Real, optional (default: ``numpy.nan``)
        The value assigned to the nodes that are not affected by any point (no
        information to project).

    return_unaffected: bool, optional (default: ``False``)
        If set to True, the function will return a tuple containing the projected
        values and a boolean mask indicating which nodes were unaffected by any
        integration point.


    Returns
    -------
    projected_values: numpy.ndarray
        An array of shape :math:`(N_{v}, P)` containing the projected property values at
        the nodes of the mesh.

    unaffected_mask: numpy.ndarray
        Only returned if ``return_unaffected`` is True. A boolean array of shape
        :math:`(N_{v},)` indicating which nodes were unaffected by any integration
        point (True for unaffected nodes). These nodes are set to ``default``.


    Notes
    -----
    See the documentation of :func:`pysdic.fem.assemble_property_projection` for the
    mathematical formulation and demonstration of the projection process.


    See Also
    --------
    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.fem.assemble_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh using a precomputed shape functions matrix.

    pysdic.fem.compute_property_interpolation:
        To interpolate properties defined at the nodes of the mesh to integration points
        within elements.


    Examples
    --------
    Lets construct a simple 2D mesh, interpolate a scalar property at 4 points per
    element and project it back to the nodes. An extra point outside the mesh is
    ignored by the projection.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            compute_property_projection,
            compute_property_interpolation
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )
        N_e = connectivity_array.shape[0]

        nodal_values = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        # Define natural coordinates for interpolation (4 points per element)
        natural_coordinates = numpy.array(
            [[0.3, 0.3],
             [0.2, 0.5],
             [0.5, 0.2],
             [0.1, 0.1]]
        )
        natural_coordinates = numpy.vstack([natural_coordinates] * N_e + [[0.0, 0.0]])
        element_indices = numpy.append(numpy.repeat(numpy.arange(N_e), 4), -1)

        point_values = compute_property_interpolation(
            nodal_values=nodal_values,
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            allow_unlocated_points=True,
        )  # NaN for the last point

        # Now project back to the nodes
        projected_values = compute_property_projection(
            point_values=point_values,
            connectivity_array=connectivity_array,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            sparse=True,
            allow_unlocated_points=True,  # The last point is ignored
        )

        print(f"projected values (shape={projected_values.shape}):")
        print(projected_values)

    .. code-block:: console

        projected values (shape=(4, 1)):
        [[10.]
         [20.]
         [30.]
         [40.]]

    """
    # Input validation and conversion (once for the whole computation)
    (
        element_type, connectivity_array, n_vertices, natural_coordinates, tolerance,
        element_indices, located_mask,
    ) = _prepare_points_inputs(
        connectivity_array, element_type, natural_coordinates, element_indices,
        allow_unlocated_points, tolerance, n_vertices,
    )
    sparse = _validate_bool(sparse, "sparse")
    point_values, point_weights, default, return_unaffected = (
        _validate_projection_inputs(
            point_values, point_weights, element_indices.shape[0], default,
            return_unaffected,
        )
    )

    shape_functions, _, inside_mask = _shape_functions_impl(
        natural_coordinates, element_type, False, tolerance
    )
    _check_located_points_inside(inside_mask, located_mask, element_type, tolerance)

    # The points with element index -1 have an empty (zero) row and are ignored
    shape_function_matrix = _assemble_shape_function_matrix_impl(
        connectivity_array, element_indices, shape_functions, located_mask,
        n_vertices, sparse, "csr",
    )
    return _assemble_property_projection_impl(
        point_values, shape_function_matrix, point_weights, default, return_unaffected
    )