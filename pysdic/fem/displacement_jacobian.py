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
from typing import Optional, Union
from numbers import Integral
from numpy.typing import ArrayLike

import numpy
import scipy.sparse

from .._validation import (
    _as_array,
    _check_matching_size,
    _validate_bool,
)
from ._validation import (
    _check_finite_rows,
    _validate_connectivity,
    _validate_element_indices,
)


# Largest index of a sparse matrix stored with numpy.int32 indices
_INT32_MAX = numpy.iinfo(numpy.int32).max


def _assemble_displacement_jacobian_matrix_impl(
    property_gradients: numpy.ndarray,
    shape_functions: numpy.ndarray,
    connectivity_array: numpy.ndarray,
    element_indices: numpy.ndarray,
    located_mask: numpy.ndarray,
    n_vertices: int,
    sparse: bool,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the displacement Jacobian matrix :math:`(M \cdot P, N_{v} \cdot E)` without
    any validation (see :func:`assemble_displacement_jacobian_matrix`).

    The structure of the sparse matrix only depends on the connectivity and on the
    element indices of the points:

    - the row :math:`p M + m` of a located point has :math:`E \cdot N_{vpe}` entries, at the
      columns :math:`e N_{v} + v` for :math:`e = 0..E-1` and the vertices :math:`v` of its
      element sorted in increasing order (sorted column indices, no duplicates since the
      vertices of an element are distinct);
    - the entries of a located point with non-finite gradients are explicit zeros (stored
      entries of value 0);
    - the rows of a point not located (``located_mask`` False) have no entry.

    The sparse matrix is built directly in the CSR format, with ``numpy.int32`` indices when
    possible.

    .. important::

        Internal use only. All the inputs are assumed to be validated:
        ``property_gradients`` float64 :math:`(M, P, E)`, ``shape_functions`` float64
        :math:`(M, N_{vpe})` finite for the located points, ``connectivity_array``
        :math:`(N_{e}, N_{vpe})` with distinct vertices in each element,
        ``element_indices`` :math:`(M,)` valid for the located points.

    """
    M, P, E = property_gradients.shape
    N_vpe = shape_functions.shape[1]
    N_v = n_vertices

    # Located points (all of them in general: no copy by fancy indexing)
    all_located = bool(located_mask.all())
    rows = numpy.arange(M) if all_located else numpy.flatnonzero(located_mask)  # (N_loc,)
    N_loc = rows.size
    gradients = property_gradients if all_located else property_gradients[rows]  # (N_loc, P, E)
    point_elements = element_indices if all_located else element_indices[rows]
    point_shape_functions = shape_functions if all_located else shape_functions[rows]

    # Non-finite gradients (e.g. a point outside the image): explicit zero entries
    finite = numpy.isfinite(gradients)
    if not finite.all():
        gradients = numpy.where(finite.all(axis=(1, 2))[:, numpy.newaxis, numpy.newaxis], gradients, 0.0)

    # Vertices of the element of each point sorted (and the shape functions with them)
    order = numpy.argsort(connectivity_array, axis=1)  # (N_e, N_vpe), once per element
    sorted_connectivity = numpy.take_along_axis(connectivity_array, order, axis=1)
    vertices = sorted_connectivity[point_elements]  # (N_loc, N_vpe)
    values = numpy.take_along_axis(point_shape_functions, order[point_elements], axis=1)  # (N_loc, N_vpe)

    # J[p*M + m, e*N_v + v] = dI_p/dx_e(m) * N_v(m), ordered by (p, m) then (e, v)
    data = numpy.einsum("mpe,mv->pmev", gradients, values)  # (P, N_loc, E, N_vpe)

    nnz = P * N_loc * E * N_vpe
    index_dtype = numpy.int32 if max(nnz, N_v * E) <= _INT32_MAX else numpy.int64
    columns = (
        numpy.arange(E, dtype=index_dtype)[numpy.newaxis, :, numpy.newaxis] * N_v
        + vertices.astype(index_dtype, copy=False)[:, numpy.newaxis, :]
    )  # (N_loc, E, N_vpe)

    if not sparse:
        jacobian = numpy.zeros((P * M, E * N_v), dtype=numpy.float64)
        point_rows = numpy.arange(P)[:, numpy.newaxis] * M + rows[numpy.newaxis, :]  # (P, N_loc)
        jacobian[
            point_rows[:, :, numpy.newaxis],
            columns.reshape(N_loc, E * N_vpe)[numpy.newaxis, :, :],
        ] = data.reshape(P, N_loc, E * N_vpe)
        return jacobian

    # Number of entries of each row: E * N_vpe for the located points, 0 otherwise (same for each p)
    if all_located:
        indptr = numpy.arange(0, nnz + 1, E * N_vpe, dtype=index_dtype)
    else:
        counts = numpy.zeros((P, M), dtype=index_dtype)
        counts[:, rows] = E * N_vpe
        indptr = numpy.zeros(P * M + 1, dtype=index_dtype)
        numpy.cumsum(counts.ravel(), out=indptr[1:])
    indices = numpy.broadcast_to(
        columns.reshape(1, N_loc * E * N_vpe), (P, N_loc * E * N_vpe)
    ).ravel()  # Same columns for each component p

    jacobian = scipy.sparse.csr_matrix(
        (data.ravel(), indices, indptr), shape=(P * M, E * N_v), copy=False
    )
    jacobian.has_sorted_indices = True  # Sorted by construction (no sort by scipy)
    return jacobian


def assemble_displacement_jacobian_matrix(
    property_gradients: ArrayLike,
    shape_functions: ArrayLike,
    connectivity_array: ArrayLike,
    element_indices: ArrayLike,
    *,
    n_vertices: Optional[Integral] = None,
    sparse: bool = False,
    allow_unlocated_points: bool = False,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the displacement Jacobian matrix that maps nodal displacements to the
    variations of a property (e.g. gray levels) at integration points.

    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` vertices, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` vertices for each element.

    Lets consider :math:`M` integration points located within elements and a property
    with :math:`P` components (e.g. :math:`P = 1` for gray levels, :math:`P = 3` for
    RGB images). The gradients of the property with respect to the global coordinates
    at the integration points are given as an array of shape :math:`(M, P, E)`.

    The objective is to build the Jacobian operator
    :math:`\mathbf{J} \in \mathbb{R}^{(M \cdot P) \times (N_{v} \cdot E)}` such that:

    .. math::

        \mathbf{J} \, dU = dR

    where :math:`dU \in \mathbb{R}^{N_{v} \cdot E}` is the vector of nodal displacements
    and :math:`dR \in \mathbb{R}^{M \cdot P}` is the resulting variation of the
    property (residual) at the integration points.

    .. note::

        For (:math:`P \neq 1`), the lines are grouped by property first (i.e., all
        integration points for property 1, followed by all integration points for
        property 2, etc.).
        Thus the residual :math:`R` associated with the output must be constructed
        accordingly.

        .. math::

            R = \begin{bmatrix}
            R_{1}^{1} \\
            R_{2}^{1} \\
            \vdots \\
            R_{M}^{1} \\
            R_{1}^{2} \\
            R_{2}^{2} \\
            \vdots \\
            R_{M}^{2} \\
            \vdots \\
            R_{1}^{P} \\
            R_{2}^{P} \\
            \vdots \\
            R_{M}^{P}
            \end{bmatrix}

        where :math:`R_{i}^{j}` represents the residual at integration point
        :math:`i` for property :math:`j`.

        .. code-block:: python

            residual = numpy.random.rand(M, P)
            residual_vector = residual.flatten(order='F')  # Column-major flattening

    .. note::

        The columns of the operator correspond to the nodal displacements organized
        by direction first (i.e., all vertices for direction 1, followed by all
        vertices for direction 2, etc.).
        Thus the displacement vector :math:`dU` must be constructed accordingly.

        .. math::

            dU = \begin{bmatrix}
            dU_{1}^{x} \\
            dU_{2}^{x} \\
            \vdots \\
            dU_{N_{v}}^{x} \\
            dU_{1}^{y} \\
            dU_{2}^{y} \\
            \vdots \\
            dU_{N_{v}}^{y} \\
            \vdots \\
            dU_{1}^{E} \\
            dU_{2}^{E} \\
            \vdots \\
            dU_{N_{v}}^{E}
            \end{bmatrix}

        where :math:`dU_{i}^{j}` represents the displacement of vertex :math:`i` in
        direction :math:`j`.

        .. code-block:: python

            displacement_vector = numpy.random.rand(N_v * E) # resulting vector of size N_v * E
            displacement = displacement_vector.reshape((N_v, E), order='F')  # Column-major reshaping

    .. note::

        - Inputs ``property_gradients`` and ``shape_functions`` will be converted
          to ``numpy.float64``.
        - Inputs ``connectivity_array`` and ``element_indices`` will be
          converted to ``numpy.int64``.

    .. note::

        The connectivity is validated (non-negative vertex indices lower than the
        number of vertices, distinct vertices in each element) and the element indices
        are checked against the number of elements.

    .. important::

        The :math:`P` rows of a point are zeros, i.e. the point has a zero weight:

        - if its property gradients are not finite (e.g. a point outside the image): in
          the sparse matrix, its entries are kept as explicit zeros, so that the structure
          of the matrix only depends on the connectivity and on the element indices (the
          same for all the gradients);
        - if its element index is ``-1`` (only accepted with
          ``allow_unlocated_points=True``): its rows have no entry in the sparse matrix.

        A non-finite shape function value of a located point raises a ``ValueError``.


    Parameters
    ----------
    property_gradients: ArrayLike
        An array of shape :math:`(M, P, E)` containing the gradients of the property
        with respect to the global coordinates :math:`(x, y, z, ...)` at the :math:`M`
        integration points. If a 2D-array of shape :math:`(M, E)` is provided, it will
        be treated as a single-component property of shape :math:`(M, 1, E)`.

    shape_functions: ArrayLike
        An array of shape :math:`(M, N_{vpe})` containing the shape function values
        evaluated at the :math:`M` points for the :math:`N_{vpe}` vertices of the
        element.

    connectivity_array: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the vertices that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(M,)` containing the indices of each element
        corresponding to the :math:`M` integration points.

    n_vertices: Optional[Integral], optional (default: ``None``)
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum vertex index in
        ``connectivity_array`` plus one.

    sparse: bool, optional (default: ``False``)
        If set to True, the function will return a ``scipy.sparse.csr_matrix``
        representation of the operator, built directly in the CSR format (sorted
        column indices, no duplicates, ``numpy.int32`` indices when possible).

    allow_unlocated_points: bool, optional (default: ``False``)
        If False (default), any negative element index raises an ``IndexError``.
        If True, the element index ``-1`` marks a point not located in any element
        of the mesh, and such a point is treated as invalid (see above). Other
        negative indices always raise an ``IndexError``.


    Returns
    -------
    displacement_jacobian_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The displacement Jacobian operator of shape :math:`(M \cdot P, N_{v} \cdot E)`
        mapping nodal displacements to property variations at integration points.


    Notes
    -----
    The displacement of a point :math:`m` of the element is interpolated from the
    nodal displacements with the shape functions:
    :math:`u(m) = \sum_{v} N_{v}(m) \, U_{v}`.
    At first order, the variation of the component :math:`p` of the property at this
    point is:

    .. math::

        dR_{m}^{p} = \nabla_X I_{p}(m) \cdot u(m) = \sum_{e=1}^{E} \sum_{v=1}^{N_{v}} \frac{\partial I_{p}}{\partial x_{e}}(m) \, N_{v}(m) \, dU_{v}^{e}

    So the coefficient of the operator at row :math:`p \cdot M + m` and column
    :math:`e \cdot N_{v} + v` is:

    .. math::

        \mathbf{J}[p \cdot M + m, \; e \cdot N_{v} + v] = \frac{\partial I_{p}}{\partial x_{e}}(m) \, N_{v}(m)

    In other words, the operator is made of :math:`P \times E` blocks, each block being
    the shape function matrix :math:`\mathcal{N}` (see
    :func:`pysdic.fem.assemble_shape_function_matrix`) scaled row by row by one
    component of the gradients:

    .. math::

        \mathbf{J} = \begin{bmatrix}
        \text{diag}(\partial_{x_1} I_{1}) \, \mathcal{N} & \cdots & \text{diag}(\partial_{x_E} I_{1}) \, \mathcal{N} \\
        \vdots & \ddots & \vdots \\
        \text{diag}(\partial_{x_1} I_{P}) \, \mathcal{N} & \cdots & \text{diag}(\partial_{x_E} I_{P}) \, \mathcal{N}
        \end{bmatrix}


    See Also
    --------
    pysdic.fem.assemble_shape_function_matrix:
        To assemble the shape function matrix :math:`\mathcal{N}` used to build each
        block of the operator.

    pysdic.fem.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.


    Examples
    --------
    Lets construct a simple 2D mesh with 3 integration points, a 2-components property,
    and check that the operator maps a nodal displacement to the expected variation of
    the property.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic.fem import (
            assemble_displacement_jacobian_matrix,
            compute_triangle_3_shape_functions,
        )

        connectivity_array = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )
        N_v, E = 4, 2

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2],
             [0.0, 0.0]]
        )
        element_indices = numpy.array([0, 1, -1])  # Last point outside the mesh
        shape_functions = compute_triangle_3_shape_functions(natural_coordinates)

        # Gradients of a 2-components property, shape (M, P, E) = (3, 2, 2)
        property_gradients = numpy.array(
            [[[1.0, 0.0], [0.0, 2.0]],
             [[1.0, 1.0], [3.0, 0.0]],
             [[5.0, 5.0], [5.0, 5.0]]]
        )

        jacobian = assemble_displacement_jacobian_matrix(
            property_gradients=property_gradients,
            shape_functions=shape_functions,
            connectivity_array=connectivity_array,
            element_indices=element_indices,
            n_vertices=N_v,
            allow_unlocated_points=True,  # Accept the -1 of the last point
        )
        print(f"jacobian (shape={jacobian.shape}):")
        print(jacobian)

        # Uniform translation (1, 0) of all the vertices
        displacement = numpy.zeros((N_v, E))
        displacement[:, 0] = 1.0
        variation = jacobian @ displacement.flatten(order='F')  # (M * P,)
        print("variation (M, P):")
        print(variation.reshape((3, 2), order='F'))

    .. code-block:: console

        jacobian (shape=(6, 8)):
        [[0.5 0.2 0.3 0.  0.  0.  0.  0. ]
         [0.2 0.  0.6 0.2 0.2 0.  0.6 0.2]
         [0.  0.  0.  0.  0.  0.  0.  0. ]
         [0.  0.  0.  0.  1.  0.4 0.6 0. ]
         [0.6 0.  1.8 0.6 0.  0.  0.  0. ]
         [0.  0.  0.  0.  0.  0.  0.  0. ]]
        variation (M, P):
        [[1. 0.]
         [1. 3.]
         [0. 0.]]

    For a uniform translation, the variation is the :math:`x`-component of the gradients
    at each valid point, and zero at the invalid point.

    """
    # Input validation and conversion
    # (M, E) is accepted as a single-component property (M, 1, E).
    # Non-finite gradients are accepted (the point gets a zero weight, see above).
    property_gradients = numpy.asarray(property_gradients, dtype=numpy.float64)
    if property_gradients.ndim == 2:
        property_gradients = property_gradients[:, numpy.newaxis, :]
    property_gradients = _as_array(
        property_gradients, "property_gradients", numpy.float64,
        shape=(None, None, None), shape_hint="(M, E) or (M, P, E)", finite=False,
    )
    shape_functions = _as_array(
        shape_functions, "shape_functions", numpy.float64,
        shape=(None, None), shape_hint="(M, N_vpe)", finite=False,
    )  # Finite values checked for the located points only
    connectivity_array, n_vertices = _validate_connectivity(
        connectivity_array, n_vertices
    )
    element_indices, located_mask = _validate_element_indices(
        element_indices, connectivity_array.shape[0], allow_unlocated_points
    )
    _check_matching_size(
        property_gradients.shape[0], shape_functions.shape[0],
        "property_gradients", "shape_functions", "M",
    )
    _check_matching_size(
        element_indices.shape[0], shape_functions.shape[0],
        "element_indices", "shape_functions", "M",
    )
    _check_matching_size(
        shape_functions.shape[1], connectivity_array.shape[1],
        "shape_functions", "connectivity_array", "N_vpe",
    )
    sparse = _validate_bool(sparse, "sparse")
    _check_finite_rows(shape_functions, located_mask, "shape_functions")

    return _assemble_displacement_jacobian_matrix_impl(
        property_gradients, shape_functions, connectivity_array, element_indices,
        located_mask, n_vertices, sparse,
    )