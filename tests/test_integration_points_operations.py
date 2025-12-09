# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import pytest

import pysdic


def test_remap_vertices_coordinates():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    element_indices = numpy.random.randint(0, 5, size=(8,))

    remapped_coordinates = pysdic.remap_vertices_coordinates(
        vertices_coordinates,
        element_connectivity,
        element_indices
    )

    assert remapped_coordinates.shape == (8, 3, 3)
    for i in range(8):
        assert numpy.allclose(
            remapped_coordinates[i],
            vertices_coordinates[element_connectivity[element_indices[i]]]
        )

    


def test_remap_vertices_coordinates_with_m1():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    element_indices = numpy.array([0, 1, -1, 3, -1, 4])

    default = 15.8
    remapped_coordinates = pysdic.remap_vertices_coordinates(
        vertices_coordinates,
        element_connectivity,
        element_indices,
        skip_m1=True,
        default=default
    )

    assert remapped_coordinates.shape == (6, 3, 3)
    for i in range(6):
        if element_indices[i] == -1:
            assert numpy.allclose(remapped_coordinates[i], default)
        else:
            assert numpy.allclose(
                remapped_coordinates[i],
                vertices_coordinates[element_connectivity[element_indices[i]]]
            )


def test_assemble_shape_function_matrix():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.random.rand(8, 2) # Ensure sum < 1 for triangle
    mask = natural_coordinates.sum(axis=1) > 1.0
    natural_coordinates[mask] = natural_coordinates[mask] / natural_coordinates.sum(axis=1)[mask][:, None]
    element_indices = numpy.random.randint(0, 5, size=(8,))

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    shape_function_matrix = pysdic.assemble_shape_function_matrix(shape_functions, element_connectivity, element_indices, n_vertices=10)

    assert shape_function_matrix.shape == (8, 10)
    for i in range(8):
        elem_idx = element_indices[i]
        expected_row = numpy.zeros(10)
        expected_row[element_connectivity[elem_idx]] = shape_functions[i]
        assert numpy.allclose(shape_function_matrix[i], expected_row)


def test_assemble_shape_function_matrix_with_m1():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.random.rand(6, 2) # Ensure sum < 1 for triangle
    mask = natural_coordinates.sum(axis=1) > 1.0
    natural_coordinates[mask] = natural_coordinates[mask] / natural_coordinates.sum(axis=1)[mask][:, None]
    element_indices = numpy.array([0, -1, 2, -1, 4, 1])

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    default = 42.15
    shape_function_matrix = pysdic.assemble_shape_function_matrix(
        shape_functions,
        element_connectivity,
        element_indices,
        n_vertices=10,
        skip_m1=True,
        default=default
    )

    assert shape_function_matrix.shape == (6, 10)
    for i in range(6):
        elem_idx = element_indices[i]
        if elem_idx == -1:
            expected_row = numpy.full(10, default)
        else:
            expected_row = numpy.zeros(10)
            expected_row[element_connectivity[elem_idx]] = shape_functions[i]
        assert numpy.allclose(shape_function_matrix[i], expected_row)


def test_assemble_shape_function_matrix_sparse_versus_dense():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.random.rand(8, 2) # Ensure sum < 1 for triangle
    mask = natural_coordinates.sum(axis=1) > 1.0
    natural_coordinates[mask] = natural_coordinates[mask] / natural_coordinates.sum(axis=1)[mask][:, None]
    element_indices = numpy.random.randint(0, 5, size=(8,))

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    shape_function_matrix_dense = pysdic.assemble_shape_function_matrix(
        shape_functions,
        element_connectivity,
        element_indices,
        n_vertices=10,
        sparse=False
    )

    shape_function_matrix_sparse = pysdic.assemble_shape_function_matrix(
        shape_functions,
        element_connectivity,
        element_indices,
        n_vertices=10,
        sparse=True
    ).toarray()

    assert numpy.allclose(shape_function_matrix_dense, shape_function_matrix_sparse)


def test_interpolate_property():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.random.rand(8, 2) # Ensure sum < 1 for triangle
    mask = natural_coordinates.sum(axis=1) > 1.0
    natural_coordinates[mask] = natural_coordinates[mask] / natural_coordinates.sum(axis=1)[mask][:, None]
    element_indices = numpy.random.randint(0, 5, size=(8,))

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    vertex_properties = numpy.random.rand(10, 4)

    interpolated_properties = pysdic.interpolate_property(
        vertex_properties,
        shape_functions,
        element_connectivity,
        element_indices
    )

    assert interpolated_properties.shape == (8, 4)


def test_interpolate_property_at_nodes():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])  # Nodal points for triangle_3

    element_indices = numpy.array([0, 1, 2])

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    vertex_properties = numpy.random.rand(10, 4)

    interpolated_properties = pysdic.interpolate_property(
        vertex_properties,
        shape_functions,
        element_connectivity,
        element_indices
    )

    assert interpolated_properties.shape == (3, 4)
    for i in range(3):
        elem_idx = element_indices[i]
        expected_property = vertex_properties[element_connectivity[elem_idx][i]]
        assert numpy.allclose(interpolated_properties[i], expected_property)



def test_interpolate_property_with_m1():
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    natural_coordinates = numpy.random.rand(6, 2) # Ensure sum < 1 for triangle
    mask = natural_coordinates.sum(axis=1) > 1.0
    natural_coordinates[mask] = natural_coordinates[mask] / natural_coordinates.sum(axis=1)[mask][:, None]
    element_indices = numpy.array([0, -1, 2, -1, 4, 1])

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    vertex_properties = numpy.random.rand(10, 4)

    default = -7.3
    interpolated_properties = pysdic.interpolate_property(
        vertex_properties,
        shape_functions,
        element_connectivity,
        element_indices,
        skip_m1=True,
        default=default
    )

    assert interpolated_properties.shape == (6, 4)
    for i in range(6):
        elem_idx = element_indices[i]
        if elem_idx == -1:
            expected_property = numpy.full(4, default)
        else:
            expected_property = numpy.zeros(4)
            for a in range(3):
                expected_property += shape_functions[i, a] * vertex_properties[element_connectivity[elem_idx][a]]
        assert numpy.allclose(interpolated_properties[i], expected_property)


@pytest.mark.parametrize("P", [0, 1, 2, 3])
def test_project_property(P):
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    N_e = element_connectivity.shape[0]

    natural_coordinates = numpy.array([[0.3, 0.3], [0.2, 0.5], [0.5, 0.2], [0.1, 0.1]])
    N_p = natural_coordinates.shape[0]
    natural_coordinates = numpy.vstack([natural_coordinates] * N_e)
    element_indices = numpy.repeat(numpy.arange(N_e), N_p)
    N_p = natural_coordinates.shape[0]

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    if P != 0:
        vertex_properties = numpy.random.rand(10, P)
    else:
        vertex_properties = numpy.random.rand(10)

    integrated_points_properties = pysdic.interpolate_property(
        vertex_properties,
        shape_functions,
        element_connectivity,
        element_indices
    )

    projected_properties = pysdic.project_property_to_vertices(
        integrated_points_properties,
        shape_functions,
        element_connectivity,
        element_indices,
        n_vertices=10
    )

    assert projected_properties.shape == (10, 1) if P == 0 else (10, P)
    assert numpy.allclose(projected_properties.sum(axis=0), vertex_properties.sum(axis=0))


@pytest.mark.parametrize("P", [0, 1, 2, 3])
def test_project_property(P):
    numpy.random.seed(42)
    vertices_coordinates = numpy.random.rand(10, 3)

    element_connectivity = numpy.array([
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
        [6, 7, 8],
        [8, 9, 0],
    ])

    N_e = element_connectivity.shape[0]

    natural_coordinates = numpy.array([[0.3, 0.3], [0.2, 0.5], [0.5, 0.2], [0.1, 0.1]])
    N_p = natural_coordinates.shape[0]
    natural_coordinates = numpy.vstack([natural_coordinates] * N_e)
    element_indices = numpy.repeat(numpy.arange(N_e), N_p)
    N_p = natural_coordinates.shape[0]

    shape_functions = pysdic.triangle_3_shape_functions(natural_coordinates, return_derivatives=False)

    if P != 0:
        vertex_properties = numpy.random.rand(10, P)
    else:
        vertex_properties = numpy.random.rand(10)

    integrated_points_properties = pysdic.interpolate_property(
        vertex_properties,
        shape_functions,
        element_connectivity,
        element_indices
    )

    projected_properties = pysdic.project_property_to_vertices(
        integrated_points_properties,
        shape_functions,
        element_connectivity,
        element_indices,
        n_vertices=10,
        sparse=True
    )

    assert projected_properties.shape == (10, 1) if P == 0 else (10, P)
    assert numpy.allclose(projected_properties.sum(axis=0), vertex_properties.sum(axis=0))