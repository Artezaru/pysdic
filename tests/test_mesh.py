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

import pytest
import numpy
import meshio

# test_linear_triangle_mesh_3d.py

import numpy as np
import pytest
import os
import open3d
from pysdic import Mesh, PointCloud
from pysdic import create_triangle_3_heightmap

def simple_mesh():
    """Fixture to create a simple 2D triangle integration points instance."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")
    return mesh3d

def simple_mesh_with_properties():
    """Fixture to create a simple 2D triangle integration points instance."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

    mesh3d.set_vertices_property("temperature", np.array([100.0, 150.0, 200.0, 250.0]).reshape(-1, 1)) # shape (N, A)
    mesh3d.set_elements_property("material_id", np.array([1, 1, 2, 2]).reshape(-1, 1)) # shape (M, B)

    return mesh3d

def simple_mesh_with_complex_properties():
    """Fixture to create a simple 2D triangle integration points instance."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

    mesh3d.set_vertices_property("displacement", np.array([[0.0, 0.0, 0.0],
                                                           [0.1, 0.0, 0.0],
                                                           [0.0, 0.1, 0.0],
                                                           [0.0, 0.0, 0.1]])) # shape (N, A)
    mesh3d.set_vertices_property("temperature", np.array([100.0, 150.0, 200.0, 250.0]).reshape(-1, 1)) # shape (N, A)

    mesh3d.set_elements_property("normals", np.array([[0.0, 0.0, 1.0],
                                                      [0.0, 1.0, 0.0],
                                                      [1.0, 0.0, 0.0],
                                                      [1.0, 1.0, 1.0]])) # shape (M, B)
        
    mesh3d.set_elements_property("material_id", np.array([1, 1, 2, 2]).reshape(-1, 1)) # shape (M, B)

    return mesh3d

def simple_heightmap():
    """Fixture to create a simple 2D triangle heightmap instance."""
    surface_mesh = create_triangle_3_heightmap(
        height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        n_x=50,
        n_y=50,
    )
    return surface_mesh

def create_integration_points_mesh(mesh):
    """Fixture to create a simple 2D triangle integration points instance."""
    natural_coordinates = np.array([[0.1, 0.1],
                                    [0.6, 0.2],
                                    [0.2, 0.7],
                                    [0.3, 0.3],
                                    [0.5, 0.4],
                                    [0.4, 0.5],
                                    [0.25, 0.25],
                                    [0.75, 0.1],
                                    [0.1, 0.75]])
    #repeat for all elements
    indices = np.repeat(np.arange(mesh.n_elements).reshape(-1, 1), natural_coordinates.shape[0], axis=1).T # shape (N_points, N_elements)
    # reshape (N_points * N_elements, ) order [ e0, e0, e0, ..., e1, e1, e1, ..., e2, e2, e2, ..., e3, e3, e3 ]
    indices = indices.flatten()

    # repeat natural coordinates for all elements
    natural_coordinates = np.tile(natural_coordinates, (mesh.n_elements, 1)) # shape (N_points * N_elements, 2)

    return natural_coordinates, indices


# === Test for instanciation === #
@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_to_from_meshio(mesh):
    """ Test conversion to and from meshio Mesh object. """

    if not mesh.is_empty():
        meshio_mesh = mesh.to_meshio()
    else:
        # Check if empty mesh raises error
        with pytest.raises(ValueError):
            mesh.to_meshio()
        return

    mesh_from_meshio = Mesh.from_meshio(meshio_mesh)

    assert mesh.n_vertices == mesh_from_meshio.n_vertices
    assert mesh.n_elements == mesh_from_meshio.n_elements

    np.testing.assert_array_equal(mesh.connectivity, mesh_from_meshio.connectivity)
    np.testing.assert_array_equal(mesh.vertices.points, mesh_from_meshio.vertices.points)

    for key in mesh.list_vertices_properties():
        np.testing.assert_array_equal(
            mesh.get_vertices_property(key),
            mesh_from_meshio.get_vertices_property(key)
        )

    for key in mesh.list_elements_properties():
        np.testing.assert_array_equal(
            mesh.get_elements_property(key),
            mesh_from_meshio.get_elements_property(key)
        )


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_from_to_npz(mesh, tmp_path):
    """ Test conversion to and from NPZ file. """
    file = tmp_path / "temp_mesh.npz"

    if not mesh.is_empty():
        mesh.to_npz(file)
    else:
        # Check if empty mesh raises error
        with pytest.raises(ValueError):
            mesh.to_npz(file)
        return
    mesh_from_npz = Mesh.from_npz(file)

    assert mesh.n_vertices == mesh_from_npz.n_vertices
    assert mesh.n_elements == mesh_from_npz.n_elements

    np.testing.assert_array_equal(mesh.connectivity, mesh_from_npz.connectivity)
    np.testing.assert_array_equal(mesh.vertices.points, mesh_from_npz.vertices.points)

    for key in mesh.list_vertices_properties():
        np.testing.assert_array_equal(
            mesh.get_vertices_property(key),
            mesh_from_npz.get_vertices_property(key)
        )

    for key in mesh.list_elements_properties():
        np.testing.assert_array_equal(
            mesh.get_elements_property(key),
            mesh_from_npz.get_elements_property(key)
        )


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_from_to_vtk(mesh, tmp_path):
    """ Test conversion to and from VTK file. """
    file = tmp_path / "temp_mesh.vtk"

    if not mesh.is_empty():
        mesh.to_vtk(file)
    else:
        # Check if empty mesh raises error
        with pytest.raises(ValueError):
            mesh.to_vtk(file)
        return
    mesh_from_vtk = Mesh.from_vtk(file)

    assert mesh.n_vertices == mesh_from_vtk.n_vertices
    assert mesh.n_elements == mesh_from_vtk.n_elements

    np.testing.assert_array_equal(mesh.connectivity, mesh_from_vtk.connectivity)
    np.testing.assert_array_equal(mesh.vertices.points, mesh_from_vtk.vertices.points)

    for key in mesh.list_vertices_properties():
        np.testing.assert_array_equal(
            mesh.get_vertices_property(key),
            mesh_from_vtk.get_vertices_property(key)
        )

    for key in mesh.list_elements_properties():
        np.testing.assert_array_equal(
            mesh.get_elements_property(key),
            mesh_from_vtk.get_elements_property(key)
        )


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_from_mesh(mesh):
    """ Test conversion from another mesh instance. """
    other = mesh.copy()

    assert mesh.n_vertices == other.n_vertices
    assert mesh.n_elements == other.n_elements

    np.testing.assert_array_equal(mesh.connectivity, other.connectivity)
    np.testing.assert_array_equal(mesh.vertices.points, other.vertices.points)

    for key in mesh.list_vertices_properties():
        np.testing.assert_array_equal(
            mesh.get_vertices_property(key),
            other.get_vertices_property(key)
        )

    for key in mesh.list_elements_properties():
        np.testing.assert_array_equal(
            mesh.get_elements_property(key),
            other.get_elements_property(key)
        )

        

# === Manage vertices and elements properties === #
@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_clear_elements_properties(mesh):
    """ Test clearing of elements properties. """
    mesh.clear_elements_properties()
    assert mesh.list_elements_properties() == ()
    assert mesh._elements_properties == {}

@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_clear_vertices_properties(mesh):
    """ Test clearing of vertices properties. """
    mesh.clear_vertices_properties()
    assert mesh.list_vertices_properties() == ()
    assert mesh._vertices_properties == {}

@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_clear_properties(mesh):
    """ Test clearing of all properties. """
    mesh.clear_properties()
    assert mesh.list_vertices_properties() == ()
    assert mesh.list_elements_properties() == ()
    assert mesh._vertices_properties == {}
    assert mesh._elements_properties == {}

@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_get_elements_property(mesh):
    """ Test getting elements properties. """
    for key in mesh.list_elements_properties():
        prop = mesh.get_elements_property(key)
        assert prop.shape[0] == mesh.n_elements

    not_existing_key = "non_existing_property"
    value = mesh.get_elements_property(not_existing_key)
    assert value is None

@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_get_vertices_property(mesh):
    """ Test getting vertices properties. """
    for key in mesh.list_vertices_properties():
        prop = mesh.get_vertices_property(key)
        assert prop.shape[0] == mesh.n_vertices

    not_existing_key = "non_existing_property"
    value = mesh.get_vertices_property(not_existing_key)
    assert value is None


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_remove_elements_property(mesh):
    """ Test removing elements properties. """
    for key in mesh.list_elements_properties():
        mesh.remove_elements_property(key)
        assert key not in mesh.list_elements_properties()

    # Removing non-existing property should not raise error
    with pytest.raises(KeyError):
        mesh.remove_elements_property("non_existing_property")


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_remove_vertices_property(mesh):
    """ Test removing vertices properties. """
    for key in mesh.list_vertices_properties():
        mesh.remove_vertices_property(key)
        assert key not in mesh.list_vertices_properties()

    # Removing non-existing property should raise error
    with pytest.raises(KeyError):
        mesh.remove_vertices_property("non_existing_property")



# === Topology modification === #
@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_add_element(mesh):
    """ Test add_element method. """
    # Select 3 random vertices to form a new triangle
    if mesh.n_vertices < 3:
        return  # Cannot add connectivity if less than 3 vertices
    
    new_triangle = np.random.choice(mesh.n_vertices, size=3, replace=False).reshape(1, 3)
    
    old_n_elements = mesh.n_elements
    mesh.add_elements(new_triangle)
    assert mesh.n_elements == old_n_elements + 1
    np.testing.assert_array_equal(mesh.connectivity[-1], new_triangle.flatten())

    for key in mesh.list_elements_properties():
        prop = mesh.get_elements_property(key)
        assert prop.shape[0] == old_n_elements + 1  # New property should have one more entry
        assert np.all(np.isnan(prop[-1, :]))  # New entry should be initialized to NaN


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_add_vertices(mesh):
    """ Test add_vertices method. """
    # Create 2 new random vertices
    new_vertices = np.random.rand(2, 3)
    old_n_vertices = mesh.n_vertices
    mesh.add_vertices(new_vertices)
    assert mesh.n_vertices == old_n_vertices + 2
    np.testing.assert_array_equal(mesh.vertices.points[-2:, :], new_vertices)

    for key in mesh.list_vertices_properties():
        prop = mesh.get_vertices_property(key)
        assert prop.shape[0] == old_n_vertices + 2  # New property should have two more entries
        assert np.all(np.isnan(prop[-2:, :]))  # New entries should be initialized to NaN


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_is_empty(mesh):
    """ Test is_empty method. """
    if mesh.n_vertices == 0 or mesh.n_elements == 0:
        assert mesh.is_empty() is True
    else:
        assert mesh.is_empty() is False


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_remove_elements(mesh):
    """ Test remove_elements method. """
    if mesh.n_elements == 0:
        return  # Cannot remove elements from empty mesh

    # Remove the first element
    old_connectivity = mesh.connectivity.copy()
    element_index = 0
    old_n_elements = mesh.n_elements
    mesh.remove_elements([element_index])
    assert mesh.n_elements == old_n_elements - 1
    np.testing.assert_array_equal(mesh.connectivity, old_connectivity[1:, :])  # Remaining connectivity should be unchanged

    for key in mesh.list_elements_properties():
        prop = mesh.get_elements_property(key)
        assert prop.shape[0] == old_n_elements - 1  # Property should have one less entry


@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_remove_vertices(mesh):
    """ Test remove_vertices method. """
    if mesh.n_vertices < 2:
        return  # Cannot remove vertices if less than 2 vertices

    # Add a new vertex that is not used in connectivity
    new_vertex = np.array([[10.0, 10.0, 10.0]]).reshape(1, 3)
    mesh.add_vertices(new_vertex)
    unused_vertex_index = mesh.n_vertices - 1

    # Remove the unused vertex
    mesh.remove_vertices([unused_vertex_index])
    assert mesh.n_vertices == unused_vertex_index  # One vertex should be removed
    np.testing.assert_array_equal(mesh.vertices.points, mesh.vertices.points[:unused_vertex_index, :])  # Remaining vertices should be unchanged
    for key in mesh.list_vertices_properties():
        prop = mesh.get_vertices_property(key)
        assert prop.shape[0] == unused_vertex_index  # Property should have one less entry

    
@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_remove_unused_vertices(mesh):
    """ Test remove_unused_vertices method. """
    if mesh.n_vertices == 0:
        return  # Cannot remove vertices from empty mesh

    # Add two new vertices that are not used in connectivity
    new_vertices = np.array([[10.0, 10.0, 10.0],
                             [20.0, 20.0, 20.0]]).reshape(2, 3)
    mesh.add_vertices(new_vertices)
    unused_vertex_indices = [mesh.n_vertices - 2, mesh.n_vertices - 1]

    old_n_vertices = mesh.n_vertices
    mesh.remove_unused_vertices()
    assert mesh.n_vertices == old_n_vertices - 2  # Two vertices should be removed
    np.testing.assert_array_equal(mesh.vertices.points, mesh.vertices.points[:old_n_vertices - 2, :])  # Remaining vertices should be unchanged
    for key in mesh.list_vertices_properties():
        prop = mesh.get_vertices_property(key)
        assert prop.shape[0] == old_n_vertices - 2  # Property should have two less entries


# === Manipulating Mesh3D objects === #
@pytest.mark.parametrize("mesh", [simple_mesh(), simple_mesh_with_properties(), simple_mesh_with_complex_properties(), simple_heightmap()])
def test_copy(mesh):
    """ Test copy method. """
    mesh_copy = mesh.copy()

    assert mesh.n_vertices == mesh_copy.n_vertices
    assert mesh.n_elements == mesh_copy.n_elements

    np.testing.assert_array_equal(mesh.connectivity, mesh_copy.connectivity)
    np.testing.assert_array_equal(mesh.vertices.points, mesh_copy.vertices.points)

    for key in mesh.list_vertices_properties():
        np.testing.assert_array_equal(
            mesh.get_vertices_property(key),
            mesh_copy.get_vertices_property(key)
        )

    for key in mesh.list_elements_properties():
        np.testing.assert_array_equal(
            mesh.get_elements_property(key),
            mesh_copy.get_elements_property(key)
        )



