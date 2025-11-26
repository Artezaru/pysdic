import pytest
import numpy as np
import meshio

import pysdic.geometry as pg
from py3dframe import Frame

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_config import DISPLAY

# ==========================================
# Fixture for creating a sample PointCloud3D
# ==========================================

def random_point_cloud(Np, E):
    np.random.seed(42)
    points = np.random.rand(Np, E)  # Np random points in E dimensions
    return pg.PointCloud.from_array(points)

def other_random_point_cloud(Np, E):
    np.random.seed(43)
    points = np.random.rand(Np, E)  # Np random points in E dimensions
    return pg.PointCloud.from_array(points)

def input_frame():
    return Frame.canonical()

def output_frame():
    translation = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)  # No rotation
    return Frame.from_rotation_matrix(translation=translation, rotation_matrix=rotation)

# ==========================================
# Instance Method Tests
# ==========================================
def test_from_array():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    point_cloud = pg.PointCloud.from_array(points)
    assert isinstance(point_cloud, pg.PointCloud)
    assert point_cloud.points.shape == (4, 3)
    np.testing.assert_array_equal(point_cloud.points, points)


@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_from_meshio(Np, E):
    points = np.random.rand(Np, E)
    cells = {}
    mesh = meshio.Mesh(points=points, cells=cells)
    point_cloud = pg.PointCloud.from_meshio(mesh)
    assert isinstance(point_cloud, pg.PointCloud)
    assert point_cloud.points.shape == (Np, E)
    np.testing.assert_array_equal(point_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_from_to_xyz(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    xyz_filepath = tmp_path / "test_point_cloud.xyz"
    point_cloud.to_xyz(str(xyz_filepath))
    
    loaded_cloud = pg.PointCloud.from_xyz(str(xyz_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_from_to_xyz_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    xyz_filepath = tmp_path / "test_point_cloud.xyz"
    point_cloud.to_xyz(str(xyz_filepath))
    
    loaded_cloud = pg.PointCloud.from_xyz(str(xyz_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_obj(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath))
    
    loaded_cloud = pg.PointCloud.from_obj(str(obj_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_obj_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath))
    
    loaded_cloud = pg.PointCloud.from_obj(str(obj_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)


@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_ply(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath))
    
    loaded_cloud = pg.PointCloud.from_ply(str(ply_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_ply_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath))
    
    loaded_cloud = pg.PointCloud.from_ply(str(ply_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_ply_binary(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath), binary=True)
    
    loaded_cloud = pg.PointCloud.from_ply(str(ply_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_ply_binary_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath), binary=True)
    
    loaded_cloud = pg.PointCloud.from_ply(str(ply_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_vtk(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath))

    loaded_cloud = pg.PointCloud.from_vtk(str(vtk_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_vtk_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), only_finite=True)
    
    loaded_cloud = pg.PointCloud.from_vtk(str(vtk_filepath))
    assert np.array_equal(loaded_cloud.points, points[np.isfinite(points).all(axis=1)], equal_nan=True)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_vtk_binary(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    point_cloud = pg.PointCloud.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), binary=True)
    
    loaded_cloud = pg.PointCloud.from_vtk(str(vtk_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_from_to_vtk_binary_with_nans(tmp_path, Np, E):
    points = np.random.rand(Np, E)
    # Introduce some NaN values
    points[0, :] = np.nan
    points[1, :] = np.nan
    point_cloud = pg.PointCloud.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), binary=True, only_finite=True)
    
    loaded_cloud = pg.PointCloud.from_vtk(str(vtk_filepath))
    assert np.array_equal(loaded_cloud.points, points[np.isfinite(points).all(axis=1)], equal_nan=True)

# ==========================================
# Attribute Tests
# ==========================================
@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_points_attribute(Np, E):
    random_points = random_point_cloud(Np, E)
    assert hasattr(random_points, 'points')
    assert isinstance(random_points.points, np.ndarray)
    assert random_points.points.shape == (Np, E)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_n_points_attribute(Np, E):
    random_points = random_point_cloud(Np, E)
    assert hasattr(random_points, 'n_points')
    assert isinstance(random_points.n_points, int)
    assert random_points.n_points == Np

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_shape_attribute(Np, E):
    random_points = random_point_cloud(Np, E)
    assert hasattr(random_points, 'shape')
    assert isinstance(random_points.shape, tuple)
    assert random_points.shape == (Np, E)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_n_dimensions_attribute(Np, E):
    random_points = random_point_cloud(Np, E)
    assert hasattr(random_points, 'n_dimensions')
    assert isinstance(random_points.n_dimensions, int)
    assert random_points.n_dimensions == E

# ==========================================
# Method Tests
# ==========================================
@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_all_close(Np, E):
    random_point = random_point_cloud(Np, E)
    point_cloud_copy = random_point.copy()
    assert random_point.all_close(point_cloud_copy)

    # Modify a point slightly
    modified_points = random_point.points.copy()
    noise = np.random.normal(0, 1e-10, modified_points.shape)
    modified_points += noise
    modified_cloud = pg.PointCloud.from_array(modified_points)
    assert random_point.all_close(modified_cloud, rtol=1e-5, atol=1e-8)

    # Modify a point significantly
    modified_points = random_point.points.copy()
    noise = np.random.normal(0, 1e-2, modified_points.shape)
    modified_points += noise
    modified_cloud = pg.PointCloud.from_array(modified_points)
    assert not random_point.all_close(modified_cloud, rtol=1e-5, atol=1e-8)

    # Shuffle points
    shuffled_points = random_point.points.copy()
    reindexing = np.random.permutation(random_point.n_points)
    shuffled_points = shuffled_points[reindexing]
    shuffled_cloud = pg.PointCloud.from_array(shuffled_points)
    assert not random_point.all_close(shuffled_cloud)
    assert random_point.all_close(shuffled_cloud, ordered=False)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_copy_object(Np, E):
    random_point = random_point_cloud(Np, E)
    point_cloud_copy = random_point.copy()
    assert isinstance(point_cloud_copy, pg.PointCloud)
    assert point_cloud_copy.n_points == random_point.n_points
    np.testing.assert_array_equal(point_cloud_copy.points, random_point.points)
    
    # Ensure it's a deep copy
    point_cloud_copy.points[0] += 1.0
    assert not np.array_equal(point_cloud_copy.points, random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_as_array(Np, E):
    random_point = random_point_cloud(Np, E)
    array = random_point.to_array()
    assert isinstance(array, np.ndarray)
    assert array.shape == (Np, E)
    np.testing.assert_array_equal(array, random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_bounding_box(Np, E):
    random_point = random_point_cloud(Np, E)
    bbox = random_point.bounding_box()
    assert isinstance(bbox, tuple)
    assert len(bbox) == 2
    min_point, max_point = bbox
    assert min_point.shape == (E,)
    assert max_point.shape == (E,)
    np.testing.assert_array_equal(min_point, np.min(random_point.points, axis=0))
    np.testing.assert_array_equal(max_point, np.max(random_point.points, axis=0))

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_concatenate(Np, E):
    random_point = random_point_cloud(Np, E)
    other_random_point = other_random_point_cloud(Np, E)
    combined = random_point.concatenate(other_random_point)
    assert isinstance(combined, pg.PointCloud)
    assert combined.n_points == random_point.n_points + other_random_point.n_points
    np.testing.assert_array_equal(combined.points[:random_point.n_points], random_point.points)
    np.testing.assert_array_equal(combined.points[random_point.n_points:], other_random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_concatenate_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    other_random_point = other_random_point_cloud(Np, E)
    original_n_points = random_point.n_points
    random_point.concatenate(other_random_point, inplace=True)
    assert random_point.n_points == original_n_points + other_random_point.n_points
    np.testing.assert_array_equal(random_point.points[original_n_points:], other_random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_copy(Np, E):
    random_point = random_point_cloud(Np, E)
    point_cloud_copy = random_point.copy()
    assert isinstance(point_cloud_copy, pg.PointCloud)
    assert point_cloud_copy.n_points == random_point.n_points
    np.testing.assert_array_equal(point_cloud_copy.points, random_point.points)
    # Ensure it's a deep copy
    point_cloud_copy.points[0] += 1.0
    assert not np.array_equal(point_cloud_copy.points, random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_frame_transform(Np, E):
    random_point = random_point_cloud(Np, E)
    i_frame = input_frame()
    o_frame = output_frame()
    transformed = random_point.frame_transform(i_frame, o_frame)
    assert isinstance(transformed, pg.PointCloud)
    assert transformed.n_points == random_point.n_points
    # Since the transformation is a translation by (1,2,3), check that
    expected_points = random_point.points - np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(transformed.points, expected_points)

@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_frame_transform_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    i_frame = input_frame()
    o_frame = output_frame()
    original_points = random_point.points.copy()
    random_point.frame_transform(i_frame, o_frame, inplace=True)
    expected_points = original_points - np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(random_point.points, expected_points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_keep_points(Np, E):
    random_point = random_point_cloud(Np, E)
    # Select a subset of points to keep
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point.points[indices]
    
    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, E) + 10])  # Points far away
    kept_cloud = pg.PointCloud.from_array(kept_points)

    # Test that the kept cloud has the expected points
    extracted_cloud = random_point.keep_points(kept_cloud)
    np.testing.assert_array_equal(extracted_cloud.points, random_point.points[indices])
    assert extracted_cloud.n_points == 50

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_keep_points_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    original_points = random_point.points.copy()

    # Select a subset of points to keep
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point.points[indices]
    
    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, E) + 10])  # Points far away
    kept_cloud = pg.PointCloud.from_array(kept_points)
    # Perform inplace operation
    random_point.keep_points(kept_cloud, inplace=True)
    np.testing.assert_array_equal(random_point.points, original_points[indices])
    assert random_point.n_points == 50

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_keep_points_at(Np, E):
    random_point = random_point_cloud(Np, E)
    indices = np.random.choice(random_point.n_points, size=30, replace=False)
    kept_cloud = random_point.keep_points_at(indices)
    assert kept_cloud.n_points == 30
    np.testing.assert_array_equal(kept_cloud.points, random_point.points[indices])

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_keep_points_at_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    indices = np.random.choice(random_point.n_points, size=30, replace=False)
    original_points = random_point.points.copy()
    random_point.keep_points_at(indices, inplace=True)
    assert random_point.n_points == 30
    np.testing.assert_array_equal(random_point.points, original_points[indices])

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_merge(Np, E):
    random_point = random_point_cloud(Np, E)
    # Select a subset of points to merge
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point.points[indices]

    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, E) + 10])  # Points far away
    kept_cloud = pg.PointCloud.from_array(kept_points)
    merged_cloud = random_point.merge(kept_cloud)

    # The merged cloud should have the same points as the original since all kept points are already present
    np.testing.assert_array_equal(merged_cloud.points, np.vstack([random_point.points, kept_cloud.points[50:]]))
    assert merged_cloud.n_points == random_point.n_points + 10  # 10 new points added

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_merge_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    # Select a subset of points to merge
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point.points[indices]

    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, E) + 10])  # Points far away
    kept_cloud = pg.PointCloud.from_array(kept_points)
    original_n_points = random_point.n_points
    random_point.merge(kept_cloud, inplace=True)

    # The merged cloud should have the same points as the original since all kept points are already present
    assert random_point.n_points == original_n_points + 10  # 10 new points added
    np.testing.assert_array_equal(random_point.points[original_n_points:], kept_cloud.points[50:])

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_remove_points(Np, E):
    random_point = random_point_cloud(Np, E)
    # Select a subset of points to remove
    indices = np.arange(50)  # Remove first 50 points
    removed_points = random_point.points[indices]
    
    # Add some unused indices to test robustness
    removed_points = np.vstack([removed_points, np.random.rand(10, E) + 10])  # Points far away
    removed_cloud = pg.PointCloud.from_array(removed_points)
    reduced_cloud = random_point.remove_points(removed_cloud)
    expected_points = random_point.points[50:]
    np.testing.assert_array_equal(reduced_cloud.points, expected_points)
    assert reduced_cloud.n_points == random_point.n_points - 50


@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_remove_points_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    # Select a subset of points to remove
    indices = np.arange(50)  # Remove first 50 points
    removed_points = random_point.points[indices]
    
    # Add some unused indices to test robustness
    removed_points = np.vstack([removed_points, np.random.rand(10, E) + 10])  # Points far away
    removed_cloud = pg.PointCloud.from_array(removed_points)
    original_n_points = random_point.n_points
    original_points = random_point.points.copy()
    random_point.remove_points(removed_cloud, inplace=True)
    expected_points = original_points[50:]
    np.testing.assert_array_equal(random_point.points, expected_points)
    assert random_point.n_points == original_n_points - 50


@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_remove_points_at(Np, E):
    random_point = random_point_cloud(Np, E)
    indices = np.arange(50)  # Remove first 50 points
    reduced_cloud = random_point.remove_points_at(indices)
    expected_points = random_point.points[50:]
    np.testing.assert_array_equal(reduced_cloud.points, expected_points)
    assert reduced_cloud.n_points == random_point.n_points - 50

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_remove_points_at_inplace(Np, E):
    random_point = random_point_cloud(Np, E)
    indices = np.arange(50)  # Remove first 50 points
    original_n_points = random_point.n_points
    original_points = random_point.points.copy()
    random_point.remove_points_at(indices, inplace=True)
    expected_points = original_points[50:]
    np.testing.assert_array_equal(random_point.points, expected_points)
    assert random_point.n_points == original_n_points - 50


def test_unique():
    # Create a point cloud with duplicates
    array = np.array([[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0]])
    cloud_with_duplicates = pg.PointCloud.from_array(array)
    unique_cloud = cloud_with_duplicates.unique()
    assert unique_cloud.n_points == 4
    expected = np.array([[0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0],
                         [3.0, 3.0, 3.0]])
    np.testing.assert_array_equal(unique_cloud.points, expected)

def test_unique_inplace():
    # Create a point cloud with duplicates
    array = np.array([[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0]])
    cloud_with_duplicates = pg.PointCloud.from_array(array)
    cloud_with_duplicates.unique(inplace=True)
    assert cloud_with_duplicates.n_points == 4
    expected = np.array([[0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0],
                         [3.0, 3.0, 3.0]])
    np.testing.assert_array_equal(cloud_with_duplicates.points, expected)

# ==========================================
# Operation Tests
# ==========================================
@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_addition(Np, E):
    random_point = random_point_cloud(Np, E)
    other_random_point = other_random_point_cloud(Np, E)
    combined = random_point + other_random_point
    assert isinstance(combined, pg.PointCloud)
    assert combined.n_points == random_point.n_points + other_random_point.n_points
    np.testing.assert_array_equal(combined.points[:random_point.n_points], random_point.points)
    np.testing.assert_array_equal(combined.points[random_point.n_points:], other_random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_inplace_addition(Np, E):
    random_point = random_point_cloud(Np, E)
    other_random_point = other_random_point_cloud(Np, E)
    original_n_points = random_point.n_points
    random_point += other_random_point
    assert random_point.n_points == original_n_points + other_random_point.n_points
    np.testing.assert_array_equal(random_point.points[original_n_points:], other_random_point.points)

@pytest.mark.parametrize("Np, E", [(100, 1), (100, 2), (100, 3), (100, 4)])
def test_len(Np, E):
    random_point = random_point_cloud(Np, E)
    assert len(random_point) == random_point.n_points

# ==========================================
# Visualization Tests
# ==========================================
@pytest.mark.parametrize("Np, E", [(100, 3)])
def test_visualize(Np, E):
    random_point = random_point_cloud(Np, E)
    if DISPLAY:
        random_point.visualize(
            color="red",
            point_size=5.0,
        )