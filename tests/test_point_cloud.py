import pytest
import numpy as np

from pysdic.geometry import PointCloud3D
from py3dframe import Frame

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_config import DISPLAY

# ==========================================
# Fixture for creating a sample PointCloud3D
# ==========================================

@pytest.fixture
def random_point_cloud():
    np.random.seed(42)
    points = np.random.rand(100, 3)  # 100 random points in 3D
    return PointCloud3D.from_array(points)

@pytest.fixture
def other_random_point_cloud():
    np.random.seed(43)
    points = np.random.rand(100, 3)  # 150 random points in 3D
    return PointCloud3D.from_array(points)

@pytest.fixture
def input_frame():
    return Frame.canonical()

@pytest.fixture
def output_frame():
    translation = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)  # No rotation
    return Frame.from_rotation_matrix(translation=translation, rotation_matrix=rotation)

# ==========================================
# Instance Method Tests
# ==========================================
def test_from_array():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    point_cloud = PointCloud3D.from_array(points)
    assert isinstance(point_cloud, PointCloud3D)
    assert point_cloud.points.shape == (4, 3)
    np.testing.assert_array_equal(point_cloud.points, points)

def test_from_cls(random_point_cloud):
    point_cloud = PointCloud3D.from_cls(random_point_cloud)
    assert isinstance(point_cloud, PointCloud3D)
    np.testing.assert_array_equal(point_cloud.points, random_point_cloud.points)

def test_from_empty():
    point_cloud = PointCloud3D.from_empty()
    assert isinstance(point_cloud, PointCloud3D)
    assert point_cloud.points.shape == (0, 3)


def test_from_to_xyz(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    xyz_filepath = tmp_path / "test_point_cloud.xyz"
    point_cloud.to_xyz(str(xyz_filepath))
    
    loaded_cloud = PointCloud3D.from_xyz(str(xyz_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)


def test_from_to_xyz_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    xyz_filepath = tmp_path / "test_point_cloud.xyz"
    point_cloud.to_xyz(str(xyz_filepath))
    
    loaded_cloud = PointCloud3D.from_xyz(str(xyz_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)


def test_from_to_obj(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath))
    
    loaded_cloud = PointCloud3D.from_obj(str(obj_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)


def test_from_to_obj_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath))
    
    loaded_cloud = PointCloud3D.from_obj(str(obj_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)


def test_from_to_obj_binary(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath), binary=True)
    
    loaded_cloud = PointCloud3D.from_obj(str(obj_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)


def test_from_to_obj_binary_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    obj_filepath = tmp_path / "test_point_cloud.obj"
    point_cloud.to_obj(str(obj_filepath), binary=True)
    
    loaded_cloud = PointCloud3D.from_obj(str(obj_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)


def test_from_to_ply(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath))
    
    loaded_cloud = PointCloud3D.from_ply(str(ply_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

def test_from_to_ply_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath))
    
    loaded_cloud = PointCloud3D.from_ply(str(ply_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)

def test_from_to_ply_binary(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath), binary=True)
    
    loaded_cloud = PointCloud3D.from_ply(str(ply_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

def test_from_to_ply_binary_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    ply_filepath = tmp_path / "test_point_cloud.ply"
    point_cloud.to_ply(str(ply_filepath), binary=True)
    
    loaded_cloud = PointCloud3D.from_ply(str(ply_filepath))
    assert np.array_equal(loaded_cloud.points, points, equal_nan=True)

def test_from_to_vtk(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath))
    
    loaded_cloud = PointCloud3D.from_vtk(str(vtk_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

def test_from_to_vtk_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), only_finite=True)
    
    loaded_cloud = PointCloud3D.from_vtk(str(vtk_filepath))
    assert np.array_equal(loaded_cloud.points, points[np.isfinite(points).all(axis=1)], equal_nan=True)

def test_from_to_vtk_binary(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    point_cloud = PointCloud3D.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), binary=True)
    
    loaded_cloud = PointCloud3D.from_vtk(str(vtk_filepath))
    np.testing.assert_array_equal(loaded_cloud.points, points)

def test_from_to_vtk_binary_with_nans(tmp_path):
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan]])
    point_cloud = PointCloud3D.from_array(points)
    
    vtk_filepath = tmp_path / "test_point_cloud.vtk"
    point_cloud.to_vtk(str(vtk_filepath), binary=True, only_finite=True)
    
    loaded_cloud = PointCloud3D.from_vtk(str(vtk_filepath))
    assert np.array_equal(loaded_cloud.points, points[np.isfinite(points).all(axis=1)], equal_nan=True)

# ==========================================
# Attribute Tests
# ==========================================
def test_points_attribute(random_point_cloud):
    assert hasattr(random_point_cloud, 'points')
    assert isinstance(random_point_cloud.points, np.ndarray)
    assert random_point_cloud.points.shape == (100, 3)

def test_n_points_attribute(random_point_cloud):
    assert hasattr(random_point_cloud, 'n_points')
    assert isinstance(random_point_cloud.n_points, int)
    assert random_point_cloud.n_points == 100

def test_shape_attribute(random_point_cloud):
    assert hasattr(random_point_cloud, 'shape')
    assert isinstance(random_point_cloud.shape, tuple)
    assert random_point_cloud.shape == (100, 3)

# ==========================================
# Method Tests
# ==========================================
def test_allclose(random_point_cloud):
    point_cloud_copy = PointCloud3D.from_cls(random_point_cloud)
    assert random_point_cloud.allclose(point_cloud_copy)

    # Modify a point slightly
    modified_points = random_point_cloud.points.copy()
    noise = np.random.normal(0, 1e-10, modified_points.shape)
    modified_points += noise
    modified_cloud = PointCloud3D.from_array(modified_points)
    assert random_point_cloud.allclose(modified_cloud, rtol=1e-5, atol=1e-8)

    # Modify a point significantly
    modified_points = random_point_cloud.points.copy()
    noise = np.random.normal(0, 1e-2, modified_points.shape)
    modified_points += noise
    modified_cloud = PointCloud3D.from_array(modified_points)
    assert not random_point_cloud.allclose(modified_cloud, rtol=1e-5, atol=1e-8)

def test_copy_object(random_point_cloud):
    point_cloud_copy = random_point_cloud.copy()
    assert isinstance(point_cloud_copy, PointCloud3D)
    assert point_cloud_copy.n_points == random_point_cloud.n_points
    np.testing.assert_array_equal(point_cloud_copy.points, random_point_cloud.points)
    # Ensure it's a deep copy
    point_cloud_copy.points[0] += 1.0
    assert not np.array_equal(point_cloud_copy.points, random_point_cloud.points)

def test_as_array(random_point_cloud):
    array = random_point_cloud.as_array()
    assert isinstance(array, np.ndarray)
    assert array.shape == (100, 3)
    np.testing.assert_array_equal(array, random_point_cloud.points)

def test_bounding_box(random_point_cloud):
    bbox = random_point_cloud.bounding_box()
    assert isinstance(bbox, tuple)
    assert len(bbox) == 2
    min_point, max_point = bbox
    assert min_point.shape == (3,)
    assert max_point.shape == (3,)
    np.testing.assert_array_equal(min_point, np.min(random_point_cloud.points, axis=0))
    np.testing.assert_array_equal(max_point, np.max(random_point_cloud.points, axis=0))

def test_concatenate(random_point_cloud, other_random_point_cloud):
    combined = random_point_cloud.concatenate(other_random_point_cloud)
    assert isinstance(combined, PointCloud3D)
    assert combined.n_points == random_point_cloud.n_points + other_random_point_cloud.n_points
    np.testing.assert_array_equal(combined.points[:random_point_cloud.n_points], random_point_cloud.points)
    np.testing.assert_array_equal(combined.points[random_point_cloud.n_points:], other_random_point_cloud.points)

def test_concatenate_inplace(random_point_cloud, other_random_point_cloud):
    original_n_points = random_point_cloud.n_points
    random_point_cloud.concatenate(other_random_point_cloud, inplace=True)
    assert random_point_cloud.n_points == original_n_points + other_random_point_cloud.n_points
    np.testing.assert_array_equal(random_point_cloud.points[original_n_points:], other_random_point_cloud.points)

def test_copy(random_point_cloud):
    point_cloud_copy = random_point_cloud.copy()
    assert isinstance(point_cloud_copy, PointCloud3D)
    assert point_cloud_copy.n_points == random_point_cloud.n_points
    np.testing.assert_array_equal(point_cloud_copy.points, random_point_cloud.points)
    # Ensure it's a deep copy
    point_cloud_copy.points[0] += 1.0
    assert not np.array_equal(point_cloud_copy.points, random_point_cloud.points)

def test_frame_transform(random_point_cloud, input_frame, output_frame):
    transformed = random_point_cloud.frame_transform(input_frame, output_frame)
    assert isinstance(transformed, PointCloud3D)
    assert transformed.n_points == random_point_cloud.n_points
    # Since the transformation is a translation by (1,2,3), check that
    expected_points = random_point_cloud.points - np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(transformed.points, expected_points)

def test_frame_transform_inplace(random_point_cloud, input_frame, output_frame):
    original_points = random_point_cloud.points.copy()
    random_point_cloud.frame_transform(input_frame, output_frame, inplace=True)
    expected_points = original_points - np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(random_point_cloud.points, expected_points)

def test_keep_points(random_point_cloud):
    # Select a subset of points to keep
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point_cloud.points[indices]
    
    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, 3) + 10])  # Points far away
    kept_cloud = PointCloud3D.from_array(kept_points)

    # Test that the kept cloud has the expected points
    extracted_cloud = random_point_cloud.keep_points(kept_cloud)
    np.testing.assert_array_equal(extracted_cloud.points, random_point_cloud.points[indices])
    assert extracted_cloud.n_points == 50

def test_keep_points_inplace(random_point_cloud):
    original_points = random_point_cloud.points.copy()

    # Select a subset of points to keep
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point_cloud.points[indices]
    
    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, 3) + 10])  # Points far away
    kept_cloud = PointCloud3D.from_array(kept_points)

    # Perform inplace operation
    random_point_cloud.keep_points(kept_cloud, inplace=True)
    np.testing.assert_array_equal(random_point_cloud.points, original_points[indices])
    assert random_point_cloud.n_points == 50

def test_keep_points_at(random_point_cloud):
    indices = np.random.choice(random_point_cloud.n_points, size=30, replace=False)
    kept_cloud = random_point_cloud.keep_points_at(indices)
    assert kept_cloud.n_points == 30
    np.testing.assert_array_equal(kept_cloud.points, random_point_cloud.points[indices])

def test_keep_points_at_inplace(random_point_cloud):
    indices = np.random.choice(random_point_cloud.n_points, size=30, replace=False)
    original_points = random_point_cloud.points.copy()
    random_point_cloud.keep_points_at(indices, inplace=True)
    assert random_point_cloud.n_points == 30
    np.testing.assert_array_equal(random_point_cloud.points, original_points[indices])

def test_merge(random_point_cloud):
    # Select a subset of points to merge
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point_cloud.points[indices]

    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, 3) + 10])  # Points far away
    kept_cloud = PointCloud3D.from_array(kept_points)
    merged_cloud = random_point_cloud.merge(kept_cloud)

    # The merged cloud should have the same points as the original since all kept points are already present
    np.testing.assert_array_equal(merged_cloud.points, np.vstack([random_point_cloud.points, kept_cloud.points[50:]]))
    assert merged_cloud.n_points == random_point_cloud.n_points + 10  # 10 new points added

def test_merge_inplace(random_point_cloud):
    # Select a subset of points to merge
    indices = np.arange(50)  # Keep first 50 points
    kept_points = random_point_cloud.points[indices]

    # Add some unused indices to test robustness
    kept_points = np.vstack([kept_points, np.random.rand(10, 3) + 10])  # Points far away
    kept_cloud = PointCloud3D.from_array(kept_points)
    original_n_points = random_point_cloud.n_points
    random_point_cloud.merge(kept_cloud, inplace=True)

    # The merged cloud should have the same points as the original since all kept points are already present
    assert random_point_cloud.n_points == original_n_points + 10  # 10 new points added
    np.testing.assert_array_equal(random_point_cloud.points[original_n_points:], kept_cloud.points[50:])

def test_remove_points(random_point_cloud):
    # Select a subset of points to remove
    indices = np.arange(50)  # Remove first 50 points
    removed_points = random_point_cloud.points[indices]
    
    # Add some unused indices to test robustness
    removed_points = np.vstack([removed_points, np.random.rand(10, 3) + 10])  # Points far away
    removed_cloud = PointCloud3D.from_array(removed_points)
    reduced_cloud = random_point_cloud.remove_points(removed_cloud)
    expected_points = random_point_cloud.points[50:]
    np.testing.assert_array_equal(reduced_cloud.points, expected_points)
    assert reduced_cloud.n_points == random_point_cloud.n_points - 50

def test_remove_points_inplace(random_point_cloud):
    # Select a subset of points to remove
    indices = np.arange(50)  # Remove first 50 points
    removed_points = random_point_cloud.points[indices]
    
    # Add some unused indices to test robustness
    removed_points = np.vstack([removed_points, np.random.rand(10, 3) + 10])  # Points far away
    removed_cloud = PointCloud3D.from_array(removed_points)
    original_n_points = random_point_cloud.n_points
    original_points = random_point_cloud.points.copy()
    random_point_cloud.remove_points(removed_cloud, inplace=True)
    expected_points = original_points[50:]
    np.testing.assert_array_equal(random_point_cloud.points, expected_points)
    assert random_point_cloud.n_points == original_n_points - 50

def test_remove_points_at(random_point_cloud):
    indices = np.arange(50)  # Remove first 50 points
    reduced_cloud = random_point_cloud.remove_points_at(indices)
    expected_points = random_point_cloud.points[50:]
    np.testing.assert_array_equal(reduced_cloud.points, expected_points)
    assert reduced_cloud.n_points == random_point_cloud.n_points - 50

def test_remove_points_at_inplace(random_point_cloud):
    indices = np.arange(50)  # Remove first 50 points
    original_n_points = random_point_cloud.n_points
    original_points = random_point_cloud.points.copy()
    random_point_cloud.remove_points_at(indices, inplace=True)
    expected_points = original_points[50:]
    np.testing.assert_array_equal(random_point_cloud.points, expected_points)
    assert random_point_cloud.n_points == original_n_points - 50

def test_unique():
    # Create a point cloud with duplicates
    array = np.array([[0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [3.0, 3.0, 3.0]])
    cloud_with_duplicates = PointCloud3D.from_array(array)
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
    cloud_with_duplicates = PointCloud3D.from_array(array)
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
def test_addition(random_point_cloud, other_random_point_cloud):
    combined = random_point_cloud + other_random_point_cloud
    assert isinstance(combined, PointCloud3D)
    assert combined.n_points == random_point_cloud.n_points + other_random_point_cloud.n_points
    np.testing.assert_array_equal(combined.points[:random_point_cloud.n_points], random_point_cloud.points)
    np.testing.assert_array_equal(combined.points[random_point_cloud.n_points:], other_random_point_cloud.points)

def test_inplace_addition(random_point_cloud, other_random_point_cloud):
    original_n_points = random_point_cloud.n_points
    random_point_cloud += other_random_point_cloud
    assert random_point_cloud.n_points == original_n_points + other_random_point_cloud.n_points
    np.testing.assert_array_equal(random_point_cloud.points[original_n_points:], other_random_point_cloud.points)

def test_len(random_point_cloud):
    assert len(random_point_cloud) == random_point_cloud.n_points

# ==========================================
# Visualization Tests
# ==========================================
def test_visualize(random_point_cloud):
    if DISPLAY:
        random_point_cloud.visualize(
            color="red",
            point_size=5.0,
        )