import pytest
import numpy
import pysdic

def test_pc_from_array():
    """Test creating a PointCloud from a NumPy array."""
    n_points = 100
    n_dimensions = 3
    points = numpy.random.rand(n_points, n_dimensions)
    pc = pysdic.PointCloud.from_array(points)
    assert pc.n_points == n_points
    assert pc.n_dimensions == n_dimensions
    numpy.testing.assert_array_equal(pc.points, points)
    
def test_pc_from_array_empty():
    """Test creating an empty PointCloud from a NumPy array."""
    points = numpy.empty((0, 3))
    pc = pysdic.PointCloud.from_array(points)
    assert pc.n_points == 0
    assert pc.n_dimensions == 3
    numpy.testing.assert_array_equal(pc.points, points)
    
def test_pc_from_array_with_properties():
    """Test creating a PointCloud from a NumPy array with additional properties."""
    n_points = 50
    n_dimensions = 3
    points = numpy.random.rand(n_points, n_dimensions)
    colors = numpy.random.randint(0, 256, size=(n_points, 3), dtype=numpy.uint8)
    intensities = numpy.random.rand(n_points)
    
    