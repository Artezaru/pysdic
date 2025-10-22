import pytest
import numpy as np
import meshio

from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D
from py3dframe import Frame

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_config import DISPLAY

# ==========================================
# Fixture for creating a sample PointCloud3D
# ==========================================
@pytest.fixture
def tetrahedron():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]
    return PointCloud3D.from_array(points), cells

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

