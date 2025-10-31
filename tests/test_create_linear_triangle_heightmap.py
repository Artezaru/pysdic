import pytest
import numpy as np

from pysdic.geometry import create_linear_triangle_heightmap

import os
import sys
import cv2

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(test_dir)))

from tests.test_config import DISPLAY, TEXTURE  # noqa: E402

def test_create_linear_triangle_heightmap():
    surface_mesh = create_linear_triangle_heightmap(
        height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        n_x=50,
        n_y=50,
    )
    if DISPLAY:
        surface_mesh.visualize()
        pattern = cv2.imread(TEXTURE)
        surface_mesh.visualize_texture(pattern, show_edges=False)
        vertices_normals = surface_mesh.compute_vertices_normals()
        surface_mesh.set_vertices_property("normals", vertices_normals)
        surface_mesh.visualize_vertices_property("normals", property_axis=2)
        