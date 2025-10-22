import pytest
import numpy as np

from pysdic.geometry import create_linear_triangle_axisymmetric

import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(test_dir)))
from tests.test_config import DISPLAY, TEXTURE  # noqa: E402

def test_create_linear_triangle_axisymmetric():
    cylinder_mesh = create_linear_triangle_axisymmetric(
        profile_curve=lambda z: 1.0,
        height_bounds=(-1.0, 1.0),
        theta_bounds=(-np.pi/4, np.pi/4),
        n_height=10,
        n_theta=20,
    )
    if DISPLAY:
        cylinder_mesh.visualize()


def test_create_linear_triangle_axisymmetric_closed():
    cylinder_mesh = create_linear_triangle_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(0.0, 2.0 * np.pi * (1 - 1.0 / 50)),
            n_height=10,
            n_theta=50,
            closed=True,
        )
    if DISPLAY:
        cylinder_mesh.visualize()