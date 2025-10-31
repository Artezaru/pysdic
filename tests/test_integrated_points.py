# test_integration_points.py

import numpy as np
import pytest
from pysdic.geometry import IntegrationPoints


@pytest.fixture
def simple_integration_points():
    """Fixture to create a simple 2D triangle integration points instance."""
    natural_coords = np.array([
        [0.5, 0.5],
        [0.5, 0.0],
        [0.0, 0.5],
        [1/3, 1/3]
    ])
    element_indices = np.array([0, 0, 0, 1])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    return IntegrationPoints(natural_coords, element_indices, weights)


def test_creation(simple_integration_points):
    ip = simple_integration_points
    assert ip.n_points == 4
    assert ip.n_dimensions == 2
    assert ip.n_valids == 4
    np.testing.assert_array_equal(ip.element_indices, [0, 0, 0, 1])
    np.testing.assert_array_equal(ip.weights, [1.0, 2.0, 1.0, 1.0])
    np.testing.assert_array_equal(ip.natural_coordinates, np.array([[0.5,0.5],[0.5,0.0],[0.0,0.5],[1/3,1/3]]))


def test_add_points(simple_integration_points):
    ip = simple_integration_points
    new_coords = np.array([[0.2, 0.2], [0.1, 0.1]])
    new_ids = np.array([0, 1])
    new_weights = np.array([0.5, 0.5])
    ip.add_points(new_coords, new_ids, new_weights)
    assert ip.n_points == 6
    np.testing.assert_array_equal(ip.element_indices[-2:], [0, 1])
    np.testing.assert_array_equal(ip.weights[-2:], [0.5, 0.5])
    np.testing.assert_array_equal(ip.natural_coordinates[-2:], new_coords)


def test_remove_points(simple_integration_points):
    ip = simple_integration_points
    ip.remove_points(np.array([1, 3]))
    assert ip.n_points == 2
    np.testing.assert_array_equal(ip.element_indices, [0, 0])
    np.testing.assert_array_equal(ip.natural_coordinates, np.array([[0.5, 0.5], [0.0, 0.5]]))


def test_disable_points(simple_integration_points):
    ip = simple_integration_points
    ip.disable_points(np.array([0, 2]))
    assert ip.n_valids == 2
    np.testing.assert_array_equal(ip.element_indices, [-1, 0, -1, 1])
    np.testing.assert_array_equal(np.isnan(ip.natural_coordinates[[0,2], :]), [[True, True], [True, True]])


def test_remove_invalids(simple_integration_points):
    ip = simple_integration_points
    ip.disable_points(np.array([1,3]))
    ip.remove_invalids()
    assert ip.n_points == 2
    np.testing.assert_array_equal(ip.element_indices, [0, 0])


def test_copy_and_concat(simple_integration_points):
    ip1 = simple_integration_points
    ip2 = ip1.copy()
    ip2.add_points(np.array([[0.1, 0.1]]), np.array([0]), np.array([0.5]))
    ip_concat = ip1 + ip2
    assert ip_concat.n_points == ip1.n_points + ip2.n_points
    np.testing.assert_array_equal(ip_concat.natural_coordinates[:ip1.n_points], ip1.natural_coordinates)
    np.testing.assert_array_equal(ip_concat.element_indices[:ip1.n_points], ip1.element_indices)


def test_validation_checks():
    coords = np.array([[0.5, 0.5], [1.1, 0.0]])
    ids = np.array([0, 0])
    with pytest.raises(ValueError):
        IntegrationPoints(coords, ids)


def test_default_weights():
    coords = np.array([[0.3, 0.3]])
    ids = np.array([0])
    ip = IntegrationPoints(coords, ids)
    np.testing.assert_array_equal(ip.weights, [1.0])
