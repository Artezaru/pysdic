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

def test_segment_2_shape_functions():
    xi = numpy.array([-1.0, 0.0, 1.0, 1.5, numpy.nan])
    
    shape_functions, shape_function_derivatives = pysdic.segment_2_shape_functions(xi, return_derivatives=True)

    assert shape_functions.shape == (5, 2)
    assert shape_function_derivatives.shape == (5, 2, 1)

    assert numpy.all(shape_functions[3:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[3:, :, :] == 0.0)


def test_segment_2_nodal():
    xi = numpy.array([-1.0, 1.0])

    shape_functions = pysdic.segment_2_shape_functions(xi, return_derivatives=False)

    expected_shape_functions = numpy.eye(2)
    assert numpy.allclose(shape_functions, expected_shape_functions)


def test_segment_3_shape_functions():
    xi = numpy.array([-1.0, 0.0, 1.0, 1.5, numpy.nan])
    
    shape_functions, shape_function_derivatives = pysdic.segment_3_shape_functions(xi, return_derivatives=True)

    assert shape_functions.shape == (5, 3)
    assert shape_function_derivatives.shape == (5, 3, 1)

    assert numpy.all(shape_functions[3:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[3:, :, :] == 0.0)


def test_segment_3_nodal():
    xi = numpy.array([-1.0, 1.0, 0.0])

    shape_functions = pysdic.segment_3_shape_functions(xi, return_derivatives=False)

    expected_shape_functions = numpy.eye(3)
    assert numpy.allclose(shape_functions, expected_shape_functions)


def test_triangle_3_shape_functions():
    xi_eta = numpy.array([[0.3, 0.3], [0.5, 0.5], [0.6, 0.6], [-0.1, -0.1], [numpy.nan, numpy.nan]])
    
    shape_functions, shape_function_derivatives = pysdic.triangle_3_shape_functions(xi_eta, return_derivatives=True)

    assert shape_functions.shape == (5, 3)
    assert shape_function_derivatives.shape == (5, 3, 2)

    assert numpy.all(shape_functions[3:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[3:, :, :] == 0.0)


def test_triangle_3_nodal():
    xi_eta = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    shape_functions = pysdic.triangle_3_shape_functions(xi_eta, return_derivatives=False)

    expected_shape_functions = numpy.eye(3)
    assert numpy.allclose(shape_functions, expected_shape_functions)


def test_triangle_6_shape_functions():
    xi_eta = numpy.array([[0.2, 0.2], [0.5, 0.5], [0.6, 0.6], [-0.1, -0.1], [numpy.nan, numpy.nan]])
    
    shape_functions, shape_function_derivatives = pysdic.triangle_6_shape_functions(xi_eta, return_derivatives=True)

    assert shape_functions.shape == (5, 6)
    assert shape_function_derivatives.shape == (5, 6, 2)

    assert numpy.all(shape_functions[3:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[3:, :, :] == 0.0)


def test_triangle_6_nodal():
    xi_eta = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    shape_functions = pysdic.triangle_6_shape_functions(xi_eta, return_derivatives=False)

    expected_shape_functions = numpy.eye(6)
    assert numpy.allclose(shape_functions, expected_shape_functions)


def test_quadrangle_4_shape_functions():
    xi_eta = numpy.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5], [-1.5, -1.5], [numpy.nan, numpy.nan]])
    
    shape_functions, shape_function_derivatives = pysdic.quadrangle_4_shape_functions(xi_eta, return_derivatives=True)

    assert shape_functions.shape == (5, 4)
    assert shape_function_derivatives.shape == (5, 4, 2)

    assert numpy.all(shape_functions[2:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[2:, :, :] == 0.0)


def test_quadrangle_4_nodal():
    xi_eta = numpy.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    shape_functions = pysdic.quadrangle_4_shape_functions(xi_eta, return_derivatives=False)

    expected_shape_functions = numpy.eye(4)
    assert numpy.allclose(shape_functions, expected_shape_functions)


def test_quadrangle_8_shape_functions():
    xi_eta = numpy.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5], [-1.5, -1.5], [numpy.nan, numpy.nan]])
    
    shape_functions, shape_function_derivatives = pysdic.quadrangle_8_shape_functions(xi_eta, return_derivatives=True)

    assert shape_functions.shape == (5, 8)
    assert shape_function_derivatives.shape == (5, 8, 2)

    assert numpy.all(shape_functions[2:, :] == 0.0)
    assert numpy.all(shape_function_derivatives[2:, :, :] == 0.0)


def test_quadrangle_8_nodal():
    xi_eta = numpy.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                         [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    shape_functions = pysdic.quadrangle_8_shape_functions(xi_eta, return_derivatives=False)

    expected_shape_functions = numpy.eye(8)
    assert numpy.allclose(shape_functions, expected_shape_functions)