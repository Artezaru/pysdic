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

from __future__ import annotations

from typing import Optional, Union
from numbers import Number, Integral

from fractions import Fraction

import math
import numpy
import scipy



def _gaussian_elimination(V: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    r"""
    Solve the linear system V c = b using Gaussian elimination with partial pivoting.

    .. warning::

        This is a private function and should not be used directly. Use the public functions that require solving linear systems instead.

    Parameters
    ----------
    V : :class:`list[list[Fraction]]`
        The coefficient matrix of the linear system.

    b : :class:`list[Fraction]`
        The right-hand side vector of the linear system.

    Returns
    -------
    :class:`list[Fraction]`
        The solution vector c of the linear system.
   
    """
    size = len(b)
    # Gaussian elimination with Fractions
    M = [row[:] + [b_i] for row, b_i in zip(V, b)]
    for i in range(size):
        # Pivoting
        pivot_row = None
        for r in range(i, size):
            if M[r][i] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular, cannot compute coefficients.")
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]

        # Normalize pivot row
        pivot = M[i][i]
        M[i] = [elem / pivot for elem in M[i]]

        # Eliminate below
        for r in range(size):
            if r == i:
                continue
            factor = M[r][i]
            if factor != 0:
                M[r] = [rv - factor * iv for rv, iv in zip(M[r], M[i])]

    # Back substitution
    coeffs = [M[i][-1] for i in range(size)]
    return coeffs




def compute_forward_finite_difference_coefficients(order: Integral, spacing: Number = 1.0, accuracy: Integral = 1) -> numpy.ndarray:
    r"""
    Compute the coefficients for the forward finite difference approximation of a derivative.

    The function returns the Stencil coefficients :math:`c_j` used to approximate the n-th order derivative of a function using forward finite differences.

    .. math::

        \frac{d^n f}{dt^n} \approx \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t + j h)

    where :math:`h` is the time step size, and :math:`m` is the accuracy order and :math:`N` is the number of stencil points required to achieve the desired accuracy (:math:`N = n + m - 1`), such that the approximation error is of order :math:`O(h^m)`.

    .. math ::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t + j h) = \frac{d^n f}{dt^n} + O(h^m)

    .. seealso::

        - :func:`compute_backward_finite_difference_coefficients` to compute the backward finite difference coefficients.
        - :func:`compute_central_finite_difference_coefficients` to compute the central finite difference coefficients.
        - :func:`assemble_forward_finite_difference_matrix` to assemble the forward finite difference operator matrix.

    .. note::

        The coefficients are computed using Gauss elimination on the Vandermonde matrix constructed from the Taylor series expansion.

        
    Parameters
    ----------   
    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.


    Returns
    -------
    :class:`numpy.ndarray`
        The coefficients for the forward finite difference approximation of the derivative in the form of a 1D array :obj:`[c_{th}, c_{(t+1)h}, ..., c_{(t+N)h}]`.


    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.

        
    Examples     
    ----------

    >>> compute_forward_finite_difference_coefficients(1, 1.0, 1)
    array([-1.,  1.])

    >>> compute_forward_finite_difference_coefficients(2, 0.1, 1)
    array([ 100., -200.,  100.])

    >>> compute_forward_finite_difference_coefficients(1, 1.0, 2)
    array([-1.5,  2., -0.5])

    >>> compute_forward_finite_difference_coefficients(2, 0.1, 2)
    array([ 200, -500, 400, -100])


    Mathematical Background
    -----------------------

    By developing the function :math:`f(t + k h)` in Taylor series around :math:`t`, we have:

    .. math::

        f(t + j h) = \sum_{k=0}^{\infty} \frac{(j h)^k}{k!} \frac{d^k f}{dt^k}

    We can add the contributions of the stencil points weighted by the coefficients :math:`c_k`:

    .. math::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t + j h) = \frac{1}{h^n} \sum_{j=0}^{N} c_j \sum_{k=0}^{\infty} \frac{(j h)^k}{k!} \frac{d^k f}{dt^k}

    .. math::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t + j h) = \sum_{k=0}^{\infty} \frac{h^{k-n}}{k!} \left(\sum_{j=0}^{N} c_j j^k\right) \frac{d^k f}{dt^k}

    Thus :math:`\forall k \in [0, N]`, we want to enforce the conditions:

    .. math::

        \sum_{j=0}^{N} c_j j^k = n! \delta_{k,n}

    where :math:`\delta_{k,n}` is the Kronecker delta.

    So we have a linear system of equations to solve for the coefficients :math:`c_j`. We can demonstrate that the matrix of the system is a Vandermonde matrix, which is invertible as long as the stencil points are distinct.
    And the output coefficients are divided by :math:`h^n` to account for the time step size. The result is a finite difference approximation of the n-th order derivative with an error of order :math:`O(h^m)`.
    
    """
    if not isinstance(order, Integral):
        raise TypeError("Order must be an integer.")
    if not isinstance(accuracy, Integral):
        raise TypeError("Accuracy must be an integer.")
    if not isinstance(spacing, Number):
        raise TypeError("Spacing must be a numeric type.")
    
    if order < 0:
        raise ValueError("Order must be a positive integer.")
    if accuracy <= 0:
        raise ValueError("Accuracy must be a positive integer.")
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")
    
    N = order + accuracy - 1  # Number of stencil points
    size = N + 1

    # Construct the Vandermonde matrix
    V = [[Fraction(j ** k) for j in range(size)] for k in range(size)]
    b = [Fraction(math.factorial(order)) if k == order else 0 for k in range(size)]

    # Gaussian elimination with Fractions
    coeffs = _gaussian_elimination(V, b)
    return numpy.array(coeffs, dtype=float) / (spacing**order)




def compute_backward_finite_difference_coefficients(order: Integral, spacing: Number = 1.0, accuracy: Integral = 1) -> numpy.ndarray:
    r"""
    Compute the coefficients for the backward finite difference approximation of a derivative.

    The function returns the Stencil coefficients :math:`c_j` used to approximate the n-th order derivative of a function using backward finite differences.

    .. math::

        \frac{d^n f}{dt^n} \approx \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t - j h)

    where :math:`h` is the time step size, and :math:`m` is the accuracy order and :math:`N` is the number of stencil points required to achieve the desired accuracy (:math:`N = n + m - 1`), such that the approximation error is of order :math:`O(h^m)`.

    .. math ::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t - j h) = \frac{d^n f}{dt^n} + O(h^m)

    .. seealso::

        - :func:`compute_forward_finite_difference_coefficients` to compute the forward finite difference coefficients.
        - :func:`compute_central_finite_difference_coefficients` to compute the central finite difference coefficients.
        - :func:`assemble_backward_finite_difference_matrix` to assemble the backward finite difference operator matrix.

    .. note::

        The coefficients are computed using Gauss elimination on the Vandermonde matrix constructed from the Taylor series expansion.

        
    Parameters
    ----------   
    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.


    Returns
    -------
    :class:`numpy.ndarray`
        The coefficients for the backward finite difference approximation of the derivative in the form of a 1D array :obj:`[c_{th}, c_{(t-1)h}, ..., c_{(t-N)h}]`.


    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.

        
    Examples     
    ----------

    >>> compute_backward_finite_difference_coefficients(1, 1.0, 1)
    array([ 1., -1.])

    >>> compute_backward_finite_difference_coefficients(2, 1.0, 1)
    array([ 1., -2.,  1.])

    >>> compute_backward_finite_difference_coefficients(1, 1.0, 2)
    array([ 1.5, -2.,  0.5])

    >>> compute_backward_finite_difference_coefficients(2, 0.1, 2)
    array([ 200, -500, 400, -100])


    Mathematical Background
    -----------------------

    By developing the function :math:`f(t - k h)` in Taylor series around :math:`t`, we have:

    .. math::

        f(t - j h) = \sum_{k=0}^{\infty} \frac{(-j h)^k}{k!} \frac{d^k f}{dt^k}

    We can add the contributions of the stencil points weighted by the coefficients :math:`c_k`:

    .. math::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t - j h) = \frac{1}{h^n} \sum_{j=0}^{N} c_j \sum_{k=0}^{\infty} \frac{(-j h)^k}{k!} \frac{d^k f}{dt^k}

    .. math::

        \frac{1}{h^n} \sum_{j=0}^{N} c_j f(t - j h) = \sum_{k=0}^{\infty} \frac{h^{k-n}}{k!} \left(\sum_{j=0}^{N} c_j (-j)^k\right) \frac{d^k f}{dt^k}


    Thus :math:`\forall k \in [0, N]`, we want to enforce the conditions:

    .. math::

        \sum_{j=0}^{N} c_j (-j)^k = n! \delta_{k,n}

    where :math:`\delta_{k,n}` is the Kronecker delta.

    So we have a linear system of equations to solve for the coefficients :math:`c_j`. We can demonstrate that the matrix of the system is a Vandermonde matrix, which is invertible as long as the stencil points are distinct.
    And the output coefficients are divided by :math:`h^n` to account for the time step size. The result is a finite difference approximation of the n-th order derivative with an error of order :math:`O(h^m)`.

    The computation is similar to the one for the forward finite difference coefficients but foro odd orders, the sign of the coefficients is also inverted.
    
    """
    return (1 - 2 * (order % 2)) * compute_forward_finite_difference_coefficients(order, spacing, accuracy)




def compute_central_finite_difference_coefficients(order: Integral, spacing: Number = 1.0, accuracy: Integral = 2) -> numpy.ndarray:
    r"""
    Compute the coefficients for the central finite difference approximation of a derivative.

    The function returns the Stencil coefficients :math:`c_j` used to approximate the n-th order derivative of a function using central finite differences.

    .. math::

        \frac{d^n f}{dt^n} \approx \frac{1}{h^n} \sum_{j=-M}^{M} c_j f(t + j h)

    where :math:`h` is the time step size, and :math:`m` is the accuracy order and :math:`N` is the number of stencil points required to achieve the desired accuracy (:math:`N = n + m - 1`), such that the approximation error is of order :math:`O(h^m)`.
    Here, :math:`M = \frac{N}{2}` if :math:`N` is even, and :math:`M = \frac{N-1}{2}` if :math:`N` is odd.

    .. math ::

        \frac{1}{h^n} \sum_{j=-M}^{M} c_j f(t + j h) = \frac{d^n f}{dt^n} + O(h^m)

    .. seealso::

        - :func:`compute_forward_finite_difference_coefficients` to compute the forward finite difference coefficients.
        - :func:`compute_backward_finite_difference_coefficients` to compute the backward finite difference coefficients.
        - :func:`assemble_central_finite_difference_matrix` to assemble the central finite difference operator matrix.

    .. note::

        The coefficients are computed using Gauss elimination on the Vandermonde matrix constructed from the Taylor series expansion.

        
    Parameters
    ----------   
    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Must be a even integer. Default is :obj:`2`.


    Returns
    -------
    :class:`numpy.ndarray`
        The coefficients for the central finite difference approximation of the derivative in the form of a 1D array :obj:`[c_{-Mh}, ..., c_{-h}, c_{0}, c_{h}, ..., c_{Mh}]`.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the accuracy is not an even integer.
        If the spacing is not a strictly positive number.

        
    Examples
    --------

    >>> compute_central_finite_difference_coefficients(1, 1.0, 2)
    array([-0.5,  0. ,  0.5])

    >>> compute_central_finite_difference_coefficients(2, 1.0, 2)
    array([ 1., -2.,  1.])

    >>> compute_central_finite_difference_coefficients(1, 1.0, 4)
    array([ 1/12, -2/3, 0, 2/3, -1/12])

    >>> compute_central_finite_difference_coefficients(2, 0.1, 4)
    array([  -100/12, 400/3, -500/2, 400/3,   -100/12])


    Mathematical Background
    -----------------------

    By developing the function :math:`f(t + k h)` in Taylor series around :math:`t`, we have:

    .. math::

        f(t + j h) = \sum_{k=0}^{\infty} \frac{(j h)^k}{k!} \frac{d^k f}{dt^k}

    We can add the contributions of the stencil points weighted by the coefficients :math:`c_k`:

    .. math::

        \frac{1}{h^n} \sum_{j=-M}^{M} c_j f(t + j h) = \frac{1}{h^n} \sum_{j=-M}^{M} c_j \sum_{k=0}^{\infty} \frac{(j h)^k}{k!} \frac{d^k f}{dt^k}

    .. math::

        \frac{1}{h^n} \sum_{j=-M}^{M} c_j f(t + j h) = \sum_{k=0}^{\infty} \frac{h^{k-n}}{k!} \left(\sum_{j=-M}^{M} c_j j^k\right) \frac{d^k f}{dt^k}

    Thus :math:`\forall k \in [0, N]`, we want to enforce the conditions:

    .. math::

        \sum_{j=-M}^{M} c_j j^k = n! \delta_{k,n}

    where :math:`\delta_{k,n}` is the Kronecker delta.

    So we have a linear system of equations to solve for the coefficients :math:`c_j`. We can demonstrate that the matrix of the system is a Vandermonde matrix, which is invertible as long as the stencil points are distinct if we consider symmetry/antisymmetry of the coefficients depending on the order.
    And the output coefficients are divided by :math:`h^n` to account for the time step size. The result is a finite difference approximation of the n-th order derivative with an error of order :math:`O(h^m)`.
    
    """
    if not isinstance(order, Integral):
        raise TypeError("Order must be an integer.")
    if not isinstance(accuracy, Integral):
        raise TypeError("Accuracy must be an integer.")
    if not isinstance(spacing, Number):
        raise TypeError("Spacing must be a numeric type.")
    
    if order < 0:
        raise ValueError("Order must be a positive integer.")
    if accuracy <= 0:
        raise ValueError("Accuracy must be a positive integer.")
    if accuracy % 2 != 0:
        raise ValueError("Accuracy must be an even integer for central finite differences.")
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")

    N = order + accuracy - 1  # Number of stencil points
    size = N + 1
    M = N // 2

    even = (order % 2 == 0)

    if even:   
        # EVEN derivative → symmetric → include j=0
        # k = 0,2,4,... ; j = 0..M
        V = [[Fraction((1 if j == 0 else 2) * (j**k)) for j in range(0, M+1)] for k in range(0, order+accuracy, 2)]
        b = [Fraction(math.factorial(order) if k == order else 0) for k in range(0, order+accuracy, 2)]

    else:      
        # ODD derivative → antisymmetric → j = 1..M only
        # k = 1,3,5,... ; j = 1..M
        V = [[Fraction(2 * (j**k)) for j in range(1, M+1)] for k in range(1, order+accuracy, 2)]
        b = [Fraction(math.factorial(order) if k == order else 0) for k in range(1, order+accuracy, 2)]

    # Gaussian elimination with Fractions
    coeffs = _gaussian_elimination(V, b)

    if even:
        # coeffs = [c0, c1, c2, ..., cM] → build full stencil
        coeffs = coeffs[:0:-1] + coeffs  # [cM, ..., c2, c1, c0, c1, c2, ..., cM]
    else:
        # coeffs = [c1, c2, ..., cM] → build full stencil
        coeffs = [-c for c in coeffs[::-1]] + [0] + coeffs  # [ -cM, ..., -c2, -c1, 0, c1, c2, ..., cM]

    return numpy.array(coeffs, dtype=float) / (spacing**order)







def apply_forward_finite_difference(
    data: numpy.ndarray,
    order: Integral,
    axis: Optional[Integral] = -1,
    spacing: Number = 1.0,
    accuracy: Integral = 1,
    mode: str = 'reflect',
    value: Number = 0.0,
) -> numpy.ndarray:
    r"""
    Apply the forward finite difference operator to a time series data array along a specified axis.

    A convolution operation is performed between the input data and the forward finite difference kernel
    to approximate the temporal derivative of the specified order.

    The available modes for handling borders are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.

    +-----------------+----------------------------------------------------------------+
    | Mode            | Description                                                    |
    +=================+================================================================+
    | 'reflect'       | :math:`(d c b a | a b c d | d c b a)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'constant'      | :math:`(k k k k | a b c d | k k k k)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'nearest'       | :math:`(a a a a | a b c d | d d d d)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'mirror'        | :math:`(d c b | a b c d | c b a)`                              |
    +-----------------+----------------------------------------------------------------+
    | 'wrap'          | :math:`(a b c d | a b c d | a b c d)`                          |
    +-----------------+----------------------------------------------------------------+

    .. seealso::

        - :func:`compute_forward_finite_difference_coefficients` to compute the forward finite difference coefficients.
        - :func:`apply_backward_finite_difference` to apply the backward finite difference operator.
        - :func:`apply_central_finite_difference` to apply the central finite difference operator.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        The input time series data array.

    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    axis : :class:`int`, optional
        The axis along which to apply the finite difference operator. Default is :obj:`-1`.

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.

    mode : :class:`str`, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
        Default is :obj:`'reflect'`.

    value : :class:`Number`, optional
        The value to use for padding when :obj:`mode` is set to :obj:`'constant'`. Default is :obj:`0.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        The resulting array after applying the forward finite difference operator with the same shape as the input data.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.
        If the mode is not a valid string option for :func:`scipy.ndimage.correlate1d`.


    Examples
    --------

    >>> data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> apply_forward_finite_difference(data, order=1, spacing=1.0, accuracy=1)
    array([1., 3., 5., 7., 0.])

    >>> data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> apply_forward_finite_difference(data, order=2, spacing=1.0, accuracy=1)
    array([2., 2., 2., -7, -7])
    
        
    """
    data = numpy.asarray(data)
    if not numpy.issubdtype(data.dtype, numpy.floating):
        data = data.astype(numpy.float64)

    if not isinstance(axis, Integral):
        raise TypeError("Axis must be an integer.")
    if axis < -data.ndim or axis >= data.ndim:
        raise ValueError("Axis is out of bounds for the input data array.")
    
    if not isinstance(mode, str):
        raise TypeError("Mode must be a string.")
    valid_modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}.")
    
    if not isinstance(value, Number):
        raise TypeError("Value must be a numeric type.")
    
    coeffs = compute_forward_finite_difference_coefficients(order, spacing, accuracy)

    # Shift coefficients for convolution
    # For forward difference, the first coefficient corresponds to the current point
    if len(coeffs) % 2 == 0:
        origin = -(len(coeffs) // 2)
    else:
        origin = -(len(coeffs) - 1) // 2

    return scipy.ndimage.correlate1d(data, coeffs, axis=axis, mode=mode, cval=value, origin=origin)



def apply_backward_finite_difference(
    data: numpy.ndarray,
    order: Integral,
    axis: Optional[Integral] = -1,
    spacing: Number = 1.0,
    accuracy: Integral = 1,
    mode: str = 'reflect',
    value: Number = 0.0,
) -> numpy.ndarray:
    r"""
    Apply the backward finite difference operator to a time series data array along a specified axis.

    A convolution operation is performed between the input data and the backward finite difference kernel
    to approximate the temporal derivative of the specified order.

    The available modes for handling borders are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.

    +-----------------+----------------------------------------------------------------+
    | Mode            | Description                                                    |
    +=================+================================================================+
    | 'reflect'       | :math:`(d c b a | a b c d | d c b a)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'constant'      | :math:`(k k k k | a b c d | k k k k)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'nearest'       | :math:`(a a a a | a b c d | d d d d)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'mirror'        | :math:`(d c b | a b c d | c b a)`                              |
    +-----------------+----------------------------------------------------------------+
    | 'wrap'          | :math:`(a b c d | a b c d | a b c d)`                          |
    +-----------------+----------------------------------------------------------------+

    .. seealso::

        - :func:`compute_backward_finite_difference_coefficients` to compute the backward finite difference coefficients.
        - :func:`apply_forward_finite_difference` to apply the forward finite difference operator.
        - :func:`apply_central_finite_difference` to apply the central finite difference operator.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        The input time series data array.

    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    axis : :class:`int`, optional
        The axis along which to apply the finite difference operator. Default is :obj:`-1`.

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.

    mode : :class:`str`, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
        Default is :obj:`'reflect'`.

    value : :class:`Number`, optional
        The value to use for padding when :obj:`mode` is set to :obj:`'constant'`. Default is :obj:`0.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        The resulting array after applying the backward finite difference operator with the same shape as the input data.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.
        If the mode is not a valid string option for :func:`scipy.ndimage.correlate1d`.


    Examples
    --------

    >>> data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> apply_backward_finite_difference(data, order=1, spacing=1.0, accuracy=1)
    array([ 0.,  1.,  3.,  5.,  7.])

    >>> data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> apply_backward_finite_difference(data, order=2, spacing=1.0, accuracy=1)
    array([ 1.,  1.,  2.,  2., 2])
        
    """
    data = numpy.asarray(data)
    if not numpy.issubdtype(data.dtype, numpy.floating):
        data = data.astype(numpy.float64)

    if not isinstance(axis, Integral):
        raise TypeError("Axis must be an integer.")
    if axis < -data.ndim or axis >= data.ndim:
        raise ValueError("Axis is out of bounds for the input data array.")
    
    if not isinstance(mode, str):
        raise TypeError("Mode must be a string.")
    valid_modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}.")
    
    if not isinstance(value, Number):
        raise TypeError("Value must be a numeric type.")

    coeffs = compute_backward_finite_difference_coefficients(order, spacing, accuracy)

    # Inverse the order of coefficients for backward difference
    coeffs = coeffs[::-1]

    # Shift coefficients for convolution
    # For backward difference, the last coefficient corresponds to the current point
    if len(coeffs) % 2 == 0:
        origin = len(coeffs) // 2 - 1
    else:
        origin = (len(coeffs) - 1) // 2

    return scipy.ndimage.correlate1d(data, coeffs, axis=axis, mode=mode, cval=value, origin=origin)



def apply_central_finite_difference(
    data: numpy.ndarray,
    order: Integral,
    axis: Optional[Integral] = -1,
    spacing: Number = 1.0,
    accuracy: Integral = 2,
    mode: str = 'reflect',
    value: Number = 0.0,
) -> numpy.ndarray:
    r"""
    Apply the central finite difference operator to a time series data array along a specified axis.

    A convolution operation is performed between the input data and the central finite difference kernel
    to approximate the temporal derivative of the specified order.

    The available modes for handling borders are: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.

    +-----------------+----------------------------------------------------------------+
    | Mode            | Description                                                    |
    +=================+================================================================+
    | 'reflect'       | :math:`(d c b a | a b c d | d c b a)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'constant'      | :math:`(k k k k | a b c d | k k k k)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'nearest'       | :math:`(a a a a | a b c d | d d d d)`                          |
    +-----------------+----------------------------------------------------------------+
    | 'mirror'        | :math:`(d c b | a b c d | c b a)`                              |
    +-----------------+----------------------------------------------------------------+
    | 'wrap'          | :math:`(a b c d | a b c d | a b c d)`                          |
    +-----------------+----------------------------------------------------------------+

    .. seealso::

        - :func:`compute_central_finite_difference_coefficients` to compute the central finite difference coefficients.
        - :func:`apply_forward_finite_difference` to apply the forward finite difference operator.
        - :func:`apply_backward_finite_difference` to apply the backward finite difference operator.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        The input time series data array.

    order : :class:`int`
        The order of the derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    axis : :class:`int`, optional
        The axis along which to apply the finite difference operator. Default is :obj:`-1`.

    spacing : :class:`Number`, optional
        The time step size :math:`h`. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Must be a even integer. Default is :obj:`2`.

    mode : :class:`str`, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
        Default is :obj:`'reflect'`.

    value : :class:`Number`, optional
        The value to use for padding when :obj:`mode` is set to :obj:`'constant'`. Default is :obj:`0.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        The resulting array after applying the central finite difference operator with the same shape as the input data.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the accuracy is not an even integer.
        If the spacing is not a strictly positive number.
        If the mode is not a valid string option for :func:`scipy.ndimage.correlate1d`.


    Examples
    --------

    >>> data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    >>> apply_central_finite_difference(data, order=1, spacing=1.0, accuracy=2)
    array([0.5, 2., 4., 6., 3.5])
        
    """
    data = numpy.asarray(data)
    if not numpy.issubdtype(data.dtype, numpy.floating):
        data = data.astype(numpy.float64)

    if not isinstance(axis, Integral):
        raise TypeError("Axis must be an integer.")
    if axis < -data.ndim or axis >= data.ndim:
        raise ValueError("Axis is out of bounds for the input data array.")

    if not isinstance(mode, str):
        raise TypeError("Mode must be a string.")
    valid_modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}.")

    if not isinstance(value, Number):
        raise TypeError("Value must be a numeric type.")

    coeffs = compute_central_finite_difference_coefficients(order, spacing, accuracy)
    return scipy.ndimage.correlate1d(data, coeffs, axis=axis, mode=mode, cval=value)





def _assemble_toeplitz_matrix(
    semi_row: numpy.ndarray,
    semi_col: numpy.ndarray,
    n_times: Integral,
    n_dim: Integral,
    sparse: bool,
    weights: Optional[numpy.ndarray] = None,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the Toeplitz matrix given the semi-row and semi-column vectors.

    .. warning::

        This is a private function and should not be used directly. Use the specific finite difference assembly functions instead.

    Parameters
    ----------
    semi_row : :class:`numpy.ndarray`
        The non-zero first row vector of the Toeplitz matrix.
    
    semi_col : :class:`numpy.ndarray`
        The non-zero first column vector of the Toeplitz matrix.

    n_times : :class:`int`
        The number of time steps in the time series.

    n_dim : :class:`int`
        The number of spatial dimensions of the data at each time step.

    sparse : :class:`bool`
        Whether to return a sparse matrix.

    weights : :class:`numpy.ndarray`, optional
        An optional array of weights to apply for the identity matrix in the kronecker product. If None, the identity matrix is used. If provided, it must have shape :obj:`(n_dim,)` to weight each spatial dimension accordingly (diagonal coefficients) or :obj:`(n_dim, n_dim)` to provide a full weighting matrix. Default is :obj:`None`.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The assembled Toeplitz matrix with shape :obj:`(n_times * n_dim, n_times * n_dim)`.
    """
    if weights is None:
        if sparse:
            weights = scipy.sparse.eye(n_dim)
        else:
            weights = numpy.eye(n_dim)
    else:
        weights = numpy.asarray(weights)
        if weights.ndim == 1:
            if weights.shape[0] != n_dim:
                raise ValueError("Weights array must have shape (n_dim,) for diagonal weighting.")
            if sparse:
                weights = scipy.sparse.diags(weights)
            else:
                weights = numpy.diag(weights)
        elif weights.ndim == 2:
            if weights.shape != (n_dim, n_dim):
                raise ValueError("Weights array must have shape (n_dim, n_dim) for full weighting.")
        else:
            raise ValueError("Weights array must be either 1D or 2D.")

    length_col = min(len(semi_col), n_times)
    length_row = min(len(semi_row), n_times)

    # Sparse version
    if sparse:
        diagonals = []
        offsets = []

        # Upper diagonals (including main diagonal)
        for k in range(length_row):
            diagonals.append(numpy.full(n_times - k, semi_row[k]))
            offsets.append(k)

        # Lower diagonals
        for k in range(1, length_col):
            diagonals.append(numpy.full(n_times - k, semi_col[k]))
            offsets.append(-k)

        # Create the sparse Toeplitz matrix
        T = scipy.sparse.diags(diagonals, offsets, shape=(n_times, n_times), format='csr')

        # Kronecker product with identity matrix for spatial dimensions
        D = scipy.sparse.kron(T, weights, format='csr')

    # Dense version
    else:
        # Create the rows and columns of the Toeplitz matrix
        first_row = numpy.zeros(n_times)
        first_row[0: length_row] = semi_row[0: length_row]
        first_col = numpy.zeros(n_times)
        first_col[0: length_col] = semi_col[0: length_col]

        # Create the Toeplitz matrix
        T = scipy.linalg.toeplitz(first_col, first_row)

        # Kronecker product with identity matrix for spatial dimensions
        D = numpy.kron(T, weights)

    return D






def assemble_forward_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    n_dim: Optional[Integral] = 1,
    spacing: Number = 1.0,
    accuracy: Integral = 1,
    sparse: bool = False,
    weights: Optional[numpy.ndarray] = None,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the temporal derivation operator matrix for finite difference approximation with forward scheme.

    The operator matrix is used to approximate the temporal derivative of a time-dependent function
    using forward finite differences.

    For a time series with :math:`N_t` time steps, the operator matrix :math:`D` for the :math:`n`-th order derivative is given by the toeplitz matrix:

    .. math::

        D = \begin{bmatrix}
        a_t & a_{t+1} & a_{t+2} & \cdots & a_{t+N} \\
        0 & a_t & a_{t+1} & \cdots & a_{t+N-1} \\
        0 & 0 & a_t & \cdots & a_{t+N-2} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t} \\
        \end{bmatrix}

    where :math:`a_k` are the Stencil coefficients from the forward finite difference kernel at the given order, spacing and accuracy.

    The dimension :obj:`n_dim` allows to apply the operator to multi-dimensional data at each time step :math:`f = [f_1(t), f_2(t), \ldots, f_{n_{dim}}(t)]^T`.

    Thus the kronecker product with the identity matrix is performed to account for the spatial dimensions.

    .. math::

        D = T \otimes I_{n_{dim}} = \begin{bmatrix}
        a_t I_{n_{dim}} & a_{t+1} I_{n_{dim}} & a_{t+2} I_{n_{dim}} & \cdots & a_{t+N} I_{n_{dim}} \\
        0 & a_t I_{n_{dim}} & a_{t+1} I_{n_{dim}} & \cdots & a_{t+N-1} I_{n_{dim}} \\
        0 & 0 & a_t I_{n_{dim}} & \cdots & a_{t+N-2} I_{n_{dim}} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t} I_{n_{dim}} \\
        \end{bmatrix}


    The operator matrix is then constructed such as :math:`\frac{\partial^n f}{\partial t^n} \approx  D @ F`, where :math:`F` is the flattened time series data :math:`F = [f_1(t_1), f_2(t_1), \ldots, f_{n_{dim}}(t_1), f_1(t_2), f_2(t_2), \ldots, f_{n_{dim}}(t_2), \ldots, f_{n_{dim}}(t_{N_t})]^T`.

    .. seealso::

        - :func:`compute_forward_finite_difference_coefficients` to compute the forward finite difference coefficients for given order, spacing and accuracy.
        - :func:`assemble_backward_finite_difference_matrix` to assemble the backward finite difference operator matrix.
        - :func:`assemble_central_finite_difference_matrix` to assemble the central finite difference operator matrix.

    .. note::

        If the time series has less time steps than the number of stencil points required for the given order and accuracy, the operator matrix will be truncated accordingly.

    Parameters
    ----------
    order : :class:`int`
        The order of the temporal derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    n_times : :class:`int`
        The number of time steps in the time series.

    n_dim : :class:`int`, optional
        The number of spatial dimensions of the data at each time step. Default is :obj:`1`.

    spacing : :class:`Number`, optional
        The time step size. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.

    sparse : :class:`bool`, optional
        Whether to return a sparse matrix. Default is :obj:`False`.

    weights : :class:`numpy.ndarray`, optional
        An optional array of weights to apply for the identity matrix in the kronecker product. If None, the identity matrix is used. If provided, it must have shape :obj:`(n_dim,)` to weight each spatial dimension accordingly (diagonal coefficients) or :obj:`(n_dim, n_dim)` to provide a full weighting matrix. Default is :obj:`None`.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The temporal derivation operator matrix with shape :obj:`(n_times * n_dim, n_times * n_dim)`.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.

    
    Examples
    --------
    >>> assemble_forward_finite_difference_matrix(2, 5, spacing=1.0, accuracy=1)
    array([[ 1., -2.,  1.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  1., -2.,  1.],
           [ 0.,  0.,  0.,  1., -2.],
           [ 0.,  0.,  0.,  0.,  1.]])

    >>> assemble_forward_finite_difference_matrix(1, 4, n_dim=2, spacing=0.1, accuracy=2)
    array([[-15.,  0.,  20.,  0., -5.,  0.,  0.,  0.],
           [  0.,-15.,   0., 20.,  0., -5.,  0.,  0.],
           [  0.,  0., -15.,  0., 20.,  0., -5.,  0.],
           [  0.,  0.,   0.,-15.,  0., 20.,  0., -5.],
           [  0.,  0.,   0.,  0.,-15.,  0., 20.,  0.],
           [  0.,  0.,   0.,  0.,  0.,-15.,  0., 20.],
           [  0.,  0.,   0.,  0.,  0.,  0.,-15.,  0.],
           [  0.,  0.,   0.,  0.,  0.,  0.,  0.,-15.]])

    """
    if not isinstance(n_times, Integral):
        raise TypeError("Number of time steps must be an integer.")
    if n_times <= 0:
        raise ValueError("Number of time steps must be a positive integer.")
    if not isinstance(n_dim, Integral):
        raise TypeError("Number of dimensions must be an integer.")
    if n_dim <= 0:
        raise ValueError("Number of dimensions must be a positive integer.")
    if not isinstance(order, Integral):
        raise TypeError("Order must be an integer.")
    if not isinstance(accuracy, Integral):
        raise TypeError("Accuracy must be an integer.")
    if not isinstance(spacing, Number):
        raise TypeError("Spacing must be a numeric type.")
    if not isinstance(sparse, bool):
        raise TypeError("Sparse flag must be a boolean.")
    if order < 0:
        raise ValueError("Order must be a positive integer.")
    if accuracy <= 0:
        raise ValueError("Accuracy must be a positive integer.")
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")
    
    # Forward FD coefficients
    coeffs = compute_forward_finite_difference_coefficients(order, spacing, accuracy)

    # Create the non-zero first columns of the Toeplitz matrix
    col = numpy.array([coeffs[0]])

    # Create the non-zero first rows of the Toeplitz matrix
    row = coeffs

    return _assemble_toeplitz_matrix(row, col, n_times, n_dim, sparse, weights=weights)



def assemble_backward_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    n_dim: Optional[Integral] = 1,
    spacing: Number = 1.0,
    accuracy: Integral = 1,
    sparse: bool = False,
    weights: Optional[numpy.ndarray] = None,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the temporal derivation operator matrix for finite difference approximation with backward scheme.

    The operator matrix is used to approximate the temporal derivative of a time-dependent function
    using backward finite differences.

    For a time series with :math:`N_t` time steps, the operator matrix :math:`D` for the :math:`n`-th order derivative is given by the toeplitz matrix:

    .. math::

        D = \begin{bmatrix}
        a_t & 0 & 0 & \cdots & 0 \\
        a_{t-1} & a_t & 0 & \cdots & 0 \\
        0 & a_{t-1} & a_t & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t} \\
        \end{bmatrix}

    where :math:`a_k` are the Stencil coefficients from the backward finite difference kernel at the given order, spacing and accuracy.

    The dimension :obj:`n_dim` allows to apply the operator to multi-dimensional data at each time step :math:`f = [f_1(t), f_2(t), \ldots, f_{n_{dim}}(t)]^T`.

    Thus the kronecker product with the identity matrix is performed to account for the spatial dimensions.

    .. math::

        D = T \otimes I_{n_{dim}} = \begin{bmatrix}
        a_t I_{n_{dim}} & 0 & 0 & \cdots & 0 \\
        a_{t-1} I_{n_{dim}} & a_t I_{n_{dim}} & 0 & \cdots & 0 \\
        0 & a_{t-1} I_{n_{dim}} & a_t I_{n_{dim}} & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t} I_{n_{dim}} \\
        \end{bmatrix}

    The operator matrix is then constructed such as :math:`\frac{\partial^n f}{\partial t^n} \approx  D @ F`, where :math:`F` is the flattened time series data :math:`F = [f_1(t_1), f_2(t_1), \ldots, f_{n_{dim}}(t_1), f_1(t_2), f_2(t_2), \ldots, f_{n_{dim}}(t_2), \ldots, f_{n_{dim}}(t_{N_t})]^T`.

    .. seealso::

        - :func:`compute_backward_finite_difference_coefficients` to compute the backward finite difference coefficients for given order, spacing and accuracy.
        - :func:`assemble_forward_finite_difference_matrix` to assemble the forward finite difference operator matrix.
        - :func:`assemble_central_finite_difference_matrix` to assemble the central finite difference operator matrix.

    .. note::
    
        If the time series has less time steps than the number of stencil points required for the given order and accuracy, the operator matrix will be truncated accordingly.

    Parameters
    ----------
    order : :class:`int`
        The order of the temporal derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    n_times : :class:`int`
        The number of time steps in the time series.

    n_dim : :class:`int`, optional
        The number of spatial dimensions of the data at each time step. Default is :obj:`1`.

    spacing : :class:`Number`, optional
        The time step size. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Default is :obj:`1`.

    sparse : :class:`bool`, optional
        Whether to return a sparse matrix. Default is :obj:`False`.
    
    weights : :class:`numpy.ndarray`, optional
        An optional array of weights to apply for the identity matrix in the kronecker product. If None, the identity matrix is used. If provided, it must have shape :obj:`(n_dim,)` to weight each spatial dimension accordingly (diagonal coefficients) or :obj:`(n_dim, n_dim)` to provide a full weighting matrix. Default is :obj:`None`.


    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The temporal derivation operator matrix with shape :obj:`(n_times * n_dim, n_times * n_dim)`.


    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the spacing is not a strictly positive number.

        
    Examples
    --------

    >>> assemble_backward_finite_difference_matrix(2, 5, spacing=1.0, accuracy=1)
    array([[ 1.,  0.,  0.,  0.,  0.],
           [-2.,  1.,  0.,  0.,  0.],
           [ 1., -2.,  1.,  0.,  0.],
           [ 0.,  1., -2.,  1.,  0.],
           [ 0.,  0.,  1., -2.,  1.]])

    >>> assemble_backward_finite_difference_matrix(1, 4, n_dim=2, spacing=0.1, accuracy=2)
    array([[ 15.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [  0., 15.,  0.,  0.,  0.,  0.,  0.,  0.],
           [-20.,  0., 15.,  0.,  0.,  0.,  0.,  0.],
           [  0., -20.,  0., 15.,  0.,  0.,  0.,  0.],
           [  5.,  0., -20.,  0., 15.,  0.,  0.,  0.],
           [  0.,  5.,   0., -20.,  0.,15.,  0.,  0.],
           [  0.,  0.,   5.,   0., -20., 0.,15.,  0.],
           [  0.,  0.,   0.,   5.,   0.,-20., 0.,15.]])
    
    """
    if not isinstance(n_times, Integral):
        raise TypeError("Number of time steps must be an integer.")
    if n_times <= 0:
        raise ValueError("Number of time steps must be a positive integer.")
    if not isinstance(n_dim, Integral):
        raise TypeError("Number of dimensions must be an integer.")
    if n_dim <= 0:
        raise ValueError("Number of dimensions must be a positive integer.")
    if not isinstance(order, Integral):
        raise TypeError("Order must be an integer.")
    if not isinstance(accuracy, Integral):
        raise TypeError("Accuracy must be an integer.")
    if not isinstance(spacing, Number):
        raise TypeError("Spacing must be a numeric type.")
    if not isinstance(sparse, bool):
        raise TypeError("Sparse flag must be a boolean.")
    
    if order < 0:
        raise ValueError("Order must be a positive integer.")
    if accuracy <= 0:
        raise ValueError("Accuracy must be a positive integer.")
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")
    
    # Backward FD coefficients
    coeffs = compute_backward_finite_difference_coefficients(order, spacing, accuracy)

    # Create the non-zero first columns of the Toeplitz matrix
    col = coeffs

    # Create the non-zero first rows of the Toeplitz matrix
    row = numpy.array([coeffs[0]])

    return _assemble_toeplitz_matrix(row, col, n_times, n_dim, sparse, weights=weights)



def assemble_central_finite_difference_matrix(
    order: Integral,
    n_times: Integral,
    n_dim: Optional[Integral] = 1,
    spacing: Number = 1.0,
    accuracy: Integral = 2,
    sparse: bool = False,
    weights: Optional[numpy.ndarray] = None,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the temporal derivation operator matrix for finite difference approximation with central scheme.

    The operator matrix is used to approximate the temporal derivative of a time-dependent function
    using central finite differences.

    For a time series with :math:`N_t` time steps, the operator matrix :math:`D` for the :math:`n`-th order derivative is given by the toeplitz matrix:

    .. math::

        D = \begin{bmatrix}
        a_{t} & a_{t+1} & 0 & \cdots & 0 & 0 \\
        a_{t-1} & a_{t} & a_{t+1} & \cdots & 0 & 0 \\
        0 & a_{t-1} & a_{t} & \cdots & 0 & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t-1} & a_{t} \\
        \end{bmatrix}

    where :math:`a_k` are the Stencil coefficients from the central finite difference kernel at the given order, spacing and accuracy.

    The dimension :obj:`n_dim` allows to apply the operator to multi-dimensional data at each time step :math:`f = [f_1(t), f_2(t), \ldots, f_{n_{dim}}(t)]^T`.

    Thus the kronecker product with the identity matrix is performed to account for the spatial dimensions.

    .. math::

        D = T \otimes I_{n_{dim}} = \begin{bmatrix}
        a_{t} I_{n_{dim}} & a_{t+1} I_{n_{dim}} & 0 & \cdots & 0 & 0 \\
        a_{t-1} I_{n_{dim}} & a_{t} I_{n_{dim}} & a_{t+1} I_{n_{dim}} & \cdots & 0 & 0 \\
        0 & a_{t-1} I_{n_{dim}} & a_{t} I_{n_{dim}} & \cdots & 0 & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & 0 & \cdots & a_{t-1} I_{n_{dim}} & a_{t} I_{n_{dim}} \\
        \end{bmatrix}

    The operator matrix is then constructed such as :math:`\frac{\partial^n f}{\partial t^n} \approx  D @ F`, where :math:`F` is the flattened time series data :math:`F = [f_1(t_1), f_2(t_1), \ldots, f_{n_{dim}}(t_1), f_1(t_2), f_2(t_2), \ldots, f_{n_{dim}}(t_2), \ldots, f_{n_{dim}}(t_{N_t})]^T`.

    .. seealso::

        - :func:`compute_central_finite_difference_coefficients` to compute the central finite difference coefficients for given order, spacing and accuracy.
        - :func:`assemble_forward_finite_difference_matrix` to assemble the forward finite difference operator matrix.
        - :func:`assemble_backward_finite_difference_matrix` to assemble the backward finite difference operator matrix.

    .. note::

        If the time series has less time steps than the number of stencil points required for the given order and accuracy, the operator matrix will be truncated accordingly.


    Parameters
    ----------
    order : :class:`int`
        The order of the temporal derivative to approximate (e.g., 1 for first derivative, 2 for second derivative).

    n_times : :class:`int`
        The number of time steps in the time series.

    n_dim : :class:`int`, optional
        The number of spatial dimensions of the data at each time step. Default is :obj:`1`.

    spacing : :class:`Number`, optional
        The time step size. Default is :obj:`1.0`.

    accuracy : :class:`int`, optional
        The desired accuracy order of the approximation. Must be an even integer. Default is :obj:`2`.

    sparse : :class:`bool`, optional
        Whether to return a sparse matrix. Default is :obj:`False`.

    weights : :class:`numpy.ndarray`, optional
        An optional array of weights to apply for the identity matrix in the kronecker product. If None, the identity matrix is used. If provided, it must have shape :obj:`(n_dim,)` to weight each spatial dimension accordingly (diagonal coefficients) or :obj:`(n_dim, n_dim)` to provide a full weighting matrix. Default is :obj:`None`.


    Returns
    -------
    :class:`numpy.ndarray` or :class:`scipy.sparse.csr_matrix`
        The temporal derivation operator matrix with shape :obj:`(n_times * n_dim, n_times * n_dim)`.

    
    Raises
    ------
    ValueError
        If the order or accuracy are not positive integers.
        If the accuracy is not an even integer.
        If the spacing is not a strictly positive number.


    Examples
    --------
    
    >>> assemble_central_finite_difference_matrix(2, 5, spacing=1.0, accuracy=2)
    array([[ -2.,  1.,  0.,  0.,  0.],
           [  1., -2.,  1.,  0.,  0.],
           [  0.,  1., -2.,  1.,  0.],
           [  0.,  0.,  1., -2.,  1.],
           [  0.,  0.,  0.,  1., -2.]])

    >>> assemble_central_finite_difference_matrix(1, 4, n_dim=2, spacing=0.1, accuracy=4)
    array([[ -500/2,      0,   400/3,      0, -100/12,       0,       0,       0],
           [      0, -500/2,       0,  400/3,       0, -100/12,       0,       0],
           [  400/3,      0,  -500/2,      0,   400/3,       0, -100/12,       0],
           [      0,  400/3,       0, -500/2,       0,   400/3,       0, -100/12],
           [-100/12,      0,   400/3,      0,  -500/2,       0,   400/3,       0],
           [      0,-100/12,       0,  400/3,       0,  -500/2,       0,   400/3],
           [      0,      0, -100/12,      0,   400/3,       0,  -500/2,       0],
           [      0,      0,       0,-100/12,      0,   400/3,       0,  -500/2]])

    """
    if not isinstance(n_times, Integral):
        raise TypeError("Number of time steps must be an integer.")
    if n_times <= 0:
        raise ValueError("Number of time steps must be a positive integer.")
    if not isinstance(n_dim, Integral):
        raise TypeError("Number of dimensions must be an integer.")
    if n_dim <= 0:
        raise ValueError("Number of dimensions must be a positive integer.")
    if not isinstance(order, Integral):
        raise TypeError("Order must be an integer.")
    if not isinstance(accuracy, Integral):
        raise TypeError("Accuracy must be an integer.")
    if not isinstance(spacing, Number):
        raise TypeError("Spacing must be a numeric type.")
    if not isinstance(sparse, bool):
        raise TypeError("Sparse flag must be a boolean.")
    
    if order < 0:
        raise ValueError("Order must be a positive integer.")
    if accuracy <= 0:
        raise ValueError("Accuracy must be a positive integer.")
    if accuracy % 2 != 0:
        raise ValueError("Accuracy must be an even integer for central finite differences.")
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")
    
    # Central FD coefficients
    coeffs = compute_central_finite_difference_coefficients(order, spacing, accuracy)
    half_size = (len(coeffs) - 1) // 2

    # Create the non-zero first columns of the Toeplitz matrix
    col = coeffs[0:(half_size+1)][::-1]

    # Create the non-zero first rows of the Toeplitz matrix
    row = coeffs[(half_size):]

    return _assemble_toeplitz_matrix(row, col, n_times, n_dim, sparse, weights=weights)



if __name__ == "__main__":
    forward_solution = {
        (1, 1): [-1, 1],
        (1, 2): [-3/2, 2, -1/2],
        (1, 3): [-11/6, 3, -3/2, 1/3],
        (1, 4): [-25/12, 4, -3, 4/3, -1/4],
        (2, 1): [1, -2, 1],
        (2, 2): [2, -5, 4, -1],
        (2, 3): [35/12, -26/3, 19/2, -14/3, 11/12],
        (2, 4): [15/4, -77/6, 107/6, -13, 61/12, -5/6],
        (3, 1): [-1, 3, -3, 1],
        (3, 2): [-5/2, 9, -12, 7, -3/2],
        (3, 3): [-17/4, 71/4, -59/2, 49/2, -41/4, 7/4],
        (3, 4): [-49/8, 29, -461/8, 62, -307/8, 13, -15/8],
    }

    for key, expected in forward_solution.items():
        order, accuracy = key
        computed = compute_forward_finite_difference_coefficients(order, 1.0, accuracy)
        assert numpy.allclose(computed, numpy.array(expected)), f"Forward FD coefficients mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
    print("All forward finite difference coefficient tests passed.")

    backward_solution = {
        (1, 1): [1, -1],
        (1, 2): [3/2, -2, 1/2],
        (1, 3): [11/6, -3, 3/2, -1/3],
        (2, 1): [1, -2, 1],
        (2, 2): [2, -5, 4, -1],
        (3, 1): [1, -3, 3, -1],
        (3, 2): [5/2, -9, 12, -7, 3/2],
        (4, 1): [1, -4, 6, -4, 1],
        (4, 2): [3, -14, 26, -24, 11, -2],
    }

    for key, expected in backward_solution.items():
        order, accuracy = key
        computed = compute_backward_finite_difference_coefficients(order, 1.0, accuracy)
        assert numpy.allclose(computed, numpy.array(expected)), f"Backward FD coefficients mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
    print("All backward finite difference coefficient tests passed.")

    central_solution = {
        (1, 2): [-1/2, 0, 1/2],
        (1, 4): [1/12, -2/3, 0, 2/3, -1/12],
        (1, 6): [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60],
        (2, 2): [1, -2, 1],
        (2, 4): [-1/12, 4/3, -5/2, 4/3, -1/12],
        (2, 6): [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
        (3, 2): [-1/2, 1, 0, -1, 1/2],
        (3, 4): [1/8, -1, 13/8, 0, -13/8, 1, -1/8],
    }

    for key, expected in central_solution.items():
        order, accuracy = key
        computed = compute_central_finite_difference_coefficients(order, 1.0, accuracy)
        assert numpy.allclose(computed, numpy.array(expected)), f"Central FD coefficients mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
    print("All central finite difference coefficient tests passed.")


    data = numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])

    central_solution = {
        (1, 2): numpy.array([0.5, 2., 4., 6., 3.5]),
    }

    for key, expected in central_solution.items():
        order, accuracy = key
        computed = apply_central_finite_difference(data, order=order, spacing=1.0, accuracy=accuracy)
        assert numpy.allclose(computed, expected), f"Central FD application mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
        computed_multi = apply_central_finite_difference(numpy.tile(data, (3, 1)).T, order=order, spacing=1.0, accuracy=accuracy, axis=0)
        expected_multi = numpy.tile(expected, (3, 1)).T
        assert numpy.allclose(computed_multi, expected_multi), f"Central FD application mismatch for multi-dimensional data for order {order} and accuracy {accuracy}: expected {expected_multi}, got {computed_multi}"
    print("All central finite difference application tests passed.")

    forward_solution = {
        (1, 1): numpy.array([1., 3., 5., 7., 0.]),
        (2, 1): numpy.array([2., 2., 2., -7., -7.]),
    }

    for key, expected in forward_solution.items():
        order, accuracy = key
        computed = apply_forward_finite_difference(data, order=order, spacing=1.0, accuracy=accuracy)
        assert numpy.allclose(computed, expected), f"Forward FD application mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
        computed_multi = apply_forward_finite_difference(numpy.tile(data, (2, 1)).T, order=order, spacing=1.0, accuracy=accuracy, axis=0)
        expected_multi = numpy.tile(expected, (2, 1)).T
        assert numpy.allclose(computed_multi, expected_multi), f"Forward FD application mismatch for multi-dimensional data for order {order} and accuracy {accuracy}: expected {expected_multi}, got {computed_multi}"
    print("All forward finite difference application tests passed.")

    backward_solution = {
        (1, 1): numpy.array([0., 1., 3., 5., 7.]),
        (2, 1): numpy.array([1., 1., 2., 2., 2.]),
    }

    for key, expected in backward_solution.items():
        order, accuracy = key
        computed = apply_backward_finite_difference(data, order=order, spacing=1.0, accuracy=accuracy)
        assert numpy.allclose(computed, expected), f"Backward FD application mismatch for order {order} and accuracy {accuracy}: expected {expected}, got {computed}"
        computed_multi = apply_backward_finite_difference(numpy.tile(data, (2, 1)).T, order=order, spacing=1.0, accuracy=accuracy, axis=0)
        expected_multi = numpy.tile(expected, (2, 1)).T
        assert numpy.allclose(computed_multi, expected_multi), f"Backward FD application mismatch for multi-dimensional data for order {order} and accuracy {accuracy}: expected {expected_multi}, got {computed_multi}"
    print("All backward finite difference application tests passed.")