Numerical Tools (``pysdic.numeric``)
====================================

.. currentmodule:: pysdic.numeric

.. contents:: Table of Contents
   :local:
   :depth: 1

The submodule ``pysdic.numeric`` provides low-level numerical tools on 1D signals sampled at given abscissas (e.g. the
times of the frames): finite differences and polynomial fitting. They are not exported in the ``pysdic`` namespace:

.. code-block:: python

   from pysdic.numeric import assemble_finite_difference_matrix, fit_polynomial

They are used, for example, by :class:`pysdic.SDICTemporalRegularizationTerm` (finite difference matrix on the actual
times of the frames).


Finite differences
------------------

The finite difference coefficients :math:`c_k` approximate the derivative of order :math:`n` of a function at a point
:math:`t` from its values at the points :math:`t_k` of a stencil:

.. math::

   f^{(n)}(t) \approx \sum_k c_k \, f(t_k)

The coefficients are computed for any (non-uniform) stencil with the algorithm of Fornberg, and the stencils can be
centered, forward or backward.

Coefficients
^^^^^^^^^^^^

- :doc:`compute_finite_difference_coefficients <numeric/compute_finite_difference_coefficients>`: Coefficients of a derivative of any order on any stencil (Fornberg).
- :doc:`compute_forward_finite_difference_coefficients <numeric/compute_forward_finite_difference_coefficients>`: Coefficients on a forward stencil.
- :doc:`compute_backward_finite_difference_coefficients <numeric/compute_backward_finite_difference_coefficients>`: Coefficients on a backward stencil.
- :doc:`compute_central_finite_difference_coefficients <numeric/compute_central_finite_difference_coefficients>`: Coefficients on a centered stencil.

Matrices
^^^^^^^^

The matrix :math:`D` of a finite difference over a whole signal, :math:`f^{(n)} \approx D f` (one row per evaluated
point), for example to build a linear regularization term.

- :doc:`assemble_finite_difference_matrix <numeric/assemble_finite_difference_matrix>`: Matrix of a derivative on non-uniform abscissas, with centered stencils inside and one-sided stencils at the boundaries.
- :doc:`assemble_forward_finite_difference_matrix <numeric/assemble_forward_finite_difference_matrix>`: Matrix with forward stencils.
- :doc:`assemble_backward_finite_difference_matrix <numeric/assemble_backward_finite_difference_matrix>`: Matrix with backward stencils.
- :doc:`assemble_central_finite_difference_matrix <numeric/assemble_central_finite_difference_matrix>`: Matrix with centered stencils.

Application to a signal
^^^^^^^^^^^^^^^^^^^^^^^

- :doc:`apply_forward_finite_difference <numeric/apply_forward_finite_difference>`: Derivative of a signal with forward stencils.
- :doc:`apply_backward_finite_difference <numeric/apply_backward_finite_difference>`: Derivative of a signal with backward stencils.
- :doc:`apply_central_finite_difference <numeric/apply_central_finite_difference>`: Derivative of a signal with centered stencils.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./numeric/compute_finite_difference_coefficients.rst
   ./numeric/compute_forward_finite_difference_coefficients.rst
   ./numeric/compute_backward_finite_difference_coefficients.rst
   ./numeric/compute_central_finite_difference_coefficients.rst
   ./numeric/assemble_finite_difference_matrix.rst
   ./numeric/assemble_forward_finite_difference_matrix.rst
   ./numeric/assemble_backward_finite_difference_matrix.rst
   ./numeric/assemble_central_finite_difference_matrix.rst
   ./numeric/apply_forward_finite_difference.rst
   ./numeric/apply_backward_finite_difference.rst
   ./numeric/apply_central_finite_difference.rst


Polynomial fitting
------------------

Fit a polynomial to sampled values in the least squares sense and evaluate it, e.g. to extrapolate a displacement
from the previous frames.

- :doc:`fit_polynomial <numeric/fit_polynomial>`: Coefficients of the least squares polynomial of a given degree through sampled values.
- :doc:`evaluate_polynomial <numeric/evaluate_polynomial>`: Values of a polynomial at given abscissas.
- :doc:`polynomial_predict <numeric/polynomial_predict>`: Fit a polynomial and evaluate it at new abscissas in one call.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./numeric/fit_polynomial.rst
   ./numeric/evaluate_polynomial.rst
   ./numeric/polynomial_predict.rst