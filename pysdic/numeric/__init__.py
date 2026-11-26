"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

r"""
Numerical tools (finite differences, polynomial fitting), not exported in the ``pysdic``
namespace: use ``from pysdic.numeric import ...``.
"""

from .derivation import (
    compute_finite_difference_coefficients,
    compute_forward_finite_difference_coefficients,
    compute_backward_finite_difference_coefficients,
    compute_central_finite_difference_coefficients,
    assemble_finite_difference_matrix,
    assemble_forward_finite_difference_matrix,
    assemble_backward_finite_difference_matrix,
    assemble_central_finite_difference_matrix,
    apply_forward_finite_difference,
    apply_backward_finite_difference,
    apply_central_finite_difference,
)
from .polynomial_fitting import fit_polynomial, evaluate_polynomial, polynomial_predict

__all__ = [
    "compute_finite_difference_coefficients",
    "compute_forward_finite_difference_coefficients",
    "compute_backward_finite_difference_coefficients",
    "compute_central_finite_difference_coefficients",
    "assemble_finite_difference_matrix",
    "assemble_forward_finite_difference_matrix",
    "assemble_backward_finite_difference_matrix",
    "assemble_central_finite_difference_matrix",
    "apply_forward_finite_difference",
    "apply_backward_finite_difference",
    "apply_central_finite_difference",
    "fit_polynomial",
    "evaluate_polynomial",
    "polynomial_predict",
]
