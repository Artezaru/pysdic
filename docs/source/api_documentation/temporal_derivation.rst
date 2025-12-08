.. currentmodule:: pysdic

Build derivation operators to regularize temporal derivatives
=====================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

   
Description
-------------------------------------------

Context
~~~~~~~~~~~~~

Lets consider a time-dependent function :math:`f(t)` sampled at :math:`N_t` discrete time steps with a constant time interval :math:`\Delta t`.
We want to regularize the temporal derivative of :math:`f(t)` as :

.. math::
   
   \frac{d^n f}{d t^n} \approx \hat{g}

The derivation operator is built to approximate the temporal derivative of :math:`f(t)` using finite difference schemes such as central, forward, or backward finite differences.

.. math::
   
   \frac{d^n f}{d t^n} \approx D^n f

where :math:`D^n` is the temporal derivation operator matrix constructed using finite difference kernels (Stencil coefficients) for the :math:`n^{th}` order derivative.

Finite Difference Stencil coefficients :math:`c_j`
-------------------------------------------------------------------

The package provides functions to construct finite difference stencil coefficients for central, forward, and backward finite difference schemes.

.. autosummary::
   :toctree: ../generated/

    compute_central_finite_difference_coefficients
    compute_forward_finite_difference_coefficients
    compute_backward_finite_difference_coefficients

The finite difference stencil coefficients are used to approximate the temporal derivative of a function at discrete time steps.
To apply the finite difference scheme to a time-dependent function, we can convolve the function with the stencil coefficients.

.. autosummary::
   :toctree: ../generated/

    apply_central_finite_difference
    apply_forward_finite_difference
    apply_backward_finite_difference
    
    

Temporal Derivation Matrix :math:`D^n`
-------------------------------------------

The package provides a function to assemble the temporal derivation operator matrix :math:`D^n` using finite difference kernels.

.. autosummary::
   :toctree: ../generated/

    assemble_central_finite_difference_matrix
    assemble_forward_finite_difference_matrix
    assemble_backward_finite_difference_matrix


Cost function development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In pratice, we want to find the function :math:`dU(t)` that minimizes the Stereo Digital Image Correlation (SDIC) cost function with an additional regularization term on the temporal derivative of :math:`U(t)`.

Without regularization, the SDIC cost function can be written as :

.. math::
   
   \Phi_{\text{SDIC}}(dU) = \|J dU - r\|^2

where :math:`J` is the SDIC Jacobian matrix, :math:`dU` is the increment on displacement field, and :math:`r` is the residual vector.

The least squares formulation of the SDIC problem can be solved as :

.. math::
   
   J^T J dU = J^T r

When adding a regularization term on the temporal derivative of :math:`dU(t)`, the complete cost function becomes :

.. math::
   
   \Phi(dU) = \Phi_{\text{SDIC}}(dU) + \lambda \| D^n U - \hat{g} \|^2

where :math:`\lambda` is a regularization parameter, and :math:`D^n` is the temporal derivation operator matrix for the :math:`n^{th}` order derivative.

The regularization term can be development such as : 

.. math::

   \min \| D^n U - \hat{g} \|^2 \leftrightarrow (D^n)^T D^n U = (D^n)^T \hat{g} \leftrightarrow (D^n)^T D^n dU = (D^n)^T (\hat{g} - D^n U_{\text{init}})
