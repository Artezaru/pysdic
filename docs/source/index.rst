Welcome to pysdic's documentation!
=========================================================================================

.. image:: /_static/_icons/pysdic_icon.png
   :alt: pysdic icon
   :width: 120px
   :align: center

Description of the package
--------------------------

Python Stereo-Digital Image Correlation (SDIC) including meshes manipulation, 
camera models and calibration, and build-in transformations for 3D DIC.

.. note::

   The package is designed to work with double-precision floating-point numbers to ensure numerical stability in all calculations.
   Therefore, all float arrays are automatically converted to ``numpy.float64`` for computation and all integer arrays are converted to ``numpy.int64`` for computation.
   This means that when you pass arrays to the functions in the package, they will be converted to these data types if they are not already in that format.


Contents
--------

.. grid:: 2

    .. grid-item-card:: 
      :img-top: /_static/_icons/download.png
      :text-align: center

      Installation
      ^^^

      This section describes how to install the package into a Python environment. 
      It includes instructions for installing the package using pip, as well as any necessary dependencies.

      +++

      .. button-ref:: installation
         :expand:
         :color: secondary
         :click-parent:

         To the installation guide

    .. grid-item-card::
      :img-top: /_static/_icons/api.png
      :text-align: center

      API Reference
      ^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in ``pysdic``. The reference describes how the
      methods work and which parameters can be used. It assumes that you have an
      understanding of the key concepts.

      +++ 

      .. button-ref:: api
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

    .. grid-item-card::
      :img-top: /_static/_icons/examples.png
      :text-align: center

      Examples Gallery
      ^^^

      This section contains a collection of examples demonstrating how to use the package for various applications.
      Each example includes a description of the problem being solved, the code used to solve it, and the resulting output.

      +++

      .. button-ref:: usage
         :expand:
         :color: secondary
         :click-parent:

         To the examples gallery

    .. grid-item-card::
      :img-top: /_static/_icons/pdf.png
      :text-align: center

      PDF Documentation
      ^^^

      The beamer presentation of the package gives an overview of a SDIC analysis with ``pysdic``:
      the objects to create, the unknowns, the correlation and regularization terms and the optimization.

      +++

      .. button-link:: https://github.com/Artezaru/pysdic/blob/master/beamer/presentation.pdf
         :expand:
         :color: secondary
         :click-parent:

         To the PDF presentation

.. toctree::
   :caption: Contents:
   :hidden:

   installation
   api
   usage


Author
------

The package ``pysdic`` was created by the following authors:

- Artezaru <artezaru.github@proton.me>

You can access the package and the documentation with the following URL:

- **Git Plateform**: https://github.com/Artezaru/pysdic.git
- **Online Documentation**: https://Artezaru.github.io/pysdic


License
-------

.. code-block:: text 

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