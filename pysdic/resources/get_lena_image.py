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

from importlib import resources
import numpy
import cv2

from .._optionals import import_optional

def get_lena_image() -> numpy.ndarray:
    r"""
    Get the Lena image as a numpy array.
    This function loads the Lena image from the package resources and returns it as a
    numpy array that can be used as a texture in visualizations.

    .. note::

        This function requires OpenCV (``cv2``) to read the image.

    .. figure:: /_static/textures/lena_texture.png
        :alt: Lena image
        :align: center
        :width: 200px

    Returns
    -------
    numpy.ndarray
        The Lena image as a numpy array with shape (474, 474) and dtype uint8.

    Raises
    ------
    ImportError
        If OpenCV (``cv2``) is not installed.

    FileNotFoundError
        If the image cannot be read from the package resources.

    """
    path = resources.files("pysdic.resources") / "lena_texture.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:  # cv2.imread returns None instead of raising an error
        raise FileNotFoundError(f"The Lena image cannot be read from the package resources ('{path}').")
    image = numpy.asarray(image, dtype=numpy.uint8)
    return image