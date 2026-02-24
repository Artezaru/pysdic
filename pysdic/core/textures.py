# Copyright 2026 Artezaru
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

from importlib import resources
import cv2
import numpy

def get_lena_texture() -> numpy.ndarray:
    r"""
    Get the Lena texture image as a numpy array. This function loads the Lena image from the package resources and returns it as a numpy array that can be used as a texture in visualizations.
    
    .. figure:: /_static/textures/lena_texture.png
        :alt: Lena texture image
        :align: center
        :width: 200px
    
    Returns
    -------
    :class:`numpy.ndarray`
        The Lena texture image as a numpy array with shape (474, 474) and dtype uint8.   
    
    """
    path = resources.files("pysdic.resources") / "lena_texture.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = numpy.asarray(image, dtype=numpy.uint8)
    return image

