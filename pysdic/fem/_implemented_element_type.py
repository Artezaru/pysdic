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

from types import MappingProxyType

# Read-only mapping (the nested mappings are read-only too): the supported element
# types and their properties cannot be modified from outside the module.
_MAPPING_ELEMENT_TYPE_TO_PROPERTIES = {
    "segment_2": {
        "expected_N_vpe": 2,
        "expected_K": 1,
        "meshio_cell_type": "line",
        "vtk_cell_type": 3,
    },
    "segment_3": {
        "expected_N_vpe": 3,
        "expected_K": 1,
        "meshio_cell_type": "line3",
        "vtk_cell_type": 21,
    },
    "triangle_3": {
        "expected_N_vpe": 3,
        "expected_K": 2,
        "meshio_cell_type": "triangle",
        "vtk_cell_type": 5,
    },
    "triangle_6": {
        "expected_N_vpe": 6,
        "expected_K": 2,
        "meshio_cell_type": "triangle6",
        "vtk_cell_type": 22,
    },
    "quadrangle_4": {
        "expected_N_vpe": 4,
        "expected_K": 2,
        "meshio_cell_type": "quad",
        "vtk_cell_type": 9,
    },
    "quadrangle_8": {
        "expected_N_vpe": 8,
        "expected_K": 2,
        "meshio_cell_type": "quad8",
        "vtk_cell_type": 23,
    },
}
_MAPPING_ELEMENT_TYPE_TO_PROPERTIES = MappingProxyType(
    {
        element_type: MappingProxyType(properties)
        for element_type, properties in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES.items()
    }
)