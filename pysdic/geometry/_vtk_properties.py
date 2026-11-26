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
Conversion of the properties of pysdic objects (:class:`pysdic.PointCloud`,
:class:`pysdic.Mesh`, ...) to and from VTK data arrays (``vtkPointData``,
``vtkCellData``).

.. important::

    Internal use only. vtk is an optional dependency: the callers must call
    ``import_optional("vtk", ...)`` before using these functions.

"""

from typing import Any, Dict

import numpy


def _check_vtk_property_shapes(data: Dict[str, numpy.ndarray]) -> None:
    r"""
    Check that the properties can be stored in VTK arrays (at most 2 dimensions).

    .. important::

        Internal use only (shared by the pysdic objects exported to VTK).

    """
    for key, values in data.items():
        if values.ndim > 2:
            raise ValueError(
                f"Property '{key}' has shape {values.shape}: VTK arrays have at most 2 "
                f"dimensions (N, A). Consider using 'save_properties=False'."
            )


def _dict_to_vtk_attributes(data: Dict[str, numpy.ndarray], attributes: Any) -> None:
    r"""
    Add the arrays of ``data`` to VTK attributes (``vtkPointData`` or ``vtkCellData``),
    named after their keys (deep copies). Boolean arrays are stored as ``uint8``.

    .. important::

        Internal use only (shared by the pysdic objects exported to VTK).
        vtk must be available (call ``import_optional("vtk", ...)`` before).

    """
    from vtkmodules.util.numpy_support import numpy_to_vtk

    for key, values in data.items():
        if values.dtype == numpy.bool_:
            values = values.astype(numpy.uint8)
        array = numpy_to_vtk(numpy.ascontiguousarray(values), deep=True)
        array.SetName(key)
        attributes.AddArray(array)


def _vtk_attributes_to_dict(attributes: Any, label: str) -> Dict[str, numpy.ndarray]:
    r"""
    Read the arrays of VTK attributes (``vtkPointData`` or ``vtkCellData``) as numpy
    arrays (views on the VTK memory: copy them before keeping them).

    Non-numeric arrays (e.g. ``vtkStringArray``) raise a ``TypeError`` and unnamed
    arrays a ``ValueError``.

    .. important::

        Internal use only (shared by the pysdic objects exported to VTK).
        vtk must be available (call ``import_optional("vtk", ...)`` before).

    """
    from vtkmodules.vtkCommonCore import vtkDataArray
    from vtkmodules.util.numpy_support import vtk_to_numpy

    data = {}
    for index in range(attributes.GetNumberOfArrays()):
        array = attributes.GetAbstractArray(index)
        name = array.GetName()
        if not isinstance(array, vtkDataArray):
            raise TypeError(
                f"The {label} array '{name}' is not numeric (type {type(array)}). "
                f"Use 'load_properties=False' to ignore the {label}."
            )
        if not name:
            raise ValueError(f"The {label} array at index {index} has no name.")
        data[name] = vtk_to_numpy(array)
    return data