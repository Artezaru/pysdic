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

from typing import Optional

import numpy

from ..core.objects.mesh import Mesh
from ..core.objects.point_cloud import PointCloud

class BlenderMesh(Mesh):
    r"""
    A Blender-specific mesh class extending the core Mesh.

    Specific mesh with 'triangle_3' element type and 3D coordinates for Blender.

    This class can include Blender-specific methods and attributes for handling meshes within Blender.

    .. note::

        This class requires Blender's Python environment with the `bpy` module available.

    .. seealso::

        - :class:`pysdic.Mesh` for the base mesh class.

    Parameters
    ----------
    vertices : :class:`PointCloud`
        The vertices of the mesh as a :class:`PointCloud` instance with shape (:math:`N_v`, :math:`E`), where :math:`N_v` is the number of vertices and :math:`E` is the embedding dimension.

    connectivity : :class:`numpy.ndarray`
        The connectivity of the mesh as a numpy ndarray with shape (:math:`N_e`, :math:`N_{vpe}`), where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

    vertices_properties : Optional[:class:`dict`], optional
        A dictionary to store properties of the vertices, each property should be a numpy ndarray of shape (:math:`N_v`, :math`A`) where :math:`N_v` is the number of vertices and :math:`A` is the number of attributes for that property, by default None.

    elements_properties : Optional[:class:`dict`], optional
        A dictionary to store properties of the elements, each property should be a numpy ndarray of shape (:math:`N_e`, :math`B`) where :math:`N_e` is the number of elements and :math:`B` is the number of attributes for that property, by default None.
    
    internal_bypass : :class:`bool`, optional
        If :obj:`True`, internal checks are bypassed for better performance, by default :obj:`False`.
    """

    def __init__(self, vertices: PointCloud, connectivity: numpy.ndarray, vertices_properties: Optional[dict] = None, elements_properties: Optional[dict] = None, elements_type : Optional[str] = 'triangle_3', internal_bypass: bool = False) -> None:
        super().__init__(vertices=vertices, connectivity=connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, elements_type=elements_type, internal_bypass=internal_bypass)
        if self.elements_type is None or self.elements_type != 'triangle_3':
            raise ValueError("BlenderMesh only supports 'triangle_3' element type.")
        
    
    @classmethod
    def from_mesh(cls, mesh: Mesh) -> BlenderMesh:
        r"""
        Create a BlenderMesh instance from a generic Mesh instance.

        This method converts a generic Mesh into a BlenderMesh, ensuring compatibility with Blender's requirements.

        .. code-block:: python

            blender_mesh = BlenderMesh.from_mesh(generic_mesh)

        Parameters
        ----------
        mesh : :class:`Mesh`
            The generic Mesh instance to convert.

        Returns
        -------
        :class:`BlenderMesh`
            A new BlenderMesh instance created from the provided Mesh.
        """
        if not isinstance(mesh, Mesh):
            raise TypeError("Input must be an instance of Mesh.")
        if mesh.elements_type != 'triangle_3':
            raise ValueError("Input Mesh must have 'triangle_3' element type to convert to BlenderMesh.")
        return cls(vertices=mesh.vertices, connectivity=mesh.connectivity, vertices_properties=mesh.vertices_properties, elements_properties=mesh.elements_properties, elements_type=mesh.elements_type, internal_bypass=mesh.internal_bypass)
    

    def to_mesh(self) -> Mesh:
        r"""
        Convert the BlenderMesh instance to a generic Mesh instance.

        This method creates a generic Mesh from the BlenderMesh, removing any Blender-specific attributes.

        .. code-block:: python

            generic_mesh = blender_mesh.to_mesh()

        Returns
        -------
        :class:`Mesh`
            A new Mesh instance created from the BlenderMesh.
        """
        return Mesh(vertices=self.vertices, connectivity=self.connectivity, vertices_properties=self.vertices_properties, elements_properties=self.elements_properties, elements_type=self.elements_type, internal_bypass=self.internal_bypass)