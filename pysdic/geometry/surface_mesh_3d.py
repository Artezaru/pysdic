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
from abc import ABC

from typing import Optional
from numbers import Number

import numpy
import pyvista
import matplotlib.pyplot as plt

from .mesh_3d import Mesh3D
from .point_cloud_3d import PointCloud3D
from .integration_points import IntegrationPoints

class SurfaceMesh3D(Mesh3D, ABC):
    r"""
    Subclass of :class:`pysdic.geometry.Mesh3D` representing a 3D mesh composed of surface elements.

    This is an abstract base class for surface 3D meshes.

    The vertices are represented as a PointCloud3D instance with shape (N, 3),
    The connectivity is represented as a numpy ndarray with shape (M, K),
    where N is the number of vertices (``n_vertices``), M is the number of elements (``n_elements``), and K is
    the number of vertices per element (``n_vertices_per_element``). 

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The number of natural coordinates :math:`(\xi, \eta)` with a topological dimension of :math:`d=2`.

    The subclasses must implement the following attributes and methods:

    - (class property) ``_n_vertices_per_element``: int, the number of vertices K per element.
    - (class property) ``_meshio_cell_type``: str, the cell type used by meshio for this type of element.
    - (class property) ``_vtk_cell_type``: int, the cell type used by VTK for this type of element.
    - (class property) ``_is_visualizable``: bool, indicates if the mesh type can be visualized using PyVista.
    - (method) ``shape_functions``: Callable[[numpy.ndarray], [numpy.ndarray, numpy.ndarray]], a method to compute the shape functions at given natural coordinates (and optional Jacobians).

    Parameters
    ----------
    vertices : PointCloud3D
        The vertices of the mesh as a PointCloud3D instance with shape (N, 3).

    connectivity : numpy.ndarray
        The connectivity of the mesh as a numpy ndarray with shape (M, K),
        where M is the number of elements and K is the number of vertices per element.

    vertices_properties : Optional[dict], optional
        A dictionary to store properties of the vertices, each property should be a numpy ndarray of shape (N, A) where N is the number of vertices and A is the number of attributes for that property, by default None.

    elements_properties : Optional[dict], optional
        A dictionary to store properties of the elements, each property should be a numpy ndarray of shape (M, B) where M is the number of elements and B is the number of attributes for that property, by default None.

    internal_bypass : bool, optional
        If True, internal checks are skipped for better performance, by default False.

    """
    __slots__ = ["_vertices_predefined_metadata", "_elements_predefined_metadata"]

    _n_vertices_per_element: int = None
    _n_dimensions: int = 2
    _meshio_cell_type: str = None
    _vtk_cell_type: int = None
    _is_visualizable: bool = None

    def __init__(self, vertices: PointCloud3D, connectivity: numpy.ndarray, *, vertices_properties: dict = None, elements_properties: dict = None, internal_bypass: bool = False):
        # Define expected properties informations
        if not hasattr(self, "_vertices_predefined_metadata"):
            self._vertices_predefined_metadata = {}
        if not hasattr(self, "_elements_predefined_metadata"):
            self._elements_predefined_metadata = {}

        self._elements_predefined_metadata.update({
            "uvmap": {"dim": 2 * self.n_vertices_per_element, "check_method": self._internal_check_uvmap},
        })

        super().__init__(vertices, connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, internal_bypass=internal_bypass)


    # ======================
    # Internal Checks Methods
    # ======================        
    def _internal_check_uvmap(self, uvmap: numpy.ndarray) -> None:
        r"""
        Internal method to check the validity of the uvmap property.

        Parameters
        ----------
        uvmap : numpy.ndarray
            The uvmap property to check, should be of shape (M, 2 * K) where M is the number of elements and K is the number of vertices per element.

        Raises
        ------
        ValueError
            If any uv coordinate is not in the range [0, 1].
        """
        if numpy.any(uvmap < 0) or numpy.any(uvmap > 1):
            raise ValueError("All UV coordinates must be in the range [0, 1].")


    # =======================
    # New Properties
    # =======================
    @property
    def elements_uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the UV mapping of each triangular element.

        The UV mapping is stored as a numpy ndarray of shape (M, 2 * K), where M is the number of elements and K is the number of vertices per element.

        The 6 values correspond to the UV coordinates of the 3 vertices of the triangle: (u1, v1, u2, v2, u3, v3, ..., uK, vK).

        .. note::

            The UV coordinates are stored as an element property of the mesh under the key "uvmap".

        Parameters
        ----------
        value : numpy.ndarray, optional
            A numpy ndarray of shape (M, 2 * K) to set as the UV mapping.

        Returns
        -------
        numpy.ndarray
            An array of shape (M, 2 * K) where M is the number of elements. Or None if not set.
        """
        return self.get_elements_property("uvmap")

    @elements_uvmap.setter
    def elements_uvmap(self, value: Optional[numpy.ndarray]) -> None:
        self.set_elements_property("uvmap", value)


    @property
    def vtk_cell_type(self) -> int:
        r"""
        Get the cell type integer used by vtk for this type of element to visualize the mesh.

        Returns
        -------
        int
            The cell type used by vtk.
        """
        if self._vtk_cell_type is None:
            raise NotImplementedError("Subclasses must implement vtk_cell_type property.")
        return self._vtk_cell_type

    @property
    def is_visualizable(self) -> bool:
        r"""
        Indicates if the mesh type can be visualized using PyVista.

        Returns
        -------
        bool
            True if the mesh type can be visualized, False otherwise.
        """
        if self._is_visualizable is None:
            raise NotImplementedError("Subclasses must implement is_visualizable property.")
        return self._is_visualizable


    # =======================
    # Visualization Methods
    # =======================
    def visualize(
            self,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_color: str = "gray",   
            faces_opacity: float = 0.5,
            show_vertices: bool = True,
            show_edges: bool = True,
            show_faces: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the 3D surface mesh using PyVista.

        This method creates a 3D plot of the mesh, displaying its vertices, edges, and faces.
        The appearance of the vertices, edges, and faces can be customized using various parameters.

        .. seealso::

            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.

        Parameters
        ----------
        vertices_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        vertices_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        vertices_opacity : float, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default 1.0.

        edges_color : str, optional
            Color of the edges in the mesh, by default "black".

        edges_width : int, optional
            Width of the edges in the mesh, by default 1.

        edges_opacity : float, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default 1.0.

        faces_color : str, optional
            Color of the faces in the mesh, by default "gray".

        faces_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 0.5.

        show_points : bool, optional
            Whether to display the vertices (points) of the mesh, by default True.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        show_faces : bool, optional
            Whether to display the faces of the mesh, by default True.

        title : Optional[str], optional
            Title of the plot, by default None.

        show_axes : bool, optional
            Whether to display the axes in the plot, by default True.

        show_grid : bool, optional
            Whether to display the grid in the plot, by default True.


        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D:

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            surface_mesh.visualize(faces_color='green', faces_opacity=0.7, edges_color='black')

        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_example.png
           :width: 600
           :align: center
            
           Example of a 3D triangular mesh visualization using the `visualize` method.
            
        """
        # Check if visualizable
        if not self.is_visualizable:
            raise NotImplementedError("This mesh type cannot be visualized using PyVista.")
        
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")

        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not isinstance(faces_color, str):
            raise ValueError("Faces color must be a string.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise ValueError("show_faces must be a boolean.")
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
        
        # Create a PyVista mesh
        n_cells = self.n_elements
        cells = numpy.hstack([numpy.full((n_cells, 1), self.n_vertices_per_element), self.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, self.vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, self.vertices.points)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add faces if required
        if show_faces:
            plotter.add_mesh(
                pv_mesh, 
                color=faces_color, 
                opacity=faces_opacity
            )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes() 
        if show_grid:
            plotter.show_grid()
        plotter.show()



    
    def visualize_vertices_property(
            self,
            property_key: Optional[str] = None,
            property_array: Optional[numpy.ndarray] = None,
            property_axis: Optional[int] = None,
            property_label: Optional[str] = None,
            cmap: str = "magma",
            vmin : Optional[float] = None,
            vmax : Optional[float] = None,
            use_log_scale: bool = False,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_opacity: float = 1.0,
            show_vertices: bool = True,
            show_edges: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
            ) -> None:
        r"""
        Visualize a vertex property on the 3D surface mesh using PyVista.

        This method creates a 3D plot of the mesh, displaying its vertices colored according to the specified property.
        The appearance of the vertices can be customized using various parameters.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without coloring by a property.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.

        Parameters
        ----------
        property_key : str, optional
            The name of the vertex property to visualize. If None, `property_array` must be provided, by default None.

        property_array : numpy.ndarray, optional
            A numpy ndarray of shape (N, A) where N is the number of vertices and A is the number of attributes for that property.
            If None, `property_key` must be provided, by default None.

        property_axis : int, optional
            The axis of the property to visualize (0 for x, 1 for y, 2 for z). If None, the magnitude of the property will be visualized, by default None.

        property_label : str, optional
            The label to use for the property in the visualization legend. If None, `property_key` will be used, by default None.

        cmap : str, optional
            The colormap to use for coloring the vertices based on the property values, by default "magma".

        vmin : float, optional
            The minimum value for the colormap, by default None.

        vmax : float, optional
            The maximum value for the colormap, by default None.

        use_log_scale : bool, optional
            Whether to use a logarithmic scale for the colormap, by default False.

        vertices_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        vertices_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        vertices_opacity : float, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default 1.0.

        edges_color : str, optional
            Color of the edges in the mesh, by default "black".

        edges_width : int, optional
            Width of the edges in the mesh, by default 1.

        edges_opacity : float, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default 1.0.

        faces_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 1.0.

        show_points : bool, optional
            Whether to display the vertices (points) of the mesh, by default True.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        title : Optional[str], optional
            Title of the plot, by default None.

        show_axes : bool, optional
            Whether to display the axes in the plot, by default True.

        show_grid : bool, optional
            Whether to display the grid in the plot, by default True.

            
        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize the height (z-coordinate) of each vertex:

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            
            height = surface_mesh.vertices.points[:, 2].reshape(-1, 1)  # Use the z-coordinate as a property

            surface_mesh.visualize_vertices_property(
                property_array=height, 
                property_label='Height [m]',
                property_axis=0,
                cmap='terrain'
                )

        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_vertices_property_example.png
           :width: 600
           :align: center

           Example of a 3D triangular mesh visualization using the `visualize_vertices_property` method.

        """
        # Check if visualizable
        if not self.is_visualizable:
            raise NotImplementedError("This mesh type cannot be visualized using PyVista.")
        
        # Case of an empty mesh
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        
        # Extract the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_vertices_property(property_key, property_array, raise_error=True)

        property_array = numpy.asarray(property_array)
        if property_array.ndim == 1:
            property_array = property_array.reshape(-1, 1)
        if property_array.shape[0] != self.n_vertices:
            raise ValueError(f"property_array must have shape ({self.n_vertices}, A) where A is the number of attributes.")
        if property_array.shape[1] == 0:
            raise ValueError("property_array must have at least one attribute (shape (N, A) with A >= 1).")   
        
        # Default parameters
        if property_label is None and property_key is not None:
            property_label = property_key
        elif property_label is None:
            property_label = "property"

        # Extract the desired axis
        if property_axis is not None:
            if not isinstance(property_axis, int):
                raise ValueError("property_axis must be an integer.")
            if property_axis < 0 or property_axis >= property_array.shape[1]:
                raise ValueError(f"property_axis must be between 0 and {property_array.shape[1]-1}.")
            property_array = property_array[:, property_axis]
            property_label = f"{property_label} (Axis {property_axis})"
        else:
            # Use the magnitude of the property
            property_array = numpy.linalg.norm(property_array, axis=1)
            property_label = f"{property_label} (Magnitude)"
        # Now property_array is of shape (N,)    

        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = numpy.min(property_array)
        if vmax is None:
            vmax = numpy.max(property_array)

        # Input checks
        if not isinstance(cmap, str):
            raise ValueError("cmap must be a string.")
        if not isinstance(property_label, str):
            raise ValueError("property_label must be a string.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not (isinstance(vmin, Number) and isinstance(vmax, Number)):
            raise ValueError("vmin and vmax must be numbers.")
        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax.")
        
        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(use_log_scale, bool):
            raise ValueError("use_log_scale must be a boolean.")
        
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
    
        
        # Extract the cmap
        colormaps = plt.colormaps()
        if not cmap in colormaps:
            raise ValueError(f"cmap '{cmap}' is not a valid colormap. Available colormaps are: {colormaps}")

        # Create a PyVista mesh
        n_cells = self.n_elements
        cells = numpy.hstack([numpy.full((n_cells, 1), self.n_vertices_per_element), self.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, self.vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, self.vertices.points)

        # Add the property as point data
        pv_mesh.point_data[property_label] = property_array

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the property colormap
        plotter.add_mesh(
            pv_mesh, 
            scalars=property_label, 
            cmap=cmap,
            clim=(vmin, vmax),
            log_scale=use_log_scale,
            opacity=faces_opacity,
        )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()


    def visualize_texture(
            self,
            texture: numpy.ndarray,
            use_rgb: bool = True,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_opacity: float = 1.0,
            show_vertices: bool = True,
            show_edges: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the texture of the mesh using a texture image.

        .. warning::

            The mesh must have the `uvmap` property set for this method to work.

        .. seealso::

            - :meth:`elements_uvmap` to set or get the UV mapping of the elements.

        This method creates a 3D plot of the mesh, displaying its faces textured with the provided image.
        The texture image should be a 2D (grayscale) or 3D (RGB) numpy array.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without texture.
            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.

        Parameters
        ----------
        texture : numpy.ndarray
            The texture image to apply to the mesh. Integer arrays with values in [0, 255] with dtype ``numpy.uint8``.
            Array must have shape (height, width, 3) for RGB textures or (height, width) for grayscale textures.

        use_rgb : bool, optional
            Whether to interpret the texture as RGB (True). If False, any RGB texture will be converted to grayscale, by default True.

        vertices_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        vertices_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        vertices_opacity : float, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default 1.0.

        edges_color : str, optional
            Color of the edges in the mesh, by default "black".

        edges_width : int, optional
            Width of the edges in the mesh, by default 1.

        edges_opacity : float, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default 1.0.

        faces_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 1.0.

        show_points : bool, optional
            Whether to display the vertices (points) of the mesh, by default True.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        title : Optional[str], optional
            Title of the plot, by default None.

        show_axes : bool, optional
            Whether to display the axes in the plot, by default True.

        show_grid : bool, optional
            Whether to display the grid in the plot, by default True.

        More Information
        -------------------------
        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize a checkerboard texture on it:

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            ) # UVMAP already set in the function

            # Create a texture image
            u = numpy.linspace(0, 1, 50)
            v = numpy.linspace(0, 1, 50)
            U, V = numpy.meshgrid(u, v)

            texture_image = numpy.round(255/2 + 255/2 * numpy.sin(U * 4 * numpy.pi)).astype(numpy.uint8)  # Example texture image with shape (50, 50)

            surface_mesh.visualize_texture(texture_image, show_edges=False, show_vertices=False)
            
        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_texture_example.png
            :width: 600
            :align: center

            Example of a 3D triangular mesh visualization using the `visualize_texture` method.

        """
        # Check if visualizable
        if not self.is_visualizable:
            raise NotImplementedError("This mesh type cannot be visualized using PyVista.")
        
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        if self.elements_uvmap is None:
            raise ValueError("The mesh must have the 'uvmap' property set to visualize texture.")
        
        if not isinstance(texture, numpy.ndarray):
            raise ValueError("texture must be a numpy ndarray.")
        
        if texture.ndim < 2 or texture.ndim > 3:
            raise ValueError("texture must be a 2D (grayscale) or 3D (RGB) array.")
        if texture.ndim == 3 and texture.shape[2] not in [1, 3]:
            raise ValueError("If texture is 3D, its third dimension must be 1 (grayscale) or 3 (RGB).")
        if texture.dtype != numpy.uint8:
            raise ValueError("texture array must have dtype numpy.uint8 with values in [0, 255].")
        
        if not isinstance(use_rgb, bool):
            raise ValueError("use_rgb must be a boolean.")

        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")

        # Duplicate points per face
        fictive_vertices = numpy.zeros((self.n_elements * self.n_vertices_per_element, 3), dtype=numpy.float64)
        for i in range(self.n_vertices_per_element):
            fictive_vertices[i::self.n_vertices_per_element, :] = self.vertices.points[self.connectivity[:, i], :]

        # Create connectivity for the fictive vertices
        fictive_connectivity = numpy.arange(self.n_elements * self.n_vertices_per_element, dtype=numpy.int64).reshape(self.n_elements, self.n_vertices_per_element)

        # Create a PyVista mesh
        n_cells = fictive_connectivity.shape[0]
        cells = numpy.hstack([numpy.full((n_cells, 1), self.n_vertices_per_element), fictive_connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, self.vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, fictive_vertices)
        
        # Set texture coordinates
        pv_mesh.active_texture_coordinates = numpy.zeros((self.n_elements * self.n_vertices_per_element, 2), dtype=numpy.float64)

        # UV coordinates per vertex of each element
        uvmap = self.elements_uvmap  # shape (M, 6)
        for i in range(self.n_vertices_per_element):
            pv_mesh.active_texture_coordinates[i::self.n_vertices_per_element, 0] = uvmap[:, 2 * i]      # u_i
            pv_mesh.active_texture_coordinates[i::self.n_vertices_per_element, 1] = uvmap[:, 2 * i + 1]  # v_i

        # Create a PyVista texture

        if texture.ndim == 2:
            color_texture = numpy.repeat(texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        elif texture.ndim == 3 and texture.shape[2] == 1:
            color_texture = numpy.repeat(texture, 3, axis=2).astype(numpy.uint8)
        elif texture.ndim == 3 and use_rgb and texture.shape[2] == 3:
            color_texture = texture
        elif texture.ndim == 3 and not use_rgb and texture.shape[2] == 3:
            gray_texture = numpy.round(numpy.dot(texture[..., :3], [0.2989, 0.5870, 0.1140])).astype(numpy.uint8)
            color_texture = numpy.repeat(gray_texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        else:
            raise ValueError("Invalid texture array shape.")
        
        pvtexture = pyvista.Texture(color_texture)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the texture
        plotter.add_mesh(
            pv_mesh, 
            texture=pvtexture,
            opacity=faces_opacity,
        )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()


    def visualize_integration_points(
            self,
            integration_points: IntegrationPoints,
            points_color: str = "red",
            points_size: int = 5,
            points_opacity: float = 1.0,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_color: str = "gray",   
            faces_opacity: float = 0.5,
            show_vertices: bool = True,
            show_edges: bool = True,
            show_faces: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the 3D surface mesh with integration points using PyVista.

        This method creates a 3D plot of the mesh, displaying its vertices, edges, and faces,
        along with the integration points overlaid on the mesh.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without integration points.
            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_texture` to visualize the texture of the mesh.

            
        Parameters
        ----------
        integration_points : IntegrationPoints
            An instance of IntegrationPoints containing the points to visualize on the mesh.
            The dimensions of the integration points must match the mesh dimensions.

        points_color : str, optional
            Color of the integration points, by default "red".

        points_size : int, optional
            Size of the integration points, by default 5.
        
        points_opacity : float, optional
            Opacity of the integration points (0.0 to 1.0), by default 1.0.

        vertices_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        vertices_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        vertices_opacity : float, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default 1.0.

        edges_color : str, optional
            Color of the edges in the mesh, by default "black".

        edges_width : int, optional
            Width of the edges in the mesh, by default 1.

        edges_opacity : float, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default 1.0.

        faces_color : str, optional
            Color of the faces in the mesh, by default "gray".

        faces_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 0.5.

        show_points : bool, optional
            Whether to display the vertices (points) of the mesh, by default True.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        show_faces : bool, optional
            Whether to display the faces of the mesh, by default True.

        title : Optional[str], optional
            Title of the plot, by default None.

        show_axes : bool, optional
            Whether to display the axes in the plot, by default True.

        show_grid : bool, optional
            Whether to display the grid in the plot, by default True.

        More Information
        -------------------------

        This method only display the mesh and integration points without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize some intersection points on it:

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )

            # Create some rays to cast
            ray_origins = numpy.random.uniform(-1, 1, (100, 3))
            ray_origins[:, 2] = 3.0  # Start above the surface
            ray_directions = numpy.tile(numpy.array([[0, 0, -1]]), (100, 1))  # Pointing downwards

            intersection_points = surface_mesh.cast_rays(ray_origins, ray_directions)

            surface_mesh.visualize_integration_points(intersection_points, points_size=8)

            
        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_integration_points_example.png
            :width: 600
            :align: center

            Example of a 3D triangular mesh visualization using the `visualize_integration_points` method.

        """
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        if not isinstance(integration_points, IntegrationPoints):
            raise ValueError("integration_points must be an instance of IntegrationPoints.")
        if integration_points.n_dimensions != self._n_dimensions:
            raise ValueError(f"integration_points must have {self._n_dimensions} dimensions.")
        
        if not isinstance(points_color, str):
            raise ValueError("Points color must be a string.")
        if not (isinstance(points_size, Number) and points_size > 0):
            raise ValueError("Points size must be a positive number.")
        if not (isinstance(points_opacity, Number) and 0.0 <= points_opacity <= 1.0):
            raise ValueError("Points opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not isinstance(faces_color, str):
            raise ValueError("Faces color must be a string.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise ValueError("show_faces must be a boolean.")
        
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
        
        # Create a PyVista mesh
        n_cells = self.n_elements
        cells = numpy.hstack([numpy.full((n_cells, 1), self.n_vertices_per_element), self.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, self.vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, self.vertices.points)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add faces if required
        if show_faces:
            plotter.add_mesh(
                pv_mesh, 
                color=faces_color, 
                opacity=faces_opacity
            )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Add integration points
        points_coordinates = self.integration_points_to_global_coordinates(integration_points)
        plotter.add_points(
            points_coordinates, 
            color=points_color, 
            point_size=points_size,
            opacity=points_opacity,
            render_points_as_spheres=True
        )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()