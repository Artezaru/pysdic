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

from typing import Callable, Optional
from numbers import Integral

import numpy
from py3dframe import Frame, FrameTransform

from .linear_triangle_mesh_3d import LinearTriangleMesh3D

def create_linear_triangle_heightmap(
    height_function: Callable[[float, float], float] = lambda x, y: 0.0,
    frame: Optional[Frame] = None,
    x_bounds: tuple[float, float] = (-1.0, 1.0),
    y_bounds: tuple[float, float] = (-1.0, 1.0),
    n_x: int = 10,
    n_y: int = 10,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
) -> LinearTriangleMesh3D:
    r"""
    Create a 3D :class:`LinearTriangleMesh3D` XY-plane mesh with variable height defined by a surface function.

    The surface is defined by a function that takes two arguments (x and y) and returns a scalar height z.
    The returned value is interpreted as the vertical position of the surface at that point.

    The ``frame`` parameter defines the orientation and the position of the mesh in 3D space.
    The (x, y) grid is centered on the frame origin, lying in the local XY plane.
    The height (z) is applied along the local Z-axis of the frame.

    The ``x_bounds`` and ``y_bounds`` parameters define the rectangular domain over which the mesh is generated.
    ``n_x`` and ``n_y`` determine the number of vertices along the x and y directions, respectively.
    Nodes are uniformly distributed across both directions.

    .. note::

        - ``n_x`` and ``n_y`` refer to the number of **vertices**, not segments.
        - The height function must return a finite scalar value for every (x, y) in the specified domain.

    For example, the following code generates a sinusoidal surface mesh centered on the world origin:

    .. code-block:: python

        from pysdic.geometry import create_linear_triangle_heightmap
        import numpy as np

        surface_mesh = create_linear_triangle_heightmap(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            n_x=50,
            n_y=50,
        )

        surface_mesh.visualize()

    .. figure:: ../../../pysdic/resources/create_linear_triangle_heightmap_example.png
        :width: 400
        :align: center

        Sinusoidal height map over a square domain centered at the origin.

    Nodes are ordered first along the x direction, then along the y direction.
    So the vertex at y index ``i_Y`` and x index ``i_X`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_Y * n_x + i_X, :]

    Each quadrilateral face is defined by the vertices:

    - :math:`(i_X, i_Y)`
    - :math:`(i_X + 1, i_Y)`
    - :math:`(i_X + 1, i_Y + 1)`
    - :math:`(i_X, i_Y + 1)`

    This quadrilateral is split into two triangles.

    .. seealso:: 
        
        - :class:`LinearTriangleMesh3D` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    height_function : Callable[[float, float], float]
        A function that takes x and y coordinates and returns the corresponding height (z).
        
    frame : Frame, optional
        The reference frame for the mesh. Defaults to the canonical frame.
        
    x_bounds : tuple[float, float], optional
        The lower and upper bounds for the x coordinate. Default is (-1.0, 1.0).
        
    y_bounds : tuple[float, float], optional
        The lower and upper bounds for the y coordinate. Default is (-1.0, 1.0).

    n_x : int, optional
        Number of vertices along the x direction. Must be more than 1. Default is 10.

    n_y : int, optional
        Number of vertices along the y direction. Must be more than 1. Default is 10.

    first_diagonal : bool, optional
        If True, the quad is split along the first diagonal (bottom-left to top-right). Default is True.

    direct : bool, optional
        If True, triangle vertices are ordered counterclockwise for an observer looking from above. Default is True.

    uv_layout : int, optional
        The UV mapping strategy. Default is 0.

    Returns
    -------
    TriangleMesh3D
        The generated surface mesh as a TriangleMesh3D object.


    Geometry of the mesh
    ------------------------

    Diagonal selection
    ~~~~~~~~~~~~~~~~~~~

    According to the ``first_diagonal`` parameter, each quadrilateral face is split into two triangles as follows:
    
    - If ``first_diagonal`` is ``True``:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X + 1, i_Y + 1)`
        - Triangle 2: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y + 1)`, :math:`(i_X, i_Y + 1)`

    - If ``first_diagonal`` is ``False``:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X, i_Y + 1)`
        - Triangle 2: :math:`(i_X, i_Y + 1)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X + 1, i_Y + 1)`

    .. figure:: ../../../pysdic/resources/create_linear_triangle_heightmap_diagonal.png
        :width: 400
        :align: center

        Diagonal selection for splitting quadrilaterals into triangles.

    Triangle orientation
    ~~~~~~~~~~~~~~~~~~~~~~

    - If ``direct`` is ``True``, triangles are oriented counterclockwise for an observer looking from above (See the Diagonal selection section).
    - If ``direct`` is ``False``, triangles are oriented clockwise for an observer looking from above. Switch the order of the last two vertices in each triangle defined above.

    UV Mapping
    ~~~~~~~~~~

    The UV coordinates are generated based on the vertex positions in the mesh and uniformly distributed in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pixel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pixel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pixel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pixel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for ``uv_layout`` and their corresponding vertex mapping:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+   
    | 0               | (0, 0)                  | (0, n_y-1)              | (n_x-1, 0)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (n_x-1, 0)              | (0, n_y-1)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_x-1, 0)              | (0, 0)                  | (n_x-1, n_y-1)           | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_y-1)              | (0, 0)                  | (n_x-1, n_y-1)           | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_y-1)              | (n_x-1, n_y-1)          | (0, 0)                   | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_x-1, 0)              | (n_x-1, n_y-1)          | (0, 0)                   | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_x-1, n_y-1)          | (0, n_y-1)              | (n_x-1, 0)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_x-1, n_y-1)          | (n_x-1, 0)              | (0, n_y-1)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

    .. figure:: ../../../pysdic/resources/create_linear_triangle_heightmap_uv_layout.png
        :width: 400
        :align: center

        UV mapping strategies for different ``uv_layout`` options.

    To check the UV mapping, you can use the following code:

    .. code-block:: python

        import numpy as np
        from pysdic.geometry import create_linear_triangle_heightmap
        import cv2

        surface_mesh = create_linear_triangle_heightmap(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            n_x=50,
            n_y=50,
            uv_layout=3,
        ) 

        image = cv2.imread("lena_image.png")
        surface_mesh.visualize_texture(image)

    .. figure:: ../../../pysdic/resources/create_linear_triangle_heightmap_lena_texture.png
        :width: 400
        :align: center

        UV mapping of the Lena image on a sinusoidal height map.

    """
    # Check the input parameters
    if frame is None:
        frame = Frame.canonical()
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")
    
    if not isinstance(height_function, Callable):
        raise TypeError("height_function must be a callable function")
    
    x_bounds = numpy.array(x_bounds, dtype=numpy.float64).flatten()
    if x_bounds.shape != (2,):
        raise ValueError("x_bounds must be a 2D array of shape (2,)")
    if x_bounds[0] == x_bounds[1]:
        raise ValueError("x_bounds must be different")
    
    y_bounds = numpy.array(y_bounds, dtype=numpy.float64).flatten()
    if y_bounds.shape != (2,):
        raise ValueError("y_bounds must be a 2D array of shape (2,)")
    if y_bounds[0] == y_bounds[1]:
        raise ValueError("y_bounds must be different")
    
    if not isinstance(n_x, Integral) or n_x < 2:
        raise ValueError("n_x must be an integer greater than 1")
    n_x = int(n_x)

    if not isinstance(n_y, Integral) or n_y < 2:
        raise ValueError("n_y must be an integer greater than 1")
    n_y = int(n_y)

    if not isinstance(first_diagonal, bool):
        raise TypeError("first_diagonal must be a boolean")
    
    if not isinstance(direct, bool):
        raise TypeError("direct must be a boolean")
    
    if not isinstance(uv_layout, Integral) or uv_layout < 0 or uv_layout > 7:
        raise ValueError("uv_layout must be an integer between 0 and 7")
    uv_layout = int(uv_layout)
    
    # Generate the transform
    transform = FrameTransform(input_frame=frame, output_frame=Frame.canonical())

    # Extract the parameters
    x_min = x_bounds[0]
    x_max = x_bounds[1]
    y_min = y_bounds[0]
    y_max = y_bounds[1]

    # Get the indices of the vertices in the array
    index = lambda ix, iy: iy*n_x + ix

    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (Nx,0) ; (0,Ny) ; (Nx,Ny)])
    lower_left = numpy.array([0.0, 0.0])
    upper_left = numpy.array([0.0, 1.0])
    lower_right = numpy.array([1.0, 0.0])
    upper_right = numpy.array([1.0, 1.0])

    if uv_layout == 0:
        uv_mapping = [lower_left, lower_right, upper_left, upper_right]
    elif uv_layout == 1:
        uv_mapping = [lower_left, upper_left, lower_right, upper_right]
    elif uv_layout == 2:
        uv_mapping = [upper_left, lower_left, upper_right, lower_right]
    elif uv_layout == 3:
        uv_mapping = [upper_left, upper_right, lower_left, lower_right]
    elif uv_layout == 4:
        uv_mapping = [lower_right, upper_right, lower_left, upper_left]
    elif uv_layout == 5:
        uv_mapping = [lower_right, lower_left, upper_right, upper_left]
    elif uv_layout == 6:
        uv_mapping = [upper_right, lower_right, upper_left, lower_left]
    elif uv_layout == 7:
        uv_mapping = [upper_right, upper_left, lower_right, lower_left]

    # Generate the vertices
    vertices_uvmap = numpy.zeros((n_x*n_y, 2))
    vertices = numpy.zeros((n_x*n_y, 3))

    for iy in range(n_y):
        for ix in range(n_x):
            # Compute the coordinates of the vertex in the local frame.
            y = y_min + (y_max - y_min)*iy/(n_y-1)
            x = x_min + (x_max - x_min)*ix/(n_x-1)
            z = height_function(x, y)

            # Convert the local point to the global frame
            local_point = numpy.array([x, y, z]).reshape((3,1))
            vertices[index(ix, iy), :] = transform.transform(point=local_point).flatten()

            # Compute the uvmap (UV Mapping for vertices)
            vertices_uvmap[index(ix, iy), :] = uv_mapping[0] + ix/(n_x-1)*(uv_mapping[1] - uv_mapping[0]) + iy/(n_y-1)*(uv_mapping[2] - uv_mapping[0])


    # Generate the mesh
    triangles = numpy.zeros((2 * (n_x - 1) * (n_y - 1), 3), dtype=numpy.int64)
    triangles_uvmap = numpy.zeros((2 * (n_x - 1) * (n_y - 1), 3, 2), dtype=numpy.float32)

    for iy in range(n_y-1):
        for ix in range(n_x-1):
            # Select the nodes of the two triangles            
            if first_diagonal:
                node_1 = index(ix, iy)
                node_2 = index(ix+1, iy)
                node_3 = index(ix+1, iy+1)
                node_4 = index(ix, iy)
                node_5 = index(ix+1, iy+1)
                node_6 = index(ix, iy+1)
            else:
                node_1 = index(ix, iy)
                node_2 = index(ix+1, iy)
                node_3 = index(ix, iy+1)
                node_4 = index(ix, iy+1)
                node_5 = index(ix+1, iy)
                node_6 = index(ix+1, iy+1)
            
            # Compute the triangle indices
            triangle_1 = 2 * (node_1 - iy)
            triangle_2 = 2 * (node_1 - iy) + 1

            # Set the triangles and their UV map
            if direct:
                triangles[triangle_1, :] = [node_1, node_2, node_3]
                triangles[triangle_2, :] = [node_4, node_5, node_6]
                triangles_uvmap[triangle_1, :, :] = [vertices_uvmap[node_1, :], vertices_uvmap[node_2, :], vertices_uvmap[node_3, :]]
                triangles_uvmap[triangle_2, :, :] = [vertices_uvmap[node_4, :], vertices_uvmap[node_5, :], vertices_uvmap[node_6, :]]
            else:
                triangles[triangle_1, :] = [node_1, node_3, node_2]
                triangles[triangle_2, :] = [node_4, node_6, node_5]
                triangles_uvmap[triangle_1, :, :] = [vertices_uvmap[node_1, :], vertices_uvmap[node_3, :], vertices_uvmap[node_2, :]]
                triangles_uvmap[triangle_2, :, :] = [vertices_uvmap[node_4, :], vertices_uvmap[node_6, :], vertices_uvmap[node_5, :]]

    triangles_uvmap = triangles_uvmap.reshape((triangles_uvmap.shape[0], 6)) # (Ntriangles, 6) - (u1,v1,u2,v2,u3,v3)

    # Prepare the mesh
    mesh = LinearTriangleMesh3D(vertices=vertices, connectivity=triangles)
    
    # Set the UV map
    mesh.elements_uvmap = triangles_uvmap
    
    return mesh