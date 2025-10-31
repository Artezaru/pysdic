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

def create_linear_triangle_axisymmetric(
    profile_curve: Callable[[float], float] = lambda _: 1.0,
    frame: Optional[Frame] = None,
    height_bounds: tuple[float, float] = (0.0, 1.0),
    theta_bounds: tuple[float, float] = (0.0, 2.0 * numpy.pi),
    n_height: int = 10,
    n_theta: int = 10,
    closed: bool = False,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
    ) -> LinearTriangleMesh3D:
    r"""
    Create a 3D axisymmetric mesh :class:`LinearTriangleMesh3D` using a given profile curve.

    The profile curve is a function that takes a single argument (height) and returns the radius at that height.
    The returned radius must be strictly positive for all z in the range defined by ``height_bounds``.

    The ``frame`` parameter defines the orientation and the position of the mesh in 3D space.
    The axis of symmetry is aligned with the z-axis of the frame, and z=0 corresponds to the origin of the frame.
    The x-axis of the frame defines the direction of :math:`\theta=0`, and the y-axis defines the direction of :math:`\theta=\pi/2`.
    
    The ``height_bounds`` parameter defines the vertical extent of the mesh, and ``theta_bounds`` defines the angular sweep around the axis.
    ``n_height`` and ``n_theta`` determine the number of vertices in the height and angular directions, respectively.
    Nodes are uniformly distributed along both directions.

    .. note::

        - ``n_height`` and ``n_theta`` refer to the number of **vertices**, not segments.

    For example, the following code generates a mesh of a half-cylinder whose flat face is centered on the world x-axis:

    .. code-block:: python

        from pysdic.geometry import create_linear_triangle_axisymmetric
        import numpy as np

        cylinder_mesh = create_linear_triangle_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(-np.pi/4, np.pi/4),
            n_height=10,
            n_theta=20,
        )

        cylinder_mesh.visualize()

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_example.png
        :width: 400
        :align: center

        Demi-cylinder mesh with the face centered on the world x-axis.

    Nodes are ordered first in theta (indexed by ``i_T``) and then in height (indexed by ``i_H``).
    So the vertex at theta index ``i_T`` and height index ``i_H`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_H * n_theta + i_T, :]

    Each quadrilateral element is defined by the vertices:

    - :math:`(i_T, i_H)`
    - :math:`(i_T + 1, i_H)`
    - :math:`(i_T + 1, i_H + 1)`
    - :math:`(i_T, i_H + 1)`

    This quadrilateral is split into two triangles.

    If ``closed`` is True, the mesh is closed in the angular direction by adding one more set of elements connecting the last and first theta positions.
    In that case, ``theta_bounds`` are ignored and the theta starts from 0 to :math:`2\pi(1 - 1/n_{theta})`.

    To generate a closed full cylinder:

    .. code-block:: python

        cylinder_mesh = create_linear_triangle_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            n_height=10,
            n_theta=50,
            closed=True,
        )

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_example_closed.png
        :width: 400
        :align: center

        Closed cylinder mesh.

    .. seealso:: 
        
        - :class:`LinearTriangleMesh3D` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    profile_curve : Callable[[float], float], optional
        A function that takes a single height coordinate z and returns a strictly positive radius.
        The default is a function that returns 1.0 for all z.

    frame : Frame, optional
        The reference frame for the mesh. Defaults to the identity frame.

    height_bounds : tuple[float, float], optional
        The lower and upper bounds for the height coordinate. Defaults to (0.0, 1.0).
        The order determines the direction of vertex placement.
        
    theta_bounds : tuple[float, float], optional
        The angular sweep in radians. Defaults to (0.0, 2.0 * numpy.pi).
        The order determines the angular direction of vertex placement. If ``closed`` is True, this parameter is ignored.

    n_height : int, optional
        Number of vertices along the height direction. Must be more than 1. Default is 10.

    n_theta : int, optional
        Number of vertices along the angular direction. Must be more than 1. Default is 10.

    closed : bool, optional
        If True, the mesh is closed in the angular direction. Default is False.

    first_diagonal : bool, optional
        If True, the quad is split along the first diagonal (bottom-left to top-right).
        Default is True.

    direct : bool, optional
        If True, triangle vertices are ordered counterclockwise for an observer looking from outside the radius. Default is True.

    uv_layout : int, optional
        The UV mapping strategy. Default is 0.

    Returns
    -------
    TriangleMesh3D
        The generated axisymmetric mesh as a TriangleMesh3D object.
    
        
    Geometry of the mesh
    ------------------------

    Diagonal selection
    ~~~~~~~~~~~~~~~~~~

    According to the ``first_diagonal`` parameter, each quadrilateral face is split into two triangles as follows:
    
    - If ``first_diagonal`` is ``True``:

        - Triangle 1: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H)`, :math:`(i_T + 1, i_H + 1)`
        - Triangle 2: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H + 1)`, :math:`(i_T, i_H + 1)`

    - If ``first_diagonal`` is ``False``:

        - Triangle 1: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H)`, :math:`(i_T, i_H + 1)`
        - Triangle 2: :math:`(i_T, i_H + 1)`, :math:`(i_T + 1, i_H)`, :math:`(i_T + 1, i_H + 1)`

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_diagonal.png
        :width: 400
        :align: center

        Diagonal selection for splitting quadrilaterals into triangles.

    Notice that if the mesh is closed, the last column of quadrilaterals connecting the last and first theta positions are added at the end of the connectivity list.

    Triangle orientation
    ~~~~~~~~~~~~~~~~~~~~~~

    - If ``direct`` is ``True``, triangles are oriented counterclockwise for an observer looking from outside the radius (See the Diagonal selection section).
    - If ``direct`` is ``False``, triangles are oriented clockwise for an observer looking from outside the radius. Switch the order of the last two vertices in each triangle defined above.

    UV Mapping
    ~~~~~~~~~~

    The UV coordinates are generated based on the vertex positions in the mesh and uniformly distributed in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pitel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pitel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pitel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pitel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for ``uv_layout`` and their corresponding vertex mapping:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+   
    | 0               | (0, 0)                  | (0, n_h-1)              | (n_t-1, 0)               | (n_t-1, n_h-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (n_t-1, 0)              | (0, n_h-1)               | (n_t-1, n_h-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_t-1, 0)              | (0, 0)                  | (n_t-1, n_h-1)           | (0, n_h-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_h-1)              | (0, 0)                  | (n_t-1, n_h-1)           | (n_t-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_h-1)              | (n_t-1, n_h-1)          | (0, 0)                   | (n_t-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_t-1, 0)              | (n_t-1, n_h-1)          | (0, 0)                   | (0, n_h-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_t-1, n_h-1)          | (0, n_h-1)              | (n_t-1, 0)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_t-1, n_h-1)          | (n_t-1, 0)              | (0, n_h-1)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    Notice that for a closed mesh, the ``n_t - 1`` becames ``n_t`` in the table above, since the a 'virtal' last vertex is the same as the first one.
    
    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_uv_layout.png
        :width: 400
        :align: center

        UV mapping strategies for different ``uv_layout`` options.

    To check the UV mapping, you can use the following code:

    .. code-block:: python

        import numpy as np
        from pysdic.geometry import create_linear_triangle_axisymmetric
        import cv2

        cylinder_mesh = create_linear_triangle_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(0, np.pi/2),
            n_height=10,
            n_theta=20,
            uv_layout=3,
        )
    
        image = cv2.imread("lena_image.png")
        cylinder_mesh.visualize_texture(image)

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_lena_texture.png
        :width: 400
        :align: center

        UV mapping of the Lena image on a demi-cylinder mesh with ``uv_layout=3``.

    """
    # Check the input parameters
    if frame is None:
        frame = Frame.canonical()
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")
    
    if not isinstance(profile_curve, Callable):
        raise TypeError("profile_curve must be a callable function")
    
    if not isinstance(n_theta, Integral) or n_theta < 2:
        raise ValueError("n_theta must be an integer greater than 1")
    n_theta = int(n_theta)

    if not isinstance(n_height, Integral) or n_height < 2:
        raise ValueError("n_height must be an integer greater than 1")
    n_height = int(n_height)

    if not isinstance(closed, bool):
        raise TypeError("closed must be a boolean")
    if closed:
        theta_bounds = (0.0, 2.0 * numpy.pi * (1.0 - 1.0/n_theta))

    theta_bounds = numpy.array(theta_bounds, dtype=numpy.float64).flatten()
    if theta_bounds.shape != (2,):
        raise ValueError("theta_bounds must be a 2D array of shape (2,)")
    if theta_bounds[0] == theta_bounds[1]:
        raise ValueError("theta_bounds must be different")
    
    height_bounds = numpy.array(height_bounds, dtype=numpy.float64).flatten()
    if height_bounds.shape != (2,):
        raise ValueError("height_bounds must be a 2D array of shape (2,)")
    if height_bounds[0] == height_bounds[1]:
        raise ValueError("height_bounds must be different")
    
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
    theta_min = theta_bounds[0]
    theta_max = theta_bounds[1]
    height_min = height_bounds[0]
    height_max = height_bounds[1]

    # Get the indices of the vertices in the array
    index = lambda it, ih: ih*n_theta + it

    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (Nt,0) ; (0,Nh) ; (Nt,Nh)])
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
    vertices_uvmap = numpy.zeros((n_theta*n_height, 2))
    vertices = numpy.zeros((n_theta*n_height, 3))

    for ih in range(n_height):
        for it in range(n_theta):
            # Compute the coordinates of the vertex in the local frame.
            height = height_min + (height_max - height_min)*ih/(n_height-1)
            theta = theta_min + (theta_max - theta_min)*it/(n_theta-1)
            radius = profile_curve(height)

            # Convert from cylindrical to cartesian coordinates
            x = radius * numpy.cos(theta)
            y = radius * numpy.sin(theta)
            z = height

            # Convert the local point to the global frame
            local_point = numpy.array([x, y, z]).reshape((3,1))
            vertices[index(it, ih), :] = transform.transform(point=local_point).flatten()

            # Compute the uvmap (UV Mapping for vertices)
            if closed: 
                # Shift 1 into UV mapping for closed meshes
                vertices_uvmap[index(it, ih), :] = uv_mapping[0] + it/(n_theta)*(uv_mapping[1] - uv_mapping[0]) + ih/(n_height-1)*(uv_mapping[2] - uv_mapping[0])
            else:
                vertices_uvmap[index(it, ih), :] = uv_mapping[0] + it/(n_theta-1)*(uv_mapping[1] - uv_mapping[0]) + ih/(n_height-1)*(uv_mapping[2] - uv_mapping[0])

    if closed:
        # Add fictive last vertex for UV mapping
        fictive_vertices_uvmap = numpy.zeros((n_height, 2))
        for ih in range(n_height):
            fictive_vertices_uvmap[ih, :] = uv_mapping[0] + 1.0*(uv_mapping[1] - uv_mapping[0]) + ih/(n_height-1)*(uv_mapping[2] - uv_mapping[0])

    # Generate the mesh
    if not closed:
        triangles = numpy.zeros((2 * (n_theta - 1) * (n_height - 1), 3), dtype=numpy.int64)
        triangles_uvmap = numpy.zeros((2 * (n_theta - 1) * (n_height - 1), 3, 2), dtype=numpy.float32)
    else:
        triangles = numpy.zeros((2 * n_theta * (n_height - 1), 3), dtype=numpy.int64)
        triangles_uvmap = numpy.zeros((2 * n_theta * (n_height - 1), 3, 2), dtype=numpy.float32)

    for ih in range(n_height-1):
        for it in range(n_theta-1):
            # Select the nodes of the two triangles            
            if first_diagonal:
                node_1 = index(it, ih)
                node_2 = index(it+1, ih)
                node_3 = index(it+1, ih+1)
                node_4 = index(it, ih)
                node_5 = index(it+1, ih+1)
                node_6 = index(it, ih+1)
            else:
                node_1 = index(it, ih)
                node_2 = index(it+1, ih)
                node_3 = index(it, ih+1)
                node_4 = index(it, ih+1)
                node_5 = index(it+1, ih)
                node_6 = index(it+1, ih+1)
            
            # Compute the triangle indices
            triangle_1 = 2 * (node_1 - ih)
            triangle_2 = 2 * (node_1 - ih) + 1

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

    # Add the last column of triangles connecting the last and first theta positions if closed
    if closed:
        for ih in range(n_height-1):
            # Select the nodes of the two triangles            
            if first_diagonal:
                node_1 = index(n_theta-1, ih)
                uv_1 = vertices_uvmap[node_1, :]
                node_2 = index(0, ih)
                uv_2 = fictive_vertices_uvmap[ih, :]
                node_3 = index(0, ih+1)
                uv_3 = fictive_vertices_uvmap[ih+1, :]
                node_4 = index(n_theta-1, ih)
                uv_4 = vertices_uvmap[node_4, :]
                node_5 = index(0, ih+1)
                uv_5 = fictive_vertices_uvmap[ih+1, :]
                node_6 = index(n_theta-1, ih+1)
                uv_6 = vertices_uvmap[node_6, :]
            else:
                node_1 = index(n_theta-1, ih)
                uv_1 = vertices_uvmap[node_1, :]
                node_2 = index(0, ih)
                uv_2 = fictive_vertices_uvmap[ih, :]
                node_3 = index(n_theta-1, ih+1)
                uv_3 = vertices_uvmap[node_3, :]
                node_4 = index(n_theta-1, ih+1)
                uv_4 = vertices_uvmap[node_4, :]
                node_5 = index(0, ih)
                uv_5 = fictive_vertices_uvmap[ih, :]
                node_6 = index(0, ih+1)
                uv_6 = fictive_vertices_uvmap[ih+1, :]
            
            # Compute the triangle indices
            triangle_1 = 2 * (n_theta - 1) * (n_height - 1) + 2 * ih
            triangle_2 = 2 * (n_theta - 1) * (n_height - 1) + 2 * ih + 1

            # Set the triangles and their UV map
            if direct:
                triangles[triangle_1, :] = [node_1, node_2, node_3]
                triangles[triangle_2, :] = [node_4, node_5, node_6]
                triangles_uvmap[triangle_1, :, :] = [uv_1, uv_2, uv_3]
                triangles_uvmap[triangle_2, :, :] = [uv_4, uv_5, uv_6]
            else:
                triangles[triangle_1, :] = [node_1, node_3, node_2]
                triangles[triangle_2, :] = [node_4, node_6, node_5]
                triangles_uvmap[triangle_1, :, :] = [uv_1, uv_3, uv_2]
                triangles_uvmap[triangle_2, :, :] = [uv_4, uv_6, uv_5]

    triangles_uvmap = triangles_uvmap.reshape((triangles_uvmap.shape[0], 6)) # (Ntriangles, 6) - (u1,v1,u2,v2,u3,v3)

    # Prepare the mesh
    mesh = LinearTriangleMesh3D(vertices=vertices, connectivity=triangles)
    
    # Set the UV map
    mesh.elements_uvmap = triangles_uvmap
    
    return mesh