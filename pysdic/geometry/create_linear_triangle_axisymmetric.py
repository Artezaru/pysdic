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

    Nodes are ordered first in height (indexed by ``i_H``) and then in theta (indexed by ``i_T``).
    So the vertex at height index ``i_H`` and angular index ``i_T`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_T * n_height + i_H, :]

    Each quadrilateral element is defined by the vertices:

    - :math:`(i_H, i_T)`
    - :math:`(i_H + 1, i_T)`
    - :math:`(i_H + 1, i_T + 1)`
    - :math:`(i_H, i_T + 1)`

    This quadrilateral is then split into two triangles depending on the value of ``first_diagonal``:

    - If ``first_diagonal`` is ``True``:

        - Triangle 1: :math:`(i_H, i_T)`, :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T + 1)`
        - Triangle 2: :math:`(i_H, i_T)`, :math:`(i_H + 1, i_T + 1)`, :math:`(i_H + 1, i_T)`

    - If ``first_diagonal`` is ``False``:

        - Triangle 1: :math:`(i_H, i_T)`, :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T)`
        - Triangle 2: :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T + 1)`, :math:`(i_H + 1, i_T)`

    These triangles are oriented in a direct (counterclockwise) order by default (for an observer outside the cylinder).
    If ``direct`` is False, the orientation is reversed by swapping the second and third vertices in each triangle.

    If ``closed`` is True, the mesh is closed in the angular direction.
    In that case, ``theta_bounds`` should be set to:

    .. math::

        (\theta_0, \theta_0 \pm 2\pi (1 - \frac{1}{n_{theta}}))

    to avoid duplicating vertices at the seam.

    To generate a closed full cylinder:

    .. code-block:: python

        cylinder_mesh = create_linear_triangle_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(0.0, 2.0 * np.pi * (1 - 1.0 / 50)),
            n_height=10,
            n_theta=50,
            closed=True,
        )

    .. figure:: ../../../pysdic/resources/create_linear_triangle_axisymmetric_example_closed.png
        :width: 400
        :align: center

        Closed cylinder mesh.

    The UV coordinates are generated based on the vertex positions in the mesh and uniformly distributed in the range [0, 1] for the OpenGL texture mapping convention.
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.
    The following options are available for ``uv_layout``:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+   
    | 0               | (0, 0)                  | (n_height-1, 0)         | (0, n_theta-1)           | (n_height-1, n_theta-1)  |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (0, n_theta-1)          | (n_height-1, 0)          | (n_height-1, n_theta-1)  |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_height-1, 0)         | (0, 0)                  | (n_height-1, n_theta-1)  | (0, n_theta-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_theta-1)          | (0, 0)                  | (n_height-1, n_theta-1)  | (n_height-1, 0)          |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_theta-1)          | (n_height-1, n_theta-1) | (0, 0)                   | (n_height-1, 0)          |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_height-1, 0)         | (n_height-1, n_theta-1) | (0, 0)                   | (0, n_theta-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_height-1, n_theta-1) | (0, n_theta-1)          | (n_height-1, 0)          | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_height-1, n_theta-1) | (n_height-1, 0)         | (0, n_theta-1)           | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    Notice that for a closed mesh, the ``N - 1`` becames ``N`` in the table above, since the a 'virtal' last vertex is the same as the first one.
    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

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
        The angular sweep in radians. Defaults to (-numpy.pi, numpy.pi).
        The order determines the angular direction of vertex placement.
    
    n_height : int, optional
        Number of vertices along the height direction. Must be more than 1. Default is 10.
    
    n_theta : int, optional
        Number of vertices along the angular direction. Must be more than 1. Default is 10.
    
    closed : bool, optional
        If True, the mesh is closed in the angular direction. Default is False.

    first_diagonal : bool, optional
        If True, the quad is split along the first diagonal (bottom-left to top-right). Default is True.
    
    direct : bool, optional
        If True, triangle vertices are ordered counterclockwise. Default is True.

    uv_layout : int, optional
        The UV mapping strategy. Default is 0.

    Returns
    -------
    TriangleMesh3D
        The generated axisymmetric mesh as a TriangleMesh3D object.
    """
    # Check the input parameters
    if frame is None:
        frame = Frame.canonical()
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")
    
    if not isinstance(profile_curve, Callable):
        raise TypeError("profile_curve must be a callable function")
    
    height_bounds = numpy.array(height_bounds, dtype=numpy.float64).flatten()
    if height_bounds.shape != (2,):
        raise ValueError("height_bounds must be a 2D array of shape (2,)")
    if height_bounds[0] == height_bounds[1]:
        raise ValueError("height_bounds must be different")
    
    theta_bounds = numpy.array(theta_bounds, dtype=numpy.float64).flatten()
    if theta_bounds.shape != (2,):
        raise ValueError("theta_bounds must be a 2D array of shape (2,)")
    if theta_bounds[0] == theta_bounds[1]:
        raise ValueError("theta_bounds must be different")
    
    if not isinstance(n_height, Integral) or n_height < 2:
        raise ValueError("n_height must be an integer greater than 1")
    n_height = int(n_height)

    if not isinstance(n_theta, Integral) or n_theta < 2:
        raise ValueError("n_theta must be an integer greater than 1")
    n_theta = int(n_theta)

    if not isinstance(closed, bool):
        raise TypeError("closed must be a boolean")
    if closed and (abs(theta_bounds[0] - theta_bounds[1]) - 2.0*numpy.pi*(1 - 1/n_theta)) > 1e-6:
        print("Warning: The theta bounds are not set to the closed condition (theta_max = theta_min + 2*pi*(1 - 1/n_theta)). The mesh will be closed in the theta direction but the output can be unexpected.")

    if not isinstance(first_diagonal, bool):
        raise TypeError("first_diagonal must be a boolean")
    
    if not isinstance(direct, bool):
        raise TypeError("direct must be a boolean")
    
    if not isinstance(uv_layout, Integral) or uv_layout < 0 or uv_layout > 7:
        raise ValueError("uv_layout must be an integer between 0 and 7")
    uv_layout = int(uv_layout)
    
    # Generate the transform
    transform = FrameTransform(input_frame=frame, output_frame=Frame())

    # Extract the parameters
    height_min = height_bounds[0]
    height_max = height_bounds[1]
    theta_min = theta_bounds[0]
    theta_max = theta_bounds[1]

    # Get the indices of the vertices in the array
    index = lambda ih, it: it*n_height + ih

    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (0,Nt) ; (Nh,0) ; (Nh,Nt)])
    lower_left = numpy.array([0.0, 0.0])
    lower_right = numpy.array([1.0, 0.0])
    upper_left = numpy.array([0.0, 1.0])
    upper_right = numpy.array([1.0, 1.0])
    if uv_layout == 0:
        uv_mapping = [lower_left, lower_right, upper_left, upper_right]
    elif uv_layout == 1:
        uv_mapping = [lower_left, upper_left, lower_right, upper_right]
    elif uv_layout == 2:
        uv_mapping = [upper_left, upper_right, lower_left, lower_right]
    elif uv_layout == 3:
        uv_mapping = [upper_left, lower_left, upper_right, lower_right]
    elif uv_layout == 4:
        uv_mapping = [lower_right, lower_left, upper_right, upper_left]
    elif uv_layout == 5:
        uv_mapping = [lower_right, upper_right, lower_left, upper_left]
    elif uv_layout == 6:
        uv_mapping = [upper_right, upper_left, lower_right, lower_left]
    elif uv_layout == 7:
        uv_mapping = [upper_right, lower_right, upper_left, lower_left]

    # Generate the vertices
    vertices = numpy.zeros((n_height*n_theta, 3))

    for it in range(n_theta):
        for ih in range(n_height):
            # Compute the coordinates of the vertex in the local frame.
            theta = theta_min + (theta_max - theta_min)*it/(n_theta-1)
            height = height_min + (height_max - height_min)*ih/(n_height-1)
            rho = profile_curve(height)
            x = rho*numpy.cos(theta)
            y = rho*numpy.sin(theta)
            z = height

            # Convert the local point to the global frame
            local_point = numpy.array([x, y, z]).reshape((3,1))
            vertices[index(ih, it), :] = transform.transform(point=local_point).flatten()

    # Generate the mesh
    triangles = []

    for it in range(n_theta-1):
        for ih in range(n_height-1):
            if first_diagonal and direct:
                triangles.append([index(ih, it), index(ih, it+1), index(ih+1, it+1)])
                triangles.append([index(ih, it), index(ih+1, it+1), index(ih+1, it)])

            elif first_diagonal and not direct:
                triangles.append([index(ih, it), index(ih+1, it+1), index(ih, it+1)])
                triangles.append([index(ih, it), index(ih+1, it), index(ih+1, it+1)])

            elif not first_diagonal and direct:
                triangles.append([index(ih, it), index(ih, it+1), index(ih+1, it)])
                triangles.append([index(ih, it+1), index(ih+1, it+1), index(ih+1, it)])

            elif not first_diagonal and not direct:
                triangles.append([index(ih, it), index(ih+1, it), index(ih, it+1)])
                triangles.append([index(ih, it+1), index(ih+1, it), index(ih+1, it+1)])

    if closed:
        for ih in range(n_height-1):
            if first_diagonal and direct:
                triangles.append([index(ih, n_theta-1), index(ih, 0), index(ih+1, 0)])
                triangles.append([index(ih, n_theta-1), index(ih+1, 0), index(ih+1, n_theta-1)])

            elif first_diagonal and not direct:
                triangles.append([index(ih, n_theta-1), index(ih+1, 0), index(ih, 0)])
                triangles.append([index(ih, n_theta-1), index(ih+1, n_theta-1), index(ih+1, 0)])

            elif not first_diagonal and direct:
                triangles.append([index(ih, n_theta-1), index(ih, 0), index(ih+1, n_theta-1)])
                triangles.append([index(ih, 0), index(ih+1, 0), index(ih+1, n_theta-1)])

            elif not first_diagonal and not direct:
                triangles.append([index(ih, n_theta-1), index(ih+1, n_theta-1), index(ih, 0)])
                triangles.append([index(ih, 0), index(ih+1, n_theta-1), index(ih+1, 0)])

    triangles = numpy.array(triangles)

    # Generate the UV map
    triangles_uvmap = numpy.zeros((triangles.shape[0], 6), dtype=numpy.float64) # [u1, v1, u2, v2, u3, v3]

    for itri, triangle in enumerate(triangles):
        # Get the vertices of the triangle
        v1 = vertices[triangle[0], :]
        v2 = vertices[triangle[1], :]
        v3 = vertices[triangle[2], :]

        # Get the index of the height and theta of the vertices
        it1, ih1 = triangle[0] // n_height, triangle[0] % n_height
        it2, ih2 = triangle[1] // n_height, triangle[1] % n_height
        it3, ih3 = triangle[2] // n_height, triangle[2] % n_height

        if not closed:
            n_height_up = n_height - 1
            n_theta_up = n_theta - 1

        else:
            n_height_up = n_height
            n_theta_up = n_theta

        # Compute the UV coordinates for each vertex (depending on the uv_layout and if we are in the last column)
        if (0 in [it1, it2, it3] and (n_theta - 1) in [it1, it2, it3]): 
            if it1 == 0:
                it1 = n_theta # Switch to the last column
            if it2 == 0:
                it2 = n_theta
            if it3 == 0:
                it3 = n_theta

        triangles_uvmap[itri, 0:2] = uv_mapping[0] + ih1/n_height_up*(uv_mapping[2] - uv_mapping[0]) + it1/n_theta_up*(uv_mapping[1] - uv_mapping[0])
        triangles_uvmap[itri, 2:4] = uv_mapping[0] + ih2/n_height_up*(uv_mapping[2] - uv_mapping[0]) + it2/n_theta_up*(uv_mapping[1] - uv_mapping[0])
        triangles_uvmap[itri, 4:6] = uv_mapping[0] + ih3/n_height_up*(uv_mapping[2] - uv_mapping[0]) + it3/n_theta_up*(uv_mapping[1] - uv_mapping[0])

    # Prepare the triangles for the mesh
    mesh = LinearTriangleMesh3D(vertices=vertices, connectivity=triangles)
    mesh.elements_uvmap = triangles_uvmap
    return mesh