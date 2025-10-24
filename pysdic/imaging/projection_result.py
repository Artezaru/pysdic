from dataclasses import dataclass
from typing import Optional
import numpy

@dataclass(slots=True)
class ProjectionResult:
    r"""
    A class to represent the result of the projection of 3D ``world_points`` to 2D ``image_points``.
    In the documentation ``N`` refers to the number of points, ``Nextrinsics`` refers to the number of extrinsic parameters, ``Ndistortion`` refers to the number of distortion parameters, and ``Nintrinsics`` refers to the number of intrinsic parameters.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. seealso::

        - :meth:`pysdic.imaging.Camera.project` for the method that performs the transformation and returns an instance of this class.

    Attributes
    ----------
    image_points : numpy.ndarray
        The projected pixel points in the image coordinate system (x, y).
        Shape (N, 2)

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix of the image points with respect to the world points.
        Shape (N, 2, 3) if `dx` is True, otherwise None.

    jacobian_dintrinsic : Optional[numpy.ndarray]
        The Jacobian matrix of the image points with respect to the intrinsic parameters.
        Shape (N, 2, Nintrinsic) if `dintrinsic` is True, otherwise None.

    jacobian_ddistortion : Optional[numpy.ndarray]
        The Jacobian matrix of the image points with respect to the distortion parameters.
        Shape (N, 2, Ndistortion) if `ddistortion` is True, otherwise None.

    jacobian_dextrinsic : Optional[numpy.ndarray]
        The Jacobian matrix of the image points with respect to the extrinsic parameters.
        Shape (N, 2, Nextrinsic) if `dextrinsic` is True, otherwise None.
        
    """
    image_points: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dintrinsic: Optional[numpy.ndarray] = None
    jacobian_ddistortion: Optional[numpy.ndarray] = None
    jacobian_dextrinsic: Optional[numpy.ndarray] = None
    
