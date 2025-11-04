from dataclasses import dataclass
from typing import Optional
import numpy

@dataclass(slots=True)
class ImageProjectionResult:
    r"""
    A class to represent the result of the projection of 3D ``world_points`` to the image gray levels.
    In the documentation ``N`` refers to the number of points, ``Nextrinsics`` refers to the number of extrinsic parameters, ``Ndistortion`` refers to the number of distortion parameters, and ``Nintrinsics`` refers to the number of intrinsic parameters.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. seealso::

        - :meth:`pysdic.imaging.View.image_project` for the method that performs the transformation and returns an instance of this class.

    Attributes
    ----------
    gray_levels : numpy.ndarray
        The values in gray levels after the projection of the 3D world points in the image.
        Shape (N, channels) where channels is 1 for grayscale images and 3 for RGB images.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix of the gray levels with respect to the world points.
        Shape (N, 1, 3) if `dx` is True, otherwise None.

    jacobian_dintrinsic : Optional[numpy.ndarray]
        The Jacobian matrix of the gray levels with respect to the intrinsic parameters.
        Shape (N, 1, Nintrinsic) if `dintrinsic` is True, otherwise None.

    jacobian_ddistortion : Optional[numpy.ndarray]
        The Jacobian matrix of the gray levels with respect to the distortion parameters.
        Shape (N, 1, Ndistortion) if `ddistortion` is True, otherwise None.

    jacobian_dextrinsic : Optional[numpy.ndarray]
        The Jacobian matrix of the gray levels with respect to the extrinsic parameters.
        Shape (N, 1, Nextrinsic) if `dextrinsic` is True, otherwise None.
    """
    gray_levels: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dintrinsic: Optional[numpy.ndarray] = None
    jacobian_ddistortion: Optional[numpy.ndarray] = None
    jacobian_dextrinsic: Optional[numpy.ndarray] = None
    
