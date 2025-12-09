from dataclasses import dataclass
from typing import Optional
import numpy

@dataclass(slots=True)
class ImageProjectionResult:
    r"""
    A class to represent the result of the projection of 3D :obj:`world_points` to the image gray levels.
    In the documentation :math:`N_p` refers to the number of points, :math:`N_{\text{extrinsic}}` refers to the number of extrinsic parameters, :math:`N_{\text{distortion}}` refers to the number of distortion parameters, and :math:`N_{\text{intrinsic}}` refers to the number of intrinsic parameters.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. seealso::

        - :meth:`pysdic.View.image_project` for the method that performs the transformation and returns an instance of this class.

    Attributes
    ----------
    gray_levels : :class:`numpy.ndarray`
        The values in gray levels after the projection of the 3D world points in the image.
        Shape (:math:`N_p`, :math:`N_{\text{channels}}`) where channels is 1 for grayscale images and 3 for RGB images.

    jacobian_dx : Optional[:class:`numpy.ndarray`]
        The Jacobian matrix of the gray levels with respect to the world points.
        Shape (:math:`N_p`, :math:`N_{\text{channels}}`, 3) if :obj:`dx` is :obj:`True`, otherwise None.

    jacobian_dintrinsic : Optional[:class:`numpy.ndarray`]
        The Jacobian matrix of the gray levels with respect to the intrinsic parameters.
        Shape (:math:`N_p`, :math:`N_{\text{channels}}`, :math:`N_{\text{intrinsic}}`) if :obj:`dintrinsic` is :obj:`True`, otherwise None.

    jacobian_ddistortion : Optional[:class:`numpy.ndarray`]
        The Jacobian matrix of the gray levels with respect to the distortion parameters.
        Shape (:math:`N_p`, :math:`N_{\text{channels}}`, :math:`N_{\text{distortion}}`) if :obj:`ddistortion` is :obj:`True`, otherwise None.
        
    jacobian_dextrinsic : Optional[:class:`numpy.ndarray`]
        The Jacobian matrix of the gray levels with respect to the extrinsic parameters.
        Shape (:math:`N_p`, :math:`N_{\text{channels}}`, :math:`N_{\text{extrinsic}}`) if :obj:`dextrinsic` is :obj:`True`, otherwise None.
    """
    gray_levels: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dintrinsic: Optional[numpy.ndarray] = None
    jacobian_ddistortion: Optional[numpy.ndarray] = None
    jacobian_dextrinsic: Optional[numpy.ndarray] = None
    
