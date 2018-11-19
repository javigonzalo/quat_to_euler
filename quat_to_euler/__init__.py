# Copyright (c) 2018, Javier González
#

#import numpy as np
import math

from quaternion import (quaternion, as_float_array)

__doc_title__ = "Quaternion to Euler for NumPy-quaternion"
__doc__ = "Adds a quaternion to Euler Unity-like convention method to NumPy."

__all__ = ['coords_to_spherical', 'quat_to_euler', 'as_float_array']

def quat_to_euler(q):
    """
    Another convention relative to euler angles instead of moble-quaternion 'as_euler_angles' function
    """
    alpha_beta_gamma = np.empty( q.shape + (3,), dtype=np.float )

    q = as_float_array( q )
    if (q[..., 0] > 1) or (q[..., 1] > 1) or (q[..., 2] > 1) or (q[..., 3] > 1):
        norm = math.sqrt(
            q[..., 0] * q[..., 0] + q[..., 1] * q[..., 1] + q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3] )
        q[..., 0] = q[..., 0] / norm
        q[..., 1] = q[..., 1] / norm
        q[..., 2] = q[..., 2] / norm
        q[..., 3] = q[..., 3] / norm

    test = q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0]

    if test > 0.499:  # Singularities, same as Unity quaternion.eulerangles <-- The same singularities
        alpha_beta_gamma[..., 1] = 360 / math.pi * np.arctan2( q[..., 1], q[..., 0] )
        alpha_beta_gamma[..., 2] = 90.0
        alpha_beta_gamma[..., 0] = 0.0
    elif test < -0.499:
        alpha_beta_gamma[..., 1] = -360 / math.pi * np.arctan2( q[..., 1], q[..., 0] )
        alpha_beta_gamma[..., 2] = -90.0
        alpha_beta_gamma[..., 0] = 0.0
    else:
        a = np.arctan2( 2 * q[..., 1] * q[..., 0] - 2 * q[..., 2] * q[..., 3],
                        1 - 2 * (q[..., 1] * q[..., 1]) - 2 * (q[..., 3] * q[..., 3]) )
        b = np.arctan2( 2 * q[..., 2] * q[..., 0] - 2 * q[..., 1] * q[..., 3],
                        1 - 2 * (q[..., 2] * q[..., 2]) - 2 * (q[..., 3] * q[..., 3]) )
        c = np.arcsin( 2 * q[..., 1] * q[..., 2] + 2 * q[..., 3] * q[..., 0] )  # Signo +
        alpha_beta_gamma[..., 0] = math.degrees( a )
        alpha_beta_gamma[..., 1] = math.degrees( b )
        alpha_beta_gamma[..., 2] = math.degrees( c )
    return alpha_beta_gamma


def coords_to_spherical(q):
    """
    Return the spherical coordinates corresponding to this quaternion using our euler angles convention
    """
    return quat_to_euler( q )[..., 1::-1]