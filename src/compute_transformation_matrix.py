from dataclasses import dataclass
from math import cos, sin

import numpy as np
from numpy.typing import NDArray


@dataclass
class QPose:
    """
    Represents a 6-DOF vehicle pose in the local NED frame.
    (NED represents x=North, y=easy, z=down)
    Read more here: https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates

    Position is expressed in meters. Orientation is represented via quaternion.
    """

    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass
class EPose:
    """
    Represents a 6-DOF vehicle pose in the local NED frame.
    (NED represents x=North, y=easy, z=down)
    Read more here: https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates

    Position is expressed in meters. Orientation is represented via Euler angles.
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def rpy_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """
    Converts roll, pitch, and yaw (Euler) angles to quaternions.

    Parameters
    ----------
    roll
        Roll angle in radians
    pitch
        Pitch angle in radians
    yaw
        Yaw angle in radains

    Returns
    -------
    qw
        Real magnitude
    qx
        Imaginary x vector
    qy
        Imaginary y vector
    qz
        Imaginary z vector

    Notes
    -----
    Conversion can be found here:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    in section "Angles (in ZYX sequence) to quaternion conversion".
    """

    cr = cos(roll / 2)
    sr = sin(roll / 2)
    cp = cos(pitch / 2)
    sp = sin(pitch / 2)
    cy = cos(yaw / 2)
    sy = sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


def get_rotation_matrix(pose: EPose | QPose) -> NDArray[np.float64]:
    """
    Builds 3x3 rotation matrix from a given pose.

    Parameters
    ----------
    pose
        Pose object containing orentation and translation data.

    Returns
    -------
    R
        3x3 Rotation matrix
    """

    match pose:
        case EPose():
            roll = pose.roll
            pitch = pose.pitch
            yaw = pose.yaw

            cr = cos(roll)
            sr = sin(roll)
            cp = cos(pitch)
            sp = sin(pitch)
            cy = cos(yaw)
            sy = sin(yaw)

            # Used equation 3.42 from https://msl.cs.uiuc.edu/planning/node102.html
            R = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )

            return R

        case QPose():
            raise NotImplementedError


def get_transformation_matrix(pose: EPose | QPose) -> NDArray[np.float64]:
    """
    Builds 4x4 transformation matrix from a given pose.

    Parameters
    ----------
    pose
        Pose object containing orentation and translation data.

    Returns
    -------
    T
        4x4 Transformation matrix
    """
    R = get_rotation_matrix(pose)
    d = np.array([pose.x, pose.y, pose.z])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = d

    return T


def transformation_matrix_inverse(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes inverse of transformation matrix without expensive inverse operator.

    Parameters
    ----------
    T
        4x4 Transformation matrix

    Returns
    -------
    T_inv
        4x4 Inverse transformation matrix
    """
    R = T[:3, :3]
    d = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ d)

    return T_inv


def get_relative_transformation_matrix(
    pose1: EPose | QPose,
    pose2: EPose | QPose,
) -> NDArray[np.float64]:
    """
    Computes relative transformation matrix between two poses

    Parameters
    ----------
    pose1
        Pose object containing orentation and translation data.
    pose2
        Different pose object containing orentation and translation data.

    Returns
    -------
    T
        Transformation matrix of pose1 relative to pose2 (i.e. {B2}^T_{B1}).
    """

    T1 = get_transformation_matrix(pose1)
    T2 = get_transformation_matrix(pose2)
    T2_inv = transformation_matrix_inverse(T2)

    return T2_inv @ T1
