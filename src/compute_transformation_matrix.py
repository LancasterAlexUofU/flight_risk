from dataclasses import dataclass
from math import sin, cos


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
    qx
        Imaginary x vector
    qy
        Imaginary y vector
    qz
        Imaginary z vector
    qw
        Real magnitude

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

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return qx, qy, qz, qw
