from dataclasses import dataclass
from math import sin, cos


@dataclass
class Pose:
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
