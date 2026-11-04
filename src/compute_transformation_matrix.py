from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R


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


def get_rotation(pose: EPose | QPose) -> R:
    match pose:
        case EPose():
            return R.from_euler("zyx", [pose.yaw, pose.pitch, pose.roll])

        case QPose():
            return R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw])


def get_transformation(pose: EPose | QPose) -> Tf:
    R = get_rotation(pose)
    d = np.array([pose.x, pose.y, pose.z])

    return Tf.from_components(d, R)


def get_relative_transformation(
    pose1: EPose | QPose,
    pose2: EPose | QPose,
) -> Tf:
    """
    Computes relative transformation matrix between two poses.

    Notes
    -----
    If pose1 = B1, and pose2 = B2, then the returned Tf would be read as
    "Frame B1 expressed in Frame B2."
    """

    tf1 = get_transformation(pose1)
    tf2_inv = get_transformation(pose2).inv()

    return tf2_inv * tf1


def calculate_lower_bound_distance_threshold(linear_velocity, lidar_frequency):
    """
    `distance_threshold` must be be strictly greater than the distance the robot
    travels between consecutive LiDAR scans.
    """


def get_nn_points_gated(
    points_B1: NDArray, points_B2: NDArray, tf_B2_B1: Tf, distance_threshold: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Notes
    -----
    Following scipy Tf naming convention:
    tf_B2_B1 is read frame B1 expressed in frame B2.
    """
    points_B2_expected = tf_B2_B1.apply(points_B1)

    tree = KDTree(points_B2)
    distance, index = tree.query(points_B2_expected)

    threshold_mask = distance <= distance_threshold
    matched_points_B2_indices_in_threshold = index[threshold_mask]
    matched_points_B2_in_threshold = points_B2[matched_points_B2_indices_in_threshold]

    points_B2_expected_threshold = points_B2_expected[threshold_mask]

    return matched_points_B2_in_threshold, points_B2_expected_threshold
