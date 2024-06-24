import sys

import numpy as np

sys.path.append("/PoseForecasters/")
import utils_show

# ==================================================================================================


def visualize(inp, target=None, pred=None):

    joint_names = [
        # "hip_middle",
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        # "shoulder_middle",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ]

    if inp is not None:
        poses_input = inp[0]
        poses_input = poses_input[:, :, [0, 2, 1]]
    else:
        poses_input = np.array([])

    if target is not None:
        poses_target = target[0]
        poses_target = poses_target[:, :, [0, 2, 1]]
    else:
        poses_target = np.array([])

    if pred is not None:
        poses_pred = pred[0]
        poses_pred = poses_pred[:, :, [0, 2, 1]]
    else:
        poses_pred = np.array([])

    utils_show.visualize_pose_trajectories(
        poses_input, poses_target, poses_pred, joint_names, {}
    )
    utils_show.show()
