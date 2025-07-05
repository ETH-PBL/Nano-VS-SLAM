import numpy as np
import cv2


def calculate_pose_error(R_gt, t_gt, R_est, t_est):
    assert R_gt.shape == R_est.shape == (3, 3), (
        "Rotation matrices must be of shape (3, 3)"
    )
    assert t_gt.shape == t_est.shape == (3,), (
        "Translation vectors must be of shape (3,)"
    )

    # Simple MSE
    t_error = np.sqrt(((t_est - t_gt) ** 2).sum())
    # From https://stackoverflow.com/questions/6522108/error-between-two-rotations
    r, _ = cv2.Rodrigues(R_est.dot(R_gt.T))
    r_error = np.linalg.norm(r)

    return t_error, r_error


def calculate_error_stats(errors):
    return {
        "mean": errors.mean(),
        "sum": errors.sum(),
        "std": errors.std(),
        "max": errors.max(),
        "min": errors.min(),
    }
