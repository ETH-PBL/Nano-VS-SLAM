# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
from tqdm import tqdm

from .descriptor import compute_homography, compute_matching_score
from .detector import compute_repeatability
import csv


def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = {}
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs[t] = np.round((np.trapz(r, x=e) / t), 4)
    return aucs


class AUCMetric:
    def __init__(self, thresholds, elements=[]):
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        self._elements += [tensor]

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)


def write_to_file(row):
    """
    Write row to csv file
    :param row:
    :return:
    """
    with open("result.csv", "a", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(row)


def evaluate_keypoint_net(
    data_loader,
    keypoint_net,
    output_shape=(320, 240),
    top_k=300,
    debug=False,
    offset=0,
    tflite=False,
):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple [W,H]
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False

    conf_threshold = 0.7
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    auc_h_ransac = AUCMetric([1, 3, 5])
    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if i < offset:
                continue
            # if use_color:
            #     image = to_color_normalized(sample['image'].to(keypoint_net.device))
            #     warped_image = to_color_normalized(sample['image_aug'].to(keypoint_net.device))
            # else:
            #     image = to_gray_normalized(sample['image'].to(keypoint_net.device))
            #     warped_image = to_gray_normalized(sample['image_aug'].to(keypoint_net.device))
            image = sample["image"].to(keypoint_net.device)
            warped_image = sample["image_aug"].to(keypoint_net.device)
            B, C, H, W = image.shape

            out_1 = keypoint_net(image)
            out_1 = keypoint_net.post_processing(out_1, H, W)
            score_1, coord_1, desc1 = out_1["score"], out_1["coord"], out_1["feat"]

            out_2 = keypoint_net(warped_image)
            out_2 = keypoint_net.post_processing(out_2, H, W)
            score_2, coord_2, desc2 = out_2["score"], out_2["coord"], out_2["feat"]

            B, C, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()

            if not tflite:
                desc1 = desc1.view(C, -1).t().cpu().numpy()
                desc2 = desc2.view(C, -1).t().cpu().numpy()
            else:
                # use this in case of tflite
                desc1 = desc1.reshape(-1, C).cpu().numpy()
                desc2 = desc2.reshape(-1, C).cpu().numpy()

            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]

            # Prepare data for eval
            data = {
                "image": sample["image"].numpy().squeeze(),
                "image_shape": output_shape[::-1],  # convert to [H,W]
                "image_aug": sample["image_aug"].numpy().squeeze(),
                "homography": sample["homography"].squeeze().numpy(),
                "prob": score_1,
                "warped_prob": score_2,
                "desc": desc1,
                "warped_desc": desc2,
            }

            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(
                data, keep_k_points=top_k, distance_thresh=3
            )
            if (rep != -1) and (loc_err != -1):
                repeatability.append(rep)
                localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3, mean_dist = compute_homography(
                data, keep_k_points=top_k, debug=debug
            )

            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)
            auc_h_ransac.update(mean_dist)
            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)

            # Compute segmentation scores
            # TODO: calculate overlap between both images
            # write_to_file([i, rep, loc_err, c1, c2, c3, mscore])
            MScore.append(mscore)

    return (
        np.mean(repeatability),
        np.mean(localization_err),
        np.mean(correctness1),
        np.mean(correctness3),
        np.mean(correctness5),
        np.mean(MScore),
        auc_h_ransac.compute(),
    )
