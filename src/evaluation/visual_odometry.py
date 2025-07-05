import cv2
import numpy as np
import kornia
import torch

from ..visual_odometry.camera import PinholeCamera
from ..visual_odometry.groundtruth import KittiVideoGroundTruth
from ..visual_odometry.feature_matcher import BfFeatureMatcher
from ..visual_odometry.utils import calculate_pose_error, calculate_error_stats

from utils.plot import map_colors, get_colormap


def samsung_params(size):
    fx = 2317.6450198781713
    fy = 2317.6450198781713
    cx = size[1] / 2.0
    cy = size[0] / 2.0
    return fx, fy, cx, cy


def plot_legend(legend):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.axis("off")  # Hide the axes for a clean legend display

    # Generate a list of patches
    patches = [
        mpatches.Patch(
            color=(color[0] / 255, color[1] / 255, color[2] / 255),
            label=f"{name} ({train_id})",
        )
        for train_id, (color, name) in legend.items()
    ]

    # Add the patches to the axes as a legend
    legend = ax.legend(handles=patches, loc="center", frameon=False)

    # Display the legend
    plt.show()


def cityscapes_color_map():
    from torchvision.datasets import Cityscapes

    classes = Cityscapes.classes

    mapping = {label.train_id: label.color for label in classes}
    # matplotlib colormap:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    new_class_colors = torch.full((256, 3), 0, dtype=torch.uint8)
    for original_class, new_class_index in mapping.items():
        new_class_colors[int(original_class)] = torch.tensor(new_class_index)

    legend = {
        label.train_id: (label.color, label.name)
        for label in classes
        if label.train_id not in [-1, 255]
    }
    return new_class_colors


def kitty_params():
    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy = 185.2157
    return fx, fy, cx, cy


def inference(
    net, image, new_size, plot=False, nn_thresh=0.7, top_k=4000, device="cuda"
):
    img = kornia.image_to_tensor(image).float() / 255.0
    image_shape = image.shape
    if new_size is not None:
        img = kornia.geometry.transform.resize(img, size=new_size).to(device)

        scale_x = new_size[1] / float(image_shape[1])
        scale_y = new_size[0] / float(image_shape[0])

    img = img.unsqueeze(0).sub(0.5).mul(2.0)
    _, _, H, W = img.shape
    img = img.to(device)
    # Forward pass of network.
    with torch.no_grad():
        out = net.forward(img)
        out = net.post_processing(out, H, W)
    score, coord, feat, seg = out["score"], out["coord"], out["feat"], out["seg"]
    score = torch.cat([coord, score], dim=1).view(3, -1).t().cpu().numpy()
    # if plot:
    #     # Add the numbers you want to check here
    #     debug = cv2.resize(map_colors(seg[0,0].cpu().numpy()), (W, H))
    #     debug_s = np.isin(seg[0,0].cpu().numpy(), self.classes_to_filter).astype("float32")
    #     debug_s = cv2.resize(debug_s, (W, H))

    #     cv2.imshow('seg', debug)
    #     cv2.imshow('seg_s', debug_s)
    #     cv2.waitKey(1)
    feat = feat.view(net.nfeatures, -1).t().cpu().numpy()

    mask = score[:, 2] > nn_thresh

    seg = seg.view(-1).cpu().numpy()

    # Filter based on confidence threshold
    feat = feat[mask, :]
    pts = score[mask, :2]
    score = score[mask, 2]
    if score.__len__() > top_k and top_k > 0:
        top_k = np.argpartition(score, -top_k)[-top_k:]
        pts = pts[top_k]
        feat = feat[top_k]
        seg = seg[top_k]

    if new_size is not None:
        pts[:, 0] = pts[:, 0] / scale_x
        pts[:, 1] = pts[:, 1] / scale_y
    return pts.copy(), feat.copy(), out


def match(matcher, kps_cur, feat_cur, kps_prev, feat_prev, top_k_matches=1000):
    idxs0, idxs1, score = matcher.match(feat_prev, feat_cur)
    score = np.asarray(score)
    kps0 = np.asarray(kps_prev[idxs0, :])
    kps1 = np.asarray(kps_cur[idxs1, :])

    if score.__len__() > top_k_matches and top_k_matches > 0:
        top_k_idxs = np.argpartition(score, top_k_matches)[:top_k_matches]

        kps0 = kps0[top_k_idxs]
        kps1 = kps1[top_k_idxs]
    return kps0, kps1


def estimatePose(kps_ref, kps_cur, cam):
    kp_ref_u = cam.undistort_points(kps_ref)
    kp_cur_u = cam.undistort_points(kps_cur)
    kpn_ref = cam.unproject_points(kp_ref_u)
    kpn_cur = cam.unproject_points(kp_cur_u)
    ransac_method = None
    try:
        ransac_method = cv2.USAC_MSAC
    except:
        ransac_method = cv2.RANSAC
    # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
    # E = kornia.geometry.epipolar.find_essential(kpn_cur, kpn_ref)
    E, mask_match = cv2.findEssentialMat(
        kpn_cur,
        kpn_ref,
        focal=1,
        pp=(0.0, 0.0),
        method=ransac_method,
        prob=0.999,
        threshold=0.0003,
    )
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0.0, 0.0))

    return R, t, mask_match, mask


def calculate_relative_error(gt, i_frame, R, t):
    # applying the relative transformation to the last ground truth position
    _, _, _, absolute_scale = gt.getPoseAndAbsoluteScale(i_frame - 1)
    t_last, rot_last = gt.extract_pose_values(i_frame - 1)

    est_t = t_last + absolute_scale * rot_last.dot(t).T
    est_R = rot_last.dot(R)

    t_curr, R_curr = gt.extract_pose_values(i_frame)
    t_error, r_error = calculate_pose_error(R_curr, t_curr, est_R, est_t[0])

    return t_error, r_error


def drawFeatureTracks(img, kps0, kps1, mask_match, line_width=2, point_size=3):
    if img.ndim == 2:
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        draw_img = img.copy()
    num_outliers = 0
    num_inliers = 0
    for i, pts in enumerate(zip(kps1, kps0)):
        if mask_match[i] == 0:
            num_outliers += 1
            continue
        p1, p2 = pts
        a, b = p1.astype(int).ravel()
        c, d = p2.astype(int).ravel()
        cv2.line(draw_img, (a, b), (c, d), (0, 255, 0), line_width)
        cv2.circle(draw_img, (a, b), point_size, (0, 0, 255), -1)
        num_inliers += 1

    return draw_img


def evaluate_visual_odometry(
    model,
    kitti_path,
    gt_path,
    video_name,
    device,
    new_size=None,
    plot=False,
    verbose=False,
):
    # load images from video
    video_path = kitti_path + "/" + video_name
    gt = KittiVideoGroundTruth(kitti_path, gt_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    size = frame.shape
    cam_params = kitty_params()

    model.eval()
    model.training = False
    fx, fy, cx, cy = cam_params

    cam = PinholeCamera(size[1], size[0], fx, fy, cx, cy, [0, 0, 0, 0, 0])
    matcher = BfFeatureMatcher(norm_type=cv2.NORM_L2)

    kps0, feat0, out = inference(model, frame, new_size, plot, device=device)
    if model.nClasses == 19:
        color_map = cityscapes_color_map()
        legend = cv2.imread("./vo/legend.png")
        legend_size = legend.shape
        legend_ratio = legend_size[1] / legend_size[0]
        resize = size[0] * 2
        if "depth" in out:
            resize = int(size[0] // 2 + size[0])
        legend = cv2.resize(
            legend,
            (int(resize * legend_ratio), resize),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        color_map = get_colormap(model.nClasses)
    color_map_depth = get_colormap(256)
    # run through all frames

    i_frame = 1
    t_errs, r_errs = [], []
    estimation_fails = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        kps1, feat1, out = inference(model, frame, new_size, plot, device=device)

        try:
            m_kps0, m_kps1 = match(matcher, kps1, feat1, kps0, feat0)
            R, t, mask_match, _ = estimatePose(m_kps0, m_kps1, cam)
        except:
            R = np.eye(3)
            t = np.zeros((3, 1))
            m_kps0 = kps0.copy()
            m_kps1 = kps1.copy()
            mask_match = np.zeros((m_kps0.shape[0], 1))
            estimation_fails += 1

        t_err, r_err = calculate_relative_error(gt, i_frame, R, t)

        if plot:
            # frame = cv2.resize(frame, (new_size[1] , new_size[0]))
            # frame = cv2.resize(frame, (size[1] , size[0]))

            draw_img = drawFeatureTracks(frame, m_kps0, m_kps1, mask_match)
            if model.nClasses == 19:
                segmentation_debug = cv2.cvtColor(
                    color_map[out["seg"][0][0].cpu()].numpy(), cv2.COLOR_BGR2RGB
                )
            else:
                segmentation_debug = (
                    color_map(out["seg"][0][0].cpu().numpy()) * 255
                ).astype(np.uint8)[..., :3]

            if "depth" in out:
                segmentation_debug = cv2.resize(
                    segmentation_debug,
                    (size[1] // 2, size[0] // 2),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_debug = out["depth"][0][0].cpu().numpy()
                depth_debug = (
                    color_map_depth(
                        cv2.resize(depth_debug, (size[1] // 2, size[0] // 2))
                    )[..., :3]
                    * 255
                ).astype(np.uint8)
                concatenated = np.hstack((segmentation_debug, depth_debug))
            else:
                concatenated = cv2.resize(
                    segmentation_debug,
                    (size[1], size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            debugimage = np.vstack((draw_img, concatenated))
            if model.nClasses == 19:
                debugimage = np.hstack((debugimage, legend))
            # cv2.imshow("segmentation", debugimage)
            # cv2.imshow("depth", depth_debug)

            cv2.imshow("frame", debugimage)
            cv2.waitKey(1)

        kps0 = kps1
        feat0 = feat1
        i_frame += 1
        t_errs.append(t_err)
        r_errs.append(r_err)
    t_errs = np.array(t_errs[1:])
    r_errs = np.array(r_errs[1:])
    total_errs = t_errs + r_errs
    t_stats = calculate_error_stats(t_errs)
    r_stats = calculate_error_stats(r_errs)
    total_stats = calculate_error_stats(total_errs)
    if verbose:
        return {
            "translation": t_stats,
            "rotation": r_stats,
            "total": total_stats,
            "estimation_fails": estimation_fails,
        }
    else:
        return total_stats
    # return error


def demo(
    model,
    video_path,
    cam_params,
    device,
    new_size=None,
    plot=False,
    verbose=False,
    track=False,
    out_path=None,
):
    # load images from video

    cap = cv2.VideoCapture(video_path)

    # set frame rate

    # resize by 50 percent

    ret, frame = cap.read()
    size = frame.shape

    model.eval()
    model.training = False
    fx, fy, cx, cy = cam_params

    cam = PinholeCamera(size[1], size[0], fx, fy, cx, cy, [0, 0, 0, 0, 0])
    matcher = BfFeatureMatcher(norm_type=cv2.NORM_L2)

    kps0, feat0, out = inference(model, frame, new_size, plot, device=device)
    if model.nClasses == 19:
        color_map = cityscapes_color_map()
        legend = cv2.imread("./vo/legend.png")
        legend_size = legend.shape
        legend_ratio = legend_size[1] / legend_size[0]
        resize = size[0] * 2
        if "depth" in out:
            resize = int(size[0] // 2 + size[0])
        legend = cv2.resize(
            legend,
            (int(resize * legend_ratio), resize),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        color_map = get_colormap(model.nClasses)
    color_map_depth = get_colormap(256)
    # run through all frames

    i_frame = 1
    t_errs, r_errs = [], []
    estimation_fails = 0
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    if track:
        video_out_key = cv2.VideoWriter(
            out_path + f"/output_key_track_{str(new_size[0])}_{str(new_size[1])}.mp4",
            fourcc,
            30.0,
            (size[1], size[0]),
        )
    else:
        video_out_key = cv2.VideoWriter(
            out_path + f"/output_key_{str(new_size[0])}_{str(new_size[1])}.mp4",
            fourcc,
            30.0,
            (size[1], size[0]),
        )
        video_out_seg = cv2.VideoWriter(
            out_path + f"/output_seg_{str(new_size[0])}_{str(new_size[1])}.mp4",
            fourcc,
            30.0,
            (size[1], size[0]),
        )
    while True:
        ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()

        if not ret or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps1, feat1, out = inference(model, frame, new_size, plot, device=device)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        R = np.eye(3)
        t = np.zeros((3, 1))
        m_kps0 = kps0.copy()
        m_kps1 = kps1.copy()
        mask_match = np.ones((m_kps1.shape[0], 1))

        if track:
            try:
                m_kps0, m_kps1 = match(matcher, kps1, feat1, kps0, feat0)
                R, t, mask_match, _ = estimatePose(m_kps0, m_kps1, cam)
            except:
                estimation_fails += 1
                print("Estimation failed")

        if plot:
            # frame = cv2.resize(frame, (new_size[1] , new_size[0]))
            # frame = cv2.resize(frame, (size[1] , size[0]))
            if track:
                draw_img = drawFeatureTracks(frame, m_kps0, m_kps1, mask_match)
            else:
                draw_img = drawFeatureTracks(frame, m_kps1, m_kps1, mask_match)
            if model.nClasses == 19:
                segmentation_debug = cv2.cvtColor(
                    color_map[out["seg"][0][0].cpu()].numpy(), cv2.COLOR_BGR2RGB
                )
            else:
                segmentation_debug = (
                    color_map(out["seg"][0][0].cpu().numpy()) * 255
                ).astype(np.uint8)[..., :3]

            if "depth" in out:
                segmentation_debug = cv2.resize(
                    segmentation_debug,
                    (size[1] // 2, size[0] // 2),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_debug = out["depth"][0][0].cpu().numpy()
                depth_debug = (
                    color_map_depth(
                        cv2.resize(depth_debug, (size[1] // 2, size[0] // 2))
                    )[..., :3]
                    * 255
                ).astype(np.uint8)
                concatenated = np.hstack((segmentation_debug, depth_debug))
            else:
                concatenated = cv2.resize(
                    segmentation_debug,
                    (size[1], size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            debug_frame = cv2.resize(frame, (size[1], size[0]))
            debugimage = np.vstack((draw_img, concatenated))
            # if model.nClasses == 19:
            #     debugimage = np.hstack((debugimage, legend))
            # cv2.imshow("segmentation", debugimage)
            # cv2.imshow("depth", depth_debug)
            debugimage = cv2.resize(debugimage, (size[1], size[0]))
            cv2.imshow("frame", debugimage)
            cv2.waitKey(1)
            if out_path is not None:
                # cv2.imwrite(out_path + f"/keypoints_{i_frame}.png", draw_img)
                # cv2.imwrite(out_path + f"/segmentation_{i_frame}.png", segmentation_debug)
                # if 'depth' in out:
                #     cv2.imwrite(out_path + f"/depth_{i_frame}.png", depth_debug)
                if not track:
                    video_out_seg.write(
                        cv2.resize(
                            segmentation_debug,
                            (size[1], size[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
                video_out_key.write(draw_img)

        kps0 = kps1
        feat0 = feat1
        i_frame += 1
    cap.release()
    if not track:
        video_out_seg.release()
    video_out_key.release()
    cv2.destroyAllWindows()
