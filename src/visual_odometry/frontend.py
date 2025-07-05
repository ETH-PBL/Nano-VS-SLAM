import torch
import numpy as np
import cv2

from kp2dtiny.models.kp2dtiny import tiny_factory
from kp2dtiny.models.keypoint_net_vlad import KeypointNet, VGG16_DEFAULT
import matplotlib.pyplot as plt
from utils.plot import map_colors, get_colormap


class KP2DtinyFrontend(object):
    """Wrapper around pytorch net to help with pre and post image processing."""

    def __init__(
        self,
        new_size,
        weights_path,
        nn_thresh=0.7,
        device="cuda",
        semantic_filter=False,
        classes_to_filter=[21],
        debug=True,
        method="kp2dtiny",
        config="A",
        top_k=4000,
        v3=False,
        nClasses=28,
    ):
        self.name = method
        self.device = device
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.border_remove = 4  # Remove points this close to the border.
        self.classes_to_filter = list(classes_to_filter)
        self.apply_semantic_filer = semantic_filter
        self.weights_path = weights_path
        self.plot = debug
        self.top_k = top_k
        self.new_size = new_size

        # Load the network in inference mode.
        if self.name == "kp2dtiny":
            self.net = tiny_factory(
                config, nClasses, to_export=False, to_mcu=False, v3=v3
            )
            # self.net = KeypointNetRaw(**KP2D_TINY,  v2_seg=True, use_attention=True,nClasses=28, device=device)
            # weights_path = "./checkpoints/kp2dtiny_attention_28.ckpt"
            self.net.sample_segmentation = semantic_filter
        elif self.name == "keypointnet":
            self.net = KeypointNet(nClasses=nClasses)
        if self.weights_path is not None:
            self.net.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu"))["state_dict"]
            )
            print("Loaded weights from {}".format(weights_path))

        self.net.eval()
        self.net.device = self.device
        self.net.training = False
        self.net = self.net.to(self.device)
        self.net.device = self.device
        self.color_map = get_colormap(nClasses)

    def get_info(self):
        return {
            "nn_thresh": self.nn_thresh,
            "border_remove": self.border_remove,
            "weights_path": self.weights_path,
            "device": self.device,
            "apply_semantic_filer": self.apply_semantic_filer,
            "classes_to_filter": self.classes_to_filter,
            "plot": self.plot,
            "top_k": self.top_k,
            "new_size": self.new_size,
            "name": self.name,
            "model": self.net.gather_info(),
        }

    def run(self, img):
        img = img.unsqueeze(0).sub(0.5).mul(2.0)
        _, _, H, W = img.shape
        img = img.to(self.device)
        # Forward pass of network.
        with torch.no_grad():
            out = self.net.forward(img)
            out = self.net.post_processing(out, H, W)

            score, coord, feat, vlad, seg = (
                out["score"],
                out["coord"],
                out["feat"],
                out["vlad"],
                out["seg"],
            )
        score = torch.cat([coord, score], dim=1).view(3, -1).t().cpu().numpy()
        if self.plot:
            # Add the numbers you want to check here
            debug = cv2.resize(self.color_map(seg[0, 0].cpu().numpy()), (W, H))
            debug_s = np.isin(seg[0, 0].cpu().numpy(), self.classes_to_filter).astype(
                "float32"
            )
            debug_s = cv2.resize(debug_s, (W, H))

            cv2.imshow("seg", debug)
            cv2.imshow("seg_s", debug_s)
            cv2.waitKey(1)
        feat = feat.view(self.net.nfeatures, -1).t().cpu().numpy()

        mask = score[:, 2] > self.nn_thresh
        if self.apply_semantic_filer:
            seg_mask = ~np.isin(
                seg[0, 0].view(-1).cpu().numpy(), self.classes_to_filter
            )
            mask = mask & seg_mask
            seg = seg.view(-1).cpu().numpy()[mask]
        else:
            seg = seg.view(-1).cpu().numpy()

        # Filter based on confidence threshold
        feat = feat[mask, :]
        pts = score[mask, :2]
        score = score[mask, 2]
        if score.__len__() > self.top_k and self.top_k > 0:
            top_k = np.argpartition(score, -self.top_k)[-self.top_k :]
            pts = pts[top_k]
            feat = feat[top_k]
            seg = seg[top_k]

        # kps = convert_superpts_to_keypoints(pts)
        return pts.copy(), feat.copy(), seg.copy()
