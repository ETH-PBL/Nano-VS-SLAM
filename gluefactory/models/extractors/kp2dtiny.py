import torch
from ..base_model import BaseModel
from ..kp2dtiny.models.kp2dtiny import KP2DTinyV2, get_config

class KP2DTiny(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
        "detection_threshold": 0.7,
        "model_config": "S",
        "weights_path": None
    }
    def _init(self, conf):
        self.net = KP2DTinyV2(**get_config(conf["model_config"]), nClasses=28)
        if conf["weights_path"] is not None:
            print("Loading weights from", conf["weights_path"])
            self.net.load_state_dict(torch.load(conf["weights_path"], map_location=torch.device('cpu'))['state_dict'])

        self.net.eval()
        self.net.training = False
        self.detection_threshold = conf["detection_threshold"]
        self.max_num_keypoints = conf["max_num_keypoints"]
        print("kp2dtiny loaded")

    def _forward(self, data):
        image = data["image"].sub(0.5).mul(2.0)

        self.net.training = False
        B, C, H, W = image.shape

        # make sure image size is multiple of self.net.cell
        H = H - (H % 8)
        W = W - (W % 8)
        image = image[:, :, :H, :W]
        with torch.no_grad():
            score, coord, feat, vlad, seg = self.net(image)
            score, coord, feat, vlad, seg = self.net.post_processing(score, coord, feat, vlad, seg, H, W)

        if self.max_num_keypoints > 0:

            scores, idx = score.view(B, -1).topk(self.max_num_keypoints, dim=1)
            pts = coord.view(B, 2, -1).permute(0, 2, 1).gather(1, idx.unsqueeze(2).expand(-1, -1, 2))
            feat = feat.view(B, self.net.nfeatures, -1).permute(0, 2, 1).gather(1, idx.unsqueeze(2).expand(-1, -1, self.net.nfeatures))
        else:
            score = torch.cat([coord, score], dim=1).view(B, 3, -1).permute(0, 2, 1)
            feat = feat.view(B, self.net.nfeatures, -1).permute(0, 2, 1)

            mask = (score[:,: , 2] > self.detection_threshold)

            feat = feat[mask, :].unsqueeze(0)
            pts = score[mask, :]
            scores = pts[:,2].unsqueeze(0)
            pts = pts[:, :2].unsqueeze(0)

        pred = {
            "keypoints": pts,
            "keypoint_scores": scores,
            "descriptors": feat,
        }
        return pred

    def post_processing(self, score, center_shift, feat, vlad, seg, H, W):
        score = self.remove_border(score)
        coord = self.calculate_coord(center_shift, score, H, W)
        if self.training is False:
            coord_norm = self.normalize_coord(coord, H, W)
            feat = self.sample_feat(feat, coord_norm)
            seg = self.sample_seg(seg, coord_norm)
        return score, coord, feat, vlad, seg

    def sample_feat(self, feat, coord_norm):
        feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)
        dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
        feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return feat

    def sample_seg(self, seg, coord_norm):
        if self.sample_segmentation:
            seg = torch.nn.functional.grid_sample(seg, coord_norm, align_corners=True, mode='nearest')
        # seg = self.softmax(seg)
        seg = seg.argmax(1).unsqueeze(1)
        return seg

    def normalize_coord(self, coord, H, W):
        coord_norm = coord[:, :2].clone()
        coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
        coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
        coord_norm = coord_norm.permute(0, 2, 3, 1)
        return coord_norm
    def loss(self, pred, data):
        raise NotImplementedError
