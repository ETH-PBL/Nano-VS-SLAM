import torch
from utils.image import image_grid

class KP2DTinyV2_Quantized(object):
    def __init__(self, model=None, model_path=None, device='cpu', cell =4, global_desc_dim = 8192):
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.jit.load(model_path, map_location=device)
        else:
            raise ValueError("No model provided")
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.sample_segmentation = False
        self.cross_ratio = 2.0
        self.softmax = torch.nn.Softmax(dim=1)
        self.cell = cell

        self.global_desc_dim = global_desc_dim

    def __call__(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)
    def gather_info(self):
        return {"model": "KP2DTinyV2_Quantized", "device": self.device, "sample_segmentation": self.sample_segmentation, "cross_ratio": self.cross_ratio, "cell": self.cell}
    def post_processing(self, out, H, W):
        score, center_shift, feat, seg, vlad = out['score'], out['coord'], out['feat'], out['seg'], out['vlad']
        score = self.remove_border(score)
        coord = self.calculate_coord(center_shift, score, H, W)
        coord_norm = self.normalize_coord(coord, H, W)
        feat = self.sample_feat(feat, coord_norm)
        seg = self.sample_seg(seg, coord_norm)
        out['coord'] = coord
        out['feat'] = feat
        out['seg'] = seg
        out['vlad'] = vlad
        out['score'] = score
        return out

    def sample_feat(self, feat, coord_norm):

        feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)

        dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
        feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return feat

    def sample_seg(self, seg, coord_norm):
        if self.sample_segmentation:
            seg = torch.nn.functional.grid_sample(seg, coord_norm, align_corners=True, mode='nearest')
        seg = self.softmax(seg.float())
        seg = seg.argmax(1).unsqueeze(1)
        return seg

    def normalize_coord(self, coord, H, W):
        coord_norm = coord[:, :2].clone()
        coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
        coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
        coord_norm = coord_norm.permute(0, 2, 3, 1)
        return coord_norm
    def remove_border(self, score):
        B, _, Hc, Wc = score.shape
        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        return score * border_mask.to(score.device)

    def calculate_coord(self, center_shift, score, H,W):
        B, _, Hc, Wc = score.shape
        step = (self.cell-1) / 2.

        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)
        return coord
    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)

    def get_global_desc_dim(self):
        return self.global_desc_dim

