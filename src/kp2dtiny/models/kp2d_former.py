import torch
from .segformer import MiT
from ..modules.segformer import cast_tuple
from ..modules.aggregators.netvlad import NetVLAD
import torch.nn as nn
import inspect
from ..utils.image import image_grid
from functools import partial

KEYPOINTFORMER_DEFAULT_CONFIG = {
    'dims': (32, 64, 160, 256),
    'heads': (1, 2, 5, 8),
    'ff_expansion': (8, 8, 4, 4),
    'reduction_ratio': (8, 4, 2, 1),
    'num_layers': 2,
    'channels': 3,
    'decoder_dim': 256,
    'feat_dim': 256
}

KEYPOINTFORMER_TINY_CONFIG = {
    'dims': (16, 32, 64, 64),
    'heads': (1, 2, 4, 4),
    'ff_expansion': (4, 4, 2, 2),
    'reduction_ratio': (8, 4, 4, 2),
    'num_layers': 2,
    'channels': 3,
    'decoder_dim': 64,
    'feat_dim': 64,
}

class KeypointFormer(torch.nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        feat_dim = 256,
        num_classes = 4,
        device = "cuda"
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim), #added
            nn.ReLU(inplace = True), #added
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, 1, 1),
        )
        self.loc_head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, 2, 1),
        )
        self.feat_head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding = 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, feat_dim, 1),
        )
        self.vlad_head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1, stride = 2, padding = 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace = True),
            nn.Conv2d(decoder_dim, feat_dim, 1)
        )
        self.netvlad = NetVLAD(dim=feat_dim, vladv2=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        self.training = True
        self.cell = 8
        self.cross_ratio = 2.0
        self.device = device

    def freeze_backbone(self):
        for param in self.mit.parameters():
            param.requires_grad = False

    def only_encoder(self, x):
        B, _, H, W = x.shape
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        vlad = self.relu(self.vlad_head(fused))
        return vlad
    
    def forward(self, x):
        B, _, H, W = x.shape
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        seg =  self.segmentation_head(fused)

        score = self.sigmoid(self.score_head(fused))
        B, _, Hc, Wc = score.shape
        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        center_shift = self.tanh(self.loc_head(fused))
        feat = self.feat_head(fused)
        vlad = self.relu(self.vlad_head(fused))
        vlad = self.netvlad(vlad)

        step = (self.cell - 1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W - 1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H - 1)

        if self.training is False:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W-1)/2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = nn.functional.grid_sample(feat, coord_norm, align_corners=True)

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            seg = self.softmax(seg)
            seg = seg.argmax(1).unsqueeze(1)

        out = {"score": score, "coord": coord, "feat": feat, "vlad": vlad, "seg": seg}
        return out

    def post_processing(self, out, H, W):
        return out
    def gather_info(model):
        init_signature = inspect.signature(model.__init__)
        init_params = init_signature.parameters
        init_args = {name: getattr(model, name) for name in init_params.keys() if hasattr(model, name)}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'init_args': init_args,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'netvlad_dim': model.netvlad.get_desc_size()
        }

        return info
    
    def get_global_desc_dim(self):
        return self.netvlad.get_desc_size()

