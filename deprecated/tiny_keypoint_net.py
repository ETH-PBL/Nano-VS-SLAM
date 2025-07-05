import torch
import torch.nn.functional as F

import time
import inspect

from ..utils.image import image_grid
from ..modules.segformer import EfficientSelfAttention, PreNorm, MixFeedForward
from ..modules.netvlad import NetVLAD

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper
KP2D_TINY = {
    "use_color":True,
    "do_upsample":True,
    "with_drop":True,
    "do_cross":True,
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "bn_momentum":0.1,
    "downsample":2
}

KP2D_TINY_2 = {
    "use_color":True,
    "do_upsample":True,
    "with_drop":True,
    "do_cross":True,
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "bn_momentum":0.1,
    "downsample":3
}

KP2D_TINY_F = {
    "use_color":True,
    "do_upsample":True,
    "with_drop":True,
    "do_cross":True,
    "nfeatures":64,
    "channel_dims": [16, 32, 64, 128, 128, 256],
    "bn_momentum":0.1,
    "downsample":3
}
class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2.0, dim=self.dim)

class KeypointNetMobile(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    ----------
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    do_cross: bool
        Predict keypoints outside cell borders.
    kwargs : dict
        Extra parameters
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.training = True
        self.keypoint_net_raw = KeypointNetRaw(**kwargs)
        self.device = self.keypoint_net_raw.device
        self.softmax = torch.nn.Softmax2d()
        self.encoder_dim = self.keypoint_net_raw.encoder_dim

    #@timing_decorator
    def forward(self, x):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, _, H, W = x.shape

        score, center_shift, feat, vlad, seg = self.keypoint_net_raw(x)

        B, _, Hc, Wc = score.shape


        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        step = (self.keypoint_net_raw.cell-1) / 2.

        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.keypoint_net_raw.cell) + step

        coord_un = center_base.add(center_shift.mul(self.keypoint_net_raw.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)



        if self.training is False:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W-1)/2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            seg = self.softmax(seg)
            seg = seg.argmax(1).unsqueeze(1)
        return score, coord, feat, vlad, seg

    def load_state_dict(self, state_dict, strict=True):
        self.keypoint_net_raw.load_state_dict(state_dict, strict=strict)

    def eval(self):
        super().eval()
        self.training = False
        self.keypoint_net_raw.training = False
        self.keypoint_net_raw.eval()
    def train(self,mode: bool = True):
        super().train(mode)
        self.training = mode
        self.keypoint_net_raw.training = mode
        self.keypoint_net_raw.train(mode)


    def state_dict(self):

        return self.keypoint_net_raw.state_dict()
    def freeze_backbone(self):
        self.keypoint_net_raw.freeze_backbone()

    def only_encoder(self, x):
        return self.keypoint_net_raw.only_encoder(x)

    def gather_info(model):
        return model.keypoint_net_raw.gather_info()

    def get_global_desc_dim(self):
        return self.keypoint_net_raw.get_global_desc_dim()

class KeypointNetRaw(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    ----------
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    do_cross: bool
        Predict keypoints outside cell borders.
    kwargs : dict
        Extra parameters
    """

    def __init__(self, use_color=True, do_upsample=True, with_drop=True, do_cross=True,
                 nfeatures=256,device = 'cpu', channel_dims=[32, 64, 128, 256, 256, 512],
                 bn_momentum=0.1, nClasses=8, num_clusters=64, downsample = 3, large_netvlad=False, v2_seg=False, use_attention=False, **kwargs):
        super().__init__()
        print("Dropout:",with_drop)
        self.device = device
        self.use_color = use_color
        self.with_drop = with_drop
        self.do_cross = do_cross
        self.do_upsample = do_upsample
        self.nfeatures = nfeatures
        self.downsample = downsample
        self.nClasses = nClasses
        self.large_netvlad = large_netvlad
        self.v2_seg = v2_seg
        self.use_attention = use_attention
        self.sample_segmentation = False
        if self.use_color:
            c0 = 3
        else:
            c0 = 1

        self.bn_momentum = bn_momentum
        self.cross_ratio = 2.0

        if self.do_cross is False:
            self.cross_ratio = 1.0

        c1, c2, c3, c4, c5, d1 = channel_dims
        self.encoder_dim = c4 * 2

        self.conv1a = torch.nn.Sequential(torch.nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c1,momentum=self.bn_momentum))
        self.conv1b = torch.nn.Sequential(torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv2a = torch.nn.Sequential(torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv2b = torch.nn.Sequential(torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv3a = torch.nn.Sequential(torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv3b = torch.nn.Sequential(torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        # if self.use_attention:
        #     self.att_enc = torch.nn.ModuleList([
        #             PreNorm(c4, EfficientSelfAttention(dim = c4, heads = 4, reduction_ratio = 2)),
        #             PreNorm(c4, MixFeedForward(dim = c4, expansion_factor = 2, activation='gelu')),
        #         ])

        self.conv4a = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.conv4b = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        # Score Head.
        self.convDa = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convDb = torch.nn.Conv2d(c4, 1, kernel_size=3, stride=1, padding=1)

        # Location Head.
        self.convPa = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convPb = torch.nn.Conv2d(c4, 2, kernel_size=3, stride=1, padding=1)

        # Desc Head.
        self.convFa = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        #self.convFb = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convFb = torch.nn.Sequential(torch.nn.Conv2d(c4, d1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(d1,momentum=self.bn_momentum))
        self.convFaa = torch.nn.Sequential(torch.nn.Conv2d(c3+c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c5,momentum=self.bn_momentum))
        self.convFbb = torch.nn.Conv2d(c5, nfeatures, kernel_size=3, stride=1, padding=1)

        # Segmentation Head
        self.convSa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
        self.convSb = torch.nn.Sequential(torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1, bias=False),
                                          torch.nn.BatchNorm2d(d1, momentum=self.bn_momentum))
        self.convSaa = torch.nn.Sequential(torch.nn.Conv2d(c3+c4, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))
        self.convSbb = torch.nn.Conv2d(c5, nClasses, kernel_size=3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax2d()
        if self.v2_seg:
            if self.use_attention:
                self.attention = torch.nn.ModuleList([
                    PreNorm(c5, EfficientSelfAttention(dim = c5, heads = 4, reduction_ratio = 2)),
                    PreNorm(c5, MixFeedForward(dim = c5, expansion_factor = 2, activation='gelu')),
                    PreNorm(c5, EfficientSelfAttention(dim = c5, heads = 4, reduction_ratio = 2)),
                    PreNorm(c5, MixFeedForward(dim = c5, expansion_factor = 2, activation='gelu')),
                ])
                self.convSv2b = torch.nn.Sequential(torch.nn.Conv2d(c5, c4*4, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c4*4, momentum=self.bn_momentum))
                self.convSv2c = torch.nn.Sequential(torch.nn.Conv2d(c4*2, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))
            else:
                self.convs = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum)),
                    torch.nn.Sequential(torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))
                ])

                self.convSv2a = torch.nn.Sequential(torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))
                self.convSv2b = torch.nn.Sequential(torch.nn.Conv2d(c5, c4*4, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c4*4, momentum=self.bn_momentum))
                self.convSv2c = torch.nn.Sequential(torch.nn.Conv2d(c4*2, c5, kernel_size=3, stride=1, padding=1, bias=False),
                                                    torch.nn.BatchNorm2d(c5, momentum=self.bn_momentum))


        # Netvlad
        if self.large_netvlad:
            self.convlad1 = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False),
                                                torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
            self.convlad2 = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False),
                                                torch.nn.BatchNorm2d(c4, momentum=self.bn_momentum))
        self.convlad3 = torch.nn.Sequential(
            torch.nn.Conv2d(c4, self.encoder_dim, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(self.encoder_dim, momentum=self.bn_momentum))
        self.convlad4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=1, padding=1, bias=True))
        self.l2 = L2Norm()

        self.netvlad = NetVLAD(dim=self.encoder_dim, num_clusters=num_clusters, vladv2=False)
        self.global_desc_dim = self.netvlad.get_desc_size()

        self.upsample_seg = torch.nn.PixelShuffle(upscale_factor=2)

        self.relu = torch.nn.LeakyReLU(inplace=True)
        if self.with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cell = pow(2,self.downsample)
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)
        self.training = True
        #self.upsample = torch.nn.ConvTranspose2d(c4, c3, 2, stride=2, padding=0,output_padding=0)
        self.global_desc_dim = self.netvlad.get_desc_size()
        self.softmax = torch.nn.Softmax2d()
        self.quant = torch.ao.quantization.QuantStub
        self.dequant = torch.ao.quantization.DeQuantStub
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
            'netvlad_dim': model.global_desc_dim
        }

        return info

    def get_global_desc_dim(self):
        return self.global_desc_dim
    def freeze_backbone(self):
        for layer in [
            self.conv1a, self.conv1b, self.conv2a, self.conv2b, self.conv3a,
            self.conv3b, self.conv4a, self.conv4b]:
            for param in layer.parameters():
                param.requires_grad = False

    def only_encoder(self, x):
        # This function is used for the get_cluster() function in train_NetVLAD.py
        B, _, H, W = x.shape

        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 2:
            x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 3:
            x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.with_drop:
            skip = self.dropout(skip)
        if self.downsample >= 1:
            x = self.pool(skip)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.with_drop:
            x = self.dropout(x)

        if self.large_netvlad:
            vlad = self.relu(self.convlad1(x))
            vlad = self.relu(self.convlad2(vlad))
        else:
            vlad = x
        vlad = self.relu(self.convlad3(vlad))
        if self.dropout:
            vlad = self.dropout(vlad)
        vlad = self.l2(self.convlad4(vlad))
        return vlad
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
    #@timing_decorator
    def forward(self, x):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, _, H, W = x.shape

        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 2:
            x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 3:
            x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.with_drop:
            skip = self.dropout(skip)
        if self.downsample >= 1:
            x = self.pool(skip)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.with_drop:
            x = self.dropout(x)

        score = self.relu(self.convDa(x))
        if self.with_drop:
            score = self.dropout(score)
        score = self.convDb(score).sigmoid()



        center_shift = self.relu(self.convPa(x))
        if self.with_drop:
            center_shift = self.dropout(center_shift)
        center_shift = self.convPb(center_shift).tanh()

        feat = self.relu(self.convFa(x))
        if self.with_drop:
            feat = self.dropout(feat)
        if self.do_upsample:
            feat = self.upsample(self.convFb(feat))
            feat = torch.cat([feat, skip], dim=1)
        feat = self.relu(self.convFaa(feat))
        feat = self.convFbb(feat)

        # Global Feature
        if self.large_netvlad:
            vlad = self.relu(self.convlad1(x))
            vlad = self.relu(self.convlad2(vlad)) + x
        else:
            vlad = x
        vlad = self.relu(self.convlad3(vlad))

        vlad = self.convlad4(vlad)
        vlad = self.netvlad(vlad)

        # Segmentation
        seg = self.relu(self.convSa(x))
        if self.v2_seg:
            if self.use_attention:
                seg = self.attention[0](seg)
                seg = self.attention[1](seg)
                seg = self.pool(seg)
                seg = self.attention[2](seg)
                seg = self.attention[3](seg)
                seg = self.relu(self.convSv2b(seg))
                seg = self.upsample_seg(seg)
                seg = torch.cat([seg, x], dim=1)
                seg = self.relu(self.convSv2c(seg))
            else:
                seg = self.relu(self.convs[0](seg))
                seg = self.pool(seg)
                seg = self.relu(self.convs[1](seg))
                seg = self.relu(self.convSv2a(seg))
                seg = self.relu(self.convSv2b(seg))
                seg = self.upsample_seg(seg)
                seg = torch.cat([seg, x], dim=1)
                seg = self.relu(self.convSv2c(seg))
        if self.dropout:
            seg = self.dropout(seg)
        if self.do_upsample:
            seg = self.upsample_seg(self.convSb(seg))
            seg = torch.cat([seg, skip], dim=1)
        seg = self.relu(self.convSaa(seg))
        seg = self.convSbb(seg)
        seg = self.relu(seg)  # use relu instead of softmax

        return score, center_shift, feat, vlad, seg

    def post_processing(self,score, center_shift, feat, vlad, seg, H,W):
        score = self.remove_border(score)
        coord = self.calculate_coord(center_shift, score, H,W)
        if self.training is False:
            coord_norm = self.normalize_coord(coord, H,W)
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
        seg = self.softmax(seg)
        seg = seg.argmax(1).unsqueeze(1)
        return seg

    def normalize_coord(self, coord, H,W):
        coord_norm = coord[:, :2].clone()
        coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
        coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
        coord_norm = coord_norm.permute(0, 2, 3, 1)
        return coord_norm

