import torch
import torch.nn.functional as F
import torch.nn as nn
import inspect

from ..modules.netvlad import NetVLAD
from ..utils.image import image_grid
import time
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper
KP2D_DEFAULT = {
    "use_color":True,
    "do_upsample":True,
    "with_drop":True,
    "do_cross":True,
    "nfeatures":256,
    "device": 'cpu',
    "channel_dims": [32, 64, 128, 256, 256, 512],
    "bn_momentum":0.1
}

KP2D_TEST = {
    "use_color":True,
    "do_upsample":True,
    "with_drop":True,
    "do_cross":True,
    "nfeatures":128,
    "device": 'cpu',
    "channel_dims": [1, 2, 4, 8, 16, 32],
    "bn_momentum":0.1
}

VGG16_DEFAULT = {
    "use_color": True,
    "do_upsample": True,
    "with_drop": True,
    "do_cross": True,
    "nfeatures": 128,
    "device": 'cuda',
    "channel_dims": [64, 128, 256, 512, 512, 1024],
    "bn_momentum": 0.1,
    "num_clusters": 64,
}

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class KeypointNet(torch.nn.Module):
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
                 bn_momentum=0.1, nClasses=8, num_clusters=64,large_netvlad=False,  **kwargs):
        super().__init__()
        self.large_netvlad = large_netvlad
        self.training = True
        self.device = device
        self.use_color = use_color
        self.with_drop = with_drop
        self.do_cross = do_cross
        self.do_upsample = do_upsample
        self.nfeatures = nfeatures
        self.channel_dims = channel_dims
        # nClasses refers to the number of classes in the segmentation task.
        # The classification layer uses nClasses + 1 classes, where the first class is the background.
        self.nClasses = nClasses
        if self.use_color:
            c0 = 3
        else:
            c0 = 1

        self.bn_momentum = bn_momentum
        self.cross_ratio = 2.0

        if self.do_cross is False:
            self.cross_ratio = 1.0

        c1, c2, c3, c4, c5, d1 = channel_dims
        self.encoder_dim = c4

        self.conv1a = torch.nn.Sequential(torch.nn.Conv2d(c0, c1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c1,momentum=self.bn_momentum))
        self.conv1b = torch.nn.Sequential(torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c1,momentum=self.bn_momentum))
        self.conv2a = torch.nn.Sequential(torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv2b = torch.nn.Sequential(torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c2,momentum=self.bn_momentum))
        self.conv3a = torch.nn.Sequential(torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv3b = torch.nn.Sequential(torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c3,momentum=self.bn_momentum))
        self.conv4a = torch.nn.Sequential(torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.conv4b = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))

        # Score Head.
        self.convDa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convDb = torch.nn.Conv2d(c5, 1, kernel_size=3, stride=1, padding=1)

        # Location Head.
        self.convPa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convPb = torch.nn.Conv2d(c5, 2, kernel_size=3, stride=1, padding=1)

        # Desc Head.
        self.convFa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convFb = torch.nn.Sequential(torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(d1,momentum=self.bn_momentum))
        self.convFaa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c5,momentum=self.bn_momentum))
        self.convFbb = torch.nn.Conv2d(c5, nfeatures, kernel_size=3, stride=1, padding=1)

        # Segmentation Head
        self.convSa = torch.nn.Sequential(torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convSb = torch.nn.Sequential(torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(d1,momentum=self.bn_momentum))
        self.convSaa = torch.nn.Sequential(torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c5,momentum=self.bn_momentum))
        self.convSbb = torch.nn.Conv2d(c5, nClasses, kernel_size=3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax2d()



        # Netvlad
        if self.large_netvlad:
            self.convlad1 = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
            self.convlad2 = torch.nn.Sequential(torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c4,momentum=self.bn_momentum))
        self.convlad3 = torch.nn.Sequential(torch.nn.Conv2d(c4, self.encoder_dim, kernel_size=3, stride=2, padding=1, bias=True), torch.nn.BatchNorm2d(self.encoder_dim,momentum=self.bn_momentum))
        self.convlad4 = torch.nn.Sequential(torch.nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, stride=1, padding=1, bias=True))
        self.l2 = L2Norm()
        
        self.netvlad = NetVLAD(dim=self.encoder_dim, num_clusters=num_clusters, vladv2=False)
        self.global_desc_dim = self.netvlad.get_desc_size()
        
        self.relu = torch.nn.LeakyReLU(inplace=True)
        if self.with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cell = 8
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)
        self.upsample_seg = torch.nn.PixelShuffle(upscale_factor=2)
        self.sigmoid = torch.nn.Sigmoid()
    
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
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.dropout:
            skip = self.dropout(skip)
        x = self.pool(skip)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.dropout:
            x = self.dropout(x)
        if self.large_netvlad:
            vlad = self.relu(self.convlad1(x))
            vlad = self.relu(self.convlad2(vlad))
        else:
            vlad = x
        vlad = self.relu(self.convlad3(vlad))
        vlad = self.l2(self.convlad4(vlad))
        return vlad

        
    @timing_decorator
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
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.dropout:
            skip = self.dropout(skip)
        x = self.pool(skip)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.dropout:
            x = self.dropout(x)

        B, _, Hc, Wc = x.shape

        score = self.relu(self.convDa(x))
        if self.dropout:
            score = self.dropout(score)
        score = self.convDb(score).sigmoid()

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        center_shift = self.relu(self.convPa(x))
        if self.dropout:
            center_shift = self.dropout(center_shift)
        center_shift = self.convPb(center_shift).tanh()

        step = (self.cell-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)

        feat = self.relu(self.convFa(x))
        if self.dropout:
            feat = self.dropout(feat)
        if self.do_upsample:
            feat = self.upsample(self.convFb(feat))
            feat = torch.cat([feat, skip], dim=1)
        feat = self.relu(self.convFaa(feat))
        feat = self.convFbb(feat)
        # Global Feature
        if self.large_netvlad:
            vlad = self.relu(self.convlad1(x))
            vlad = self.relu(self.convlad2(vlad))+x
        else:   
            vlad = x
        vlad = self.relu(self.convlad3(vlad))
        vlad = self.convlad4(vlad)
        vlad = self.netvlad(vlad)

        # Segmentation
        seg = self.relu(self.convSa(x))
        if self.dropout:
            seg = self.dropout(seg)
        if self.do_upsample:
            seg = self.upsample_seg(self.convSb(seg))
            seg = torch.cat([seg, skip], dim=1)
        seg = self.relu(self.convSaa(seg))
        seg = self.convSbb(seg)
        seg = self.relu(seg) # use relu instead of softmax
                             # -> softmax does not work probably because crossentropyloss applies softmax

        if self.training is False:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W-1)/2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
            seg = self.softmax(seg)
            seg = torch.nn.functional.grid_sample(seg, coord_norm, align_corners=True, mode='nearest')

            seg = seg.argmax(1).unsqueeze(1)
        out = {"score": score, "coord": coord, "feat": feat, "vlad": vlad, "seg": seg}
        return out
    def post_processing(self, out, H, W):
        return out