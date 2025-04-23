
import torch
import inspect
import time
from torch import  quantization

from ..utils.image import image_grid
from ..modules.base import AnnotatedConvBnReLUModel
from ..modules.encoders import BackBone
from ..modules.decoders.heads import SimpleTaskHead, UpscaleHead
from ..modules.decoders.vpr import VPRHead
from ..modules.decoders.segmentation import SegmentationHead, SegmentationHeadATT,SegmentationFeatHeadLightATT, SegmentationFeatHeadLight


def fuse_modules(model):
    """
    Fuse conv+bn+relu modules in a model. This is used for quantization.
    :param model: KP2DTiny model
    :return: None
    """
    # go through all modules and fuse if off type AnnotatedConvBnReLUModel
    for name, module in model.named_modules():
        if isinstance(module, AnnotatedConvBnReLUModel):
            torch.quantization.fuse_modules(module, ['conv', 'bn', 'relu'], inplace=True)
            
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

# MODEL CONFIGS
TINY_S = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "downsample":2,
    "use_attention":False,
    "leaky_relu":True,
    "encoder_dim": 64,
}

TINY_S_A = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 64,
}

GEM_S_A = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 64,
    "global_descriptor_method": "gem",
}

CONVAP_S_A = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 64,
    "global_descriptor_method": "convap",
}

TINY_N_A = {
    "nfeatures":32,
    "channel_dims": [16,24,24,48,48, 96],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "num_clusters": 32,

    "encoder_dim": 48,
}

TINY_N = {
    "nfeatures":32,
    "channel_dims": [16,24,24,48,48, 96],
    "downsample":2,
    "use_attention":False,
    "leaky_relu":True,
    "num_clusters": 32,
    "encoder_dim": 48,
}
GEM_N = {
    "nfeatures":32,
    "channel_dims": [16,24,24,48,48, 96],
    "downsample":2,
    "use_attention":False,
    "leaky_relu":True,
    "num_clusters": 32,
    "encoder_dim": 48,
    "global_descriptor_method": "gem",}
TINY_F = {
    "nfeatures":64,
    "channel_dims": [16, 32, 64, 128, 128, 256],
    "downsample":3,
    "use_attention":False,
    "leaky_relu":True,
}

V3_S = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "bn_momentum":0.1,
    "downsample":2,
    "use_attention":False,
    "leaky_relu":True,
    "encoder_dim": 64,
}
V3_S_A = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "bn_momentum":0.1,
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 64,
}
V3_S_A_CONVAP = {
    "nfeatures":32,
    "channel_dims": [16,32,32,64,64, 128],
    "bn_momentum":0.1,
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 64,
    "global_descriptor_method": "convap",
}


V3_N = {
    "nfeatures":32,
    "channel_dims": [16,24,24,48,48, 96],
    "bn_momentum":0.1,
    "downsample":2,
    "use_attention":False,
    "encoder_dim": 48,
}
V3_N_A = {
    "nfeatures":32,
    "channel_dims": [16,24,24,48,48, 96],
    "bn_momentum":0.1,
    "downsample":2,
    "use_attention":True,
    "encoder_dim": 48,
}

LARGE_D = {
    "nfeatures":128,
    "channel_dims": [64,128,128,256,256, 512],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 128,
    "global_descriptor_method": "convap",

}
LARGE_D_A_V3 = {
    "nfeatures":128,
    "channel_dims": [64,128,128,256,256, 512],
    "downsample":2,
    "use_attention":True,
    "leaky_relu":True,
    "encoder_dim": 128,
    "global_descriptor_method": "convap"
}

LARGE_D_V3 = {
    "nfeatures":128,
    "channel_dims": [64,128,128,256,256, 512],
    "downsample":2,
    "use_attention":False,
    "leaky_relu":True,
    "encoder_dim": 128,
    "global_descriptor_method": "convap"
}


KP2DTINY_CONFIGS = {"S": TINY_S,
                    "S_A": TINY_S_A, # Formerly known as just A
                    "N": TINY_N,
                    "N_A": TINY_N_A,
                    "D": LARGE_D,
                    "F": TINY_F,
                    "GEM_N": GEM_N,
                    "GEM_S_A": GEM_S_A,
                    "CONVAP_S_A": CONVAP_S_A
                    }

KP2DTINYV3_CONFIGS = {
                    "S": V3_S, # aka A_1
                    "S_A": V3_S_A, # aka A_2
                    "N": V3_N, # aka N_2
                    "N_A": V3_N_A, # aka N
                    "D": LARGE_D_V3,
                    "D_A": LARGE_D_A_V3,
                    "CONVAP_S_A": V3_S_A_CONVAP,
                    }
def tiny_factory(config, n_classes, to_mcu=False, to_export=False, v3=False):
    """
    Create a keypoint detection model.

    Args:
        config (str): The name of the configuration.
        v3 (bool, optional): Whether to use the V3 configuration. Defaults to False.
        n_classes (int): The number of classes.
        to_mcu (bool, optional): Whether to configure for MCU. Defaults to False.
        to_export (bool, optional): Whether to configure for export, meaning we disable the aggregation layer for VPR. Defaults to False.

    Returns:
        torch.nn.Module: The keypoint detection model.
    """


    conf = get_config(config, to_mcu=to_mcu, to_export=to_export, v3=v3)
    if v3:
        model = KP2DTinyV3(**conf, nClasses=n_classes)
    else:
        model = KP2DTinyV2(**conf, nClasses=n_classes)

    return model
def get_config(config, to_mcu=False, to_export=False, v3=False):
    """
    Get the configuration dictionary for the specified config.

    Args:
        config (str): The name of the configuration.
        to_mcu (bool, optional): Whether to configure for MCU. Defaults to False.
        to_export (bool, optional): Whether to configure for export, where we do not have the aggregation layer for VPR. Defaults to False.
        v3 (bool, optional): Whether to use the V3 configuration. Defaults to False.

    Returns:
        dict: The configuration dictionary.

    Raises:
        ValueError: If the specified config is not supported.
    """
    config_dict = KP2DTINYV3_CONFIGS if v3 else KP2DTINY_CONFIGS
    
    if config not in config_dict:
        raise ValueError("Config {} not supported, choose from ".format(config), list(config_dict.keys()))
    
    conf = config_dict[config]
    
    if to_mcu:
        conf["upscale_method"] = "convtranspose"
        conf["leaky_relu"] = False
        print("MCU config: upscale_method=convtranspose, leaky_relu=False")
        
    if to_export:
        conf["remove_netvlad"] = True
        print("Export Config: remove_netvlad=True")
        
    print("Configuration:", conf)
    return conf

class KP2DTinyV2(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    
    """
    nfeatures: int
    device: str
    channel_dims: list
    bn_momentum: float
    nClasses: int
    num_clusters: int
    downsample: int
    use_attention: bool

    def __init__(self,
                 nfeatures=256,device = 'cpu', channel_dims=[32, 64, 128, 256, 256, 512],
                 bn_momentum=0.1, nClasses=8, num_clusters=64, downsample = 3, use_attention = False, 
                 mem_efficient=False, upscale_method="pixelshuffle", remove_netvlad=False, leaky_relu=True, depth=False, encoder_dim=None, global_descriptor_method="netvlad",**kwargs):
        super().__init__()
        self.device = device
        with_drop = True
        self.with_drop = with_drop
        self.nfeatures = nfeatures
        self.downsample = downsample
        self.nClasses = nClasses
        self.sample_segmentation = False
        self.use_attention = use_attention
        self.leaky_relu = leaky_relu
        self.remove_netvlad = remove_netvlad
        self.upscale_method = upscale_method
        self.depth = depth
        self.num_clusters = num_clusters
        self.global_descriptor_method = global_descriptor_method
        # Hardcoded RGB input
        c0 = 3
  

        self.bn_momentum = bn_momentum
        self.cross_ratio = 2.0



        c1, c2, c3, c4, c5, d1 = channel_dims
        if encoder_dim is not None:
            self.encoder_dim = encoder_dim
        else:
            self.encoder_dim = c4

        self.backbone = BackBone(c0, c1, c2, c3, c4, downsample, with_drop, bn_momentum=self.bn_momentum,leaky_relu=leaky_relu)

        self.score_head = SimpleTaskHead(c4, c4, 1, bn_momentum=self.bn_momentum, with_drop=with_drop,leaky_relu=leaky_relu)
        self.loc_head = SimpleTaskHead(c4, c4, 2, bn_momentum=self.bn_momentum, with_drop=with_drop,leaky_relu=leaky_relu)
        #self.desc_head = UpscaleHead(c4, c4,c3*4,c3+c4,c4*2, nfeatures, with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
        #correction to be in line with legacy             --v
        self.desc_head = UpscaleHead(c4, c4,c3*4,c3+c4,    c4   , nfeatures, with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)

        if self.use_attention:
            self.seg_head = SegmentationHeadATT(c4, c5, c4+c3, nClasses, d1,with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
            if self.depth:
                self.depth_head = SegmentationHeadATT(c4, c5, c4+c3, 1, d1,with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
        else:
            self.seg_head = SegmentationHead(c4, c5, c4+c3, nClasses, d1,with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
            if self.depth:
                self.depth_head = SegmentationHead(c4, c5, c4+c3, 1, d1,with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
        # Netvlad
        self.vlad_head = VPRHead(c4, self.encoder_dim, num_clusters, with_drop,
                                     bn_momentum=self.bn_momentum,mem_efficient=mem_efficient,
                                     remove_netvlad=remove_netvlad,leaky_relu=leaky_relu, method=global_descriptor_method)
        if self.leaky_relu:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        self.cell = pow(2,self.downsample)
        self.training = True
        self.global_desc_dim = self.vlad_head.global_desc_dim
        self.softmax = torch.nn.Softmax2d()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        print(self.downsample)
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
            'netvlad_dim': model.global_desc_dim,
            'upscale_method': model.upscale_method,
            'leaky_relu': model.leaky_relu,
            'use_attention': model.use_attention,
            
        }

        return info

    def get_global_desc_dim(self):
        return self.global_desc_dim

    def init_netvlad(self, clsts, traindescs):
        self.vlad_head.netvlad.init_params(clsts, traindescs)

    def get_num_clusters(self):
        return self.vlad_head.netvlad.num_clusters

    def get_netvlad_dim(self):
        return self.global_desc_dim

    def freeze_backbone(self):
        for layer in self.backbone.children():
            for param in layer.parameters():
                param.requires_grad = False
    def freeze_segmentation(self, except_last_layer=False):
        self.seg_head.freeze(except_last_layer)
                
    def fuse(self):
        fuse_modules(self.backbone)
        fuse_modules(self.score_head)
        fuse_modules(self.loc_head)
        fuse_modules(self.desc_head)
        fuse_modules(self.seg_head)
        fuse_modules(self.vlad_head)
        
    def only_encoder(self, x):
        x,_ = self.backbone(x)
        vlad = self.vlad_head(x, only_encoder=True)
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

    def calculate_coord(self, center_shift, H,W, Hc, Wc,B):

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

        x, skip = self.backbone(x)

        score = self.score_head(x).sigmoid()
        center_shift = self.loc_head(x).tanh()
        feat = self.desc_head(x, skip)
        seg =self.seg_head(x, skip)

        vlad = self.vlad_head(x)

        out = {"score": score, "coord": center_shift, "feat": feat, "vlad": vlad, "seg": seg}
        if self.depth:
            depth = self.depth_head(x, skip).sigmoid()
            out["depth"] = depth
        return out

    def post_processing(self,out, H,W):
        score, center_shift, feat = out["score"], out["coord"], out["feat"]
        score = self.remove_border(score)
        B, _, Hc, Wc = score.shape
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
            seg = out["seg"]
            coord_norm = self.normalize_coord(coord, H,W)
            feat = self.sample_feat(feat, coord_norm)
            seg = self.sample_seg(seg, coord_norm)
            out["seg"] = seg

        out["feat"] = feat
        out["coord"] = coord
        out["score"] = score
        return out
    
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
        coord_norm[:, 0] = (coord[:, 0] / (float(W - 1) / 2.)) - 1.
        coord_norm[:, 1] = (coord[:, 1] / (float(H - 1) / 2.)) - 1.
        coord_norm = coord_norm.permute(0, 2, 3, 1)
        return coord_norm


class KP2DTinyV3(torch.nn.Module):
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
    use_color: bool
    do_cross: bool
    with_drop: bool
    nfeatures: int
    device: str
    channel_dims: list
    bn_momentum: float
    nClasses: int
    num_clusters: int
    downsample: int
    use_attention: bool

    def __init__(self, use_color=True, do_cross=True, with_drop=True,
                 nfeatures=256, device='cpu', channel_dims=[32, 64, 128, 256, 256, 512],
                 bn_momentum=0.1, nClasses=8, num_clusters=64, downsample=3, use_attention=False, encoder_dim=None,
                 mem_efficient=False, upscale_method="pixelshuffle", remove_netvlad=False, leaky_relu=True
                 ,remove_softmax=False, depth = False, global_descriptor_method="netvlad",**kwargs):
        super().__init__()
        print("Dropout:", with_drop)
        self.device = device
        self.with_drop = with_drop
        self.nfeatures = nfeatures
        self.downsample = downsample
        self.nClasses = nClasses
        self.sample_segmentation = False
        self.use_color = use_color
        self.do_cross = do_cross
        self.fuse_score_loc = True
        self.remove_softmax = remove_softmax
        self.depth = depth
        self.num_clusters = num_clusters
        self.global_descriptor_method = global_descriptor_method
        if self.use_color:
            c0 = 3
        else:
            c0 = 1

        self.bn_momentum = bn_momentum
        self.cross_ratio = 2.0

        c1, c2, c3, c4, c5, d1 = channel_dims
        if encoder_dim is not None:
            self.encoder_dim = encoder_dim
        else:
            self.encoder_dim = c4 

        self.backbone = BackBone(c0, c1, c2, c3, c4, downsample, with_drop, bn_momentum=0.1, leaky_relu=leaky_relu)

        self.score_loc_head = SimpleTaskHead(c4, c4, 3, bn_momentum=self.bn_momentum, with_drop=with_drop,
                                             leaky_relu=leaky_relu)

        # self.desc_head = UpscaleHead(c4, c4,c3*4,c3+c4,c4*2, nfeatures, with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
        # correction to be in line with legacy             --v


        if use_attention:
            # if not lite_feat_seg:
            #     self.seg_head = SegmentationFeatHeadATT(c4, c5, c4 + c3, nClasses, nfeatures, d1, with_drop, bn_momentum=self.bn_momentum,
            #                                         upscale_method=upscale_method, leaky_relu=leaky_relu)
            #else:
            self.seg_head = SegmentationFeatHeadLightATT(c4, c5, c4 + c3, nClasses, nfeatures, d1, with_drop, bn_momentum=self.bn_momentum,
                                                upscale_method=upscale_method, leaky_relu=leaky_relu, depth=depth)
        else:
            
            # if not lite_feat_seg:
            #     raise NotImplementedError("Not implemented")
            #else:
            self.seg_head = SegmentationFeatHeadLight(c4, c5, c4 + c3, nClasses, nfeatures, d1, with_drop, bn_momentum=self.bn_momentum,
                                                    upscale_method=upscale_method, leaky_relu=leaky_relu, depth = depth)

        # Netvlad
        self.vlad_head = VPRHead(c4, self.encoder_dim, num_clusters, with_drop,
                                     bn_momentum=self.bn_momentum, mem_efficient=mem_efficient,
                                     remove_netvlad=remove_netvlad, leaky_relu=leaky_relu, method=global_descriptor_method)

    # if self.depth:
    #     self.depth_head = SegmentationHeadATT(c4, c5, c4+c3, 1, d1,with_drop, bn_momentum=self.bn_momentum, upscale_method=upscale_method,leaky_relu=leaky_relu)
        if leaky_relu:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        self.cell = pow(2, self.downsample)
        self.training = True
        self.global_desc_dim = self.vlad_head.global_desc_dim
        self.softmax = torch.nn.Softmax2d()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        print(self.downsample)

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

    def init_netvlad(self, clsts, traindescs):
        self.vlad_head.netvlad.init_params(clsts, traindescs)

    def get_num_clusters(self):
        return self.vlad_head.netvlad.num_clusters

    def get_netvlad_dim(self):
        return self.global_desc_dim

    def freeze_backbone(self):
        for layer in self.backbone.children():
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_segmentation(self, except_last_layer=False):
        self.seg_head.freeze(except_last_layer)

    def fuse(self):
        fuse_modules(self.backbone)
        fuse_modules(self.score_head)
        fuse_modules(self.loc_head)
        fuse_modules(self.desc_head)
        fuse_modules(self.seg_head)
        fuse_modules(self.vlad_head)

    def only_encoder(self, x):
        x, _ = self.backbone(x)
        vlad = self.vlad_head(x, only_encoder=True)
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

    def calculate_coord(self, center_shift, H, W, Hc, Wc, B):

        step = (self.cell - 1) / 2.

        center_base = image_grid(B, Hc, Wc,
                                 dtype=center_shift.dtype,
                                 device=center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step
        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W - 1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H - 1)
        return coord

    # @timing_decorator
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

        x, skip = self.backbone(x)
        if self.fuse_score_loc:
            score_loc = self.score_loc_head(x)
            score = score_loc[:, 0:1]
            center_shift = score_loc[:, 1:3]
        else:
            score = self.score_head(x)
            center_shift = self.loc_head(x)
        score = score.sigmoid()
        center_shift = center_shift.tanh()
        if self.depth:
             seg, feat, depth = self.seg_head(x, skip)

        else:
            seg, feat = self.seg_head(x, skip)

        if not self.training and not self.remove_softmax:
            seg = self.softmax(seg)

        vlad = self.vlad_head(x)

        out = {"score": score, "coord": center_shift, "feat": feat, "vlad": vlad, "seg": seg}

        if self.depth:
            out["depth"] = depth.sigmoid()
        return out


    def post_processing(self, out, H, W):
        score, center_shift, feat = out["score"], out["coord"], out["feat"]

        score = self.remove_border(score)
        B, _, Hc, Wc = score.shape
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
            seg = out["seg"]
            coord_norm = self.normalize_coord(coord, H, W)
            feat = self.sample_feat(feat, coord_norm)
            seg = self.sample_seg(seg, coord_norm)
            out["seg"] = seg


        out["feat"] = feat
        out["coord"] = coord
        out["score"] = score

        return out

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
        coord_norm[:, 0] = (coord[:, 0] / (float(W - 1) / 2.)) - 1.
        coord_norm[:, 1] = (coord[:, 1] / (float(H - 1) / 2.)) - 1.
        coord_norm = coord_norm.permute(0, 2, 3, 1)
        return coord_norm

