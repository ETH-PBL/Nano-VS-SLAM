import torch
import torchvision

import kp2dtiny.modules.netvlad as netvlad
from kp2dtiny.modules.base import L2Norm

class VGG16Net(torch.nn.Module):
    def __init__(self, pretrained=True,encoder_dim = 512, num_clusters=64, vladv2=False):
        super(VGG16Net, self).__init__()

        encoder = torchvision.models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        self.layers = list(encoder.features.children())[:-2]
        self.netvlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
        self.encoder_dim = encoder_dim
        self.num_clusters = num_clusters
        self.l2norm = L2Norm(dim=1)
        if pretrained:
            self.freeze_backbone()

        self.encoder = torch.nn.Sequential(*self.layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        vlad = self.netvlad.forward(self.encoder(x))
        return None, None, None, vlad, None

    def only_encoder(self, x):
        return self.l2norm(self.encoder(x))

    def gather_info(self):
        info = {"encoder": {"num_layers": len(self.layers)}}
        return info
    def get_global_desc_dim(self):
        return self.netvlad.get_desc_size()
    def freeze_backbone(self):
        for l in self.layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
