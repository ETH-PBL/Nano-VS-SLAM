from torch.nn import Module, Dropout2d
from ..base import AnnotatedConvBnReLUModel, L2Norm
from ..aggregators.netvlad import NetVLAD, NetVLADMemoryEfficient
from ..aggregators.gem import GeM
from ..aggregators.convap import ConvAP


class VPRHead(Module):
    def __init__(
        self,
        c_in,
        encoder_dim,
        num_clusters,
        with_drop,
        bn_momentum=0.1,
        mem_efficient=False,
        remove_netvlad=False,
        leaky_relu=True,
        method="netvlad",
    ):
        super().__init__()
        self.convlad1 = AnnotatedConvBnReLUModel(
            c_in,
            encoder_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.convlad2 = AnnotatedConvBnReLUModel(
            encoder_dim,
            encoder_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.convlad3 = AnnotatedConvBnReLUModel(
            encoder_dim,
            encoder_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=bn_momentum,
            leaky_relu=leaky_relu,
        )

        self.l2 = L2Norm()
        self.with_drop = with_drop
        self.dropout = Dropout2d(0.2)
        self.remove_netvlad = remove_netvlad
        if method == "netvlad":
            if not remove_netvlad:
                if mem_efficient:
                    self.netvlad = NetVLADMemoryEfficient(
                        dim=encoder_dim, num_clusters=num_clusters, vladv2=False
                    )
                else:
                    self.netvlad = NetVLAD(
                        dim=encoder_dim, num_clusters=num_clusters, vladv2=False
                    )
                self.global_desc_dim = self.netvlad.get_desc_size()
            else:
                self.global_desc_dim = 0
        elif method == "gem":
            self.netvlad = GeM(encoder_dim, unshuffle=4)
            self.global_desc_dim = encoder_dim * self.netvlad.get_factor()
        elif method == "convap":
            s = 4
            self.netvlad = ConvAP(encoder_dim, encoder_dim, s, s)
            self.global_desc_dim = encoder_dim * s * s

    def forward(self, x, only_encoder: bool = False):
        vlad = self.convlad1(x)
        if self.with_drop:
            vlad = self.dropout(vlad)
        vlad = self.convlad2(vlad)
        vlad = self.convlad3(vlad)
        if not self.remove_netvlad:
            if only_encoder:
                vlad = self.l2(vlad)
            else:
                vlad = self.netvlad(vlad)
        return vlad
