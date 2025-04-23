from torch.nn import Module, Conv2d, PixelShuffle, ModuleList, Dropout2d, MaxPool2d
from ..base import AnnotatedConvBnReLUModel, TransposedConvUpsampleModel
from ..segformer import SegFormerAttentionModule
import torch
from torch import quantization

class SegmentationHead(Module):
    """
    Segmentation Head without Attention (V2)
    """

    def __init__(self, c_in, c_hidden, c_exp, c_out, d1, with_drop, bn_momentum=0.1, upscale_method='pixelshuffle',
                 leaky_relu=True):

        super().__init__()
        self.convs = ModuleList([
            AnnotatedConvBnReLUModel(c_in, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden + d1 // 4, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_exp, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            Conv2d(c_hidden, c_out, kernel_size=3, stride=1, padding=1)
        ])
        self.with_drop = with_drop
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        if upscale_method == 'pixelshuffle':
            self.upsample = PixelShuffle(upscale_factor=2)
            self.upsample2 = PixelShuffle(upscale_factor=2)
        elif upscale_method == 'convtranspose':
            self.upsample = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
            self.upsample2 = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
        else:
            raise NotImplementedError("Upscale method not implemented")
        self.dropout = Dropout2d(0.2)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.q = True

    def forward(self, x, skip):
        # Segmentation

        seg = self.convs[0](x)
        seg = self.convs[1](seg)

        seg = self.pool(seg)

        seg = self.convs[2](seg)
        seg = self.convs[3](seg)
        seg = self.convs[4](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample(seg)

        seg = torch.cat([seg, x], dim=1)

        seg = self.convs[5](seg)

        if self.with_drop:
            seg = self.dropout(seg)

        seg = self.upsample2(self.convs[6](seg))
        seg = torch.cat([seg, skip], dim=1)

        seg = self.convs[7](seg)
        if self.q:
            seg = self.quant(seg)
        seg = self.convs[8](seg)
        if self.q:
            seg = self.dequant(seg)
        return seg

    def freeze(self, except_last_layer=False):

        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = False

        if except_last_layer:
            for param in self.convs[8].parameters():
                param.requires_grad = True

class SegmentationFeatHeadLight(Module):
    """
    Segmentation Head fused with Feature Descriptor Head (V3)
    Computes Segmentation + Features (+ Depth if depth = True)
    """

    def __init__(self, c_in, c_hidden, c_exp, c_out, n_feat, d1, with_drop, bn_momentum=0.1,
                 upscale_method='pixelshuffle',
                 leaky_relu=True, depth=False):

        super().__init__()
        self.dim_split = c_hidden // 2
        c_hidden_b = c_hidden
        if depth:
            c_hidden_b = c_hidden_b + self.dim_split
        assert c_hidden % 2 == 0, "c_hidden must be divisible by 2"
        self.convs = ModuleList([
            AnnotatedConvBnReLUModel(c_in, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden + d1 // 4, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_exp, c_hidden_b, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            Conv2d(self.dim_split, c_out, kernel_size=3, stride=1, padding=1)
        ])

        self.depth = depth
        self.featB = Conv2d(self.dim_split, n_feat, kernel_size=3, stride=1, padding=1)
        if self.depth:
            self.featD = Conv2d(self.dim_split, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.with_drop = with_drop
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        if upscale_method == 'pixelshuffle':
            self.upsample = PixelShuffle(upscale_factor=2)
            self.upsample2 = PixelShuffle(upscale_factor=2)
            # self.upsample = CustomPixelShuffle(upscale_factor=2)
            # self.upsample2 = CustomPixelShuffle(upscale_factor=2)
        elif upscale_method == 'convtranspose':
            self.upsample = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
            self.upsample2 = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
        else:
            raise NotImplementedError("Upscale method not implemented")

        self.dropout = Dropout2d(0.2)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def freeze(self, except_last_layer=False):

        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = False

        if except_last_layer:
            for param in self.convs[8].parameters():
                param.requires_grad = True

    def forward(self, x, skip):
        # Segmentation

        seg = self.convs[0](x)
        seg = self.convs[1](seg)

        seg = self.pool(seg)

        seg = self.convs[2](seg)
        seg = self.convs[3](seg)

        seg = self.convs[4](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample(seg)
        seg = torch.cat([seg, x], dim=1)
        seg = self.convs[5](seg)
        seg = self.convs[6](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample2(seg)
        seg = torch.cat([seg, skip], dim=1)

        seg = self.convs[7](seg)
        seg = self.quant(seg)
        feat = self.featB(seg[:, :self.dim_split])
        if self.depth:
            depth = self.featD(seg[:, self.dim_split:self.dim_split * 2])
        seg_out = self.convs[8](seg[:, -self.dim_split:])
        seg_out = self.dequant(seg_out)
        if self.depth:
            return seg_out, feat, depth
        else:
            return seg_out, feat

class SegmentationHeadATT(Module):
    """
    Segmentation Head (V2) using Attention module
    Computes Segmentation
    """

    def __init__(self, c_in, c_hidden, c_exp, c_out, d1, with_drop, bn_momentum=0.1, upscale_method='pixelshuffle',
                 leaky_relu=True):

        super().__init__()
        self.convs = ModuleList([
            AnnotatedConvBnReLUModel(c_in, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            SegFormerAttentionModule(c_hidden),
            SegFormerAttentionModule(c_hidden),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden + d1 // 4, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_exp, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            Conv2d(c_hidden, c_out, kernel_size=3, stride=1, padding=1)
        ])
        self.with_drop = with_drop
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        if upscale_method == 'pixelshuffle':
            self.upsample = PixelShuffle(upscale_factor=2)
            self.upsample2 = PixelShuffle(upscale_factor=2)
        elif upscale_method == 'convtranspose':
            self.upsample = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
            self.upsample2 = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
        else:
            raise NotImplementedError("Upscale method not implemented")

        self.dropout = Dropout2d(0.2)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x, skip):
        # Segmentation

        seg = self.convs[0](x)
        seg = self.convs[1](seg)

        seg = self.pool(seg)

        seg = self.convs[2](seg)
        seg = self.convs[3](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample(seg)
        seg = torch.cat([seg, x], dim=1)
        seg = self.convs[4](seg)
        seg = self.convs[5](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample2(seg)
        seg = torch.cat([seg, skip], dim=1)
        seg = self.convs[6](seg)
        seg = self.quant(seg)
        seg = self.convs[7](seg)
        seg = self.dequant(seg)
        return seg

    def freeze(self, except_last_layer=False):

        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = False

        if except_last_layer:
            for param in self.convs[7].parameters():
                param.requires_grad = True


class SegmentationFeatHeadLightATT(Module):
    def __init__(self, c_in, c_hidden, c_exp, c_out, n_feat, d1, with_drop, bn_momentum=0.1,
                 upscale_method='pixelshuffle',
                 leaky_relu=True, depth=False):

        super().__init__()
        self.depth = depth
        self.dim_split = c_hidden // 2
        c_hidden_b = c_hidden
        if depth:
            c_hidden_b = c_hidden_b + self.dim_split
        assert c_hidden % 2 == 0, "c_hidden must be divisible by 2"
        self.convs = ModuleList([
            AnnotatedConvBnReLUModel(c_in, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            SegFormerAttentionModule(c_hidden),
            SegFormerAttentionModule(c_hidden),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden + d1 // 4, c_hidden, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_hidden, d1, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            AnnotatedConvBnReLUModel(c_exp, c_hidden_b, kernel_size=3, stride=1, padding=1, bias=False,
                                     bn_momentum=bn_momentum, leaky_relu=leaky_relu),
            Conv2d(self.dim_split, c_out, kernel_size=3, stride=1, padding=1)

        ])

        self.featB = Conv2d(self.dim_split, n_feat, kernel_size=3, stride=1, padding=1)
        if self.depth:
            self.featD = Conv2d(self.dim_split, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.with_drop = with_drop
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        if upscale_method == 'pixelshuffle':
            self.upsample = PixelShuffle(upscale_factor=2)
            self.upsample2 = PixelShuffle(upscale_factor=2)
        elif upscale_method == 'convtranspose':
            self.upsample = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
            self.upsample2 = TransposedConvUpsampleModel(d1, leaky_relu=leaky_relu)
        else:
            raise NotImplementedError("Upscale method not implemented")

        self.dropout = Dropout2d(0.2)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def freeze(self, except_last_layer=False):

        for layer in self.children():
            for param in layer.parameters():
                param.requires_grad = False

        if except_last_layer:
            for param in self.convs[7].parameters():
                param.requires_grad = True

    def forward(self, x, skip):
        # Segmentation

        seg = self.convs[0](x)
        seg = self.convs[1](seg)

        seg = self.pool(seg)

        seg = self.convs[2](seg)
        seg = self.convs[3](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample(seg)
        seg = torch.cat([seg, x], dim=1)
        seg = self.convs[4](seg)
        seg = self.convs[5](seg)
        if self.with_drop:
            seg = self.dropout(seg)
        seg = self.upsample2(seg)
        seg = torch.cat([seg, skip], dim=1)

        seg = self.convs[6](seg)
        seg = self.quant(seg)
        feat = self.featB(seg[:, :self.dim_split])
        if self.depth:
            depth = self.featD(seg[:, self.dim_split:self.dim_split * 2])
        seg_out = self.convs[7](seg[:, -self.dim_split:])
        seg_out = self.dequant(seg_out)
        if self.depth:
            return seg_out, feat, depth
        else:
            return seg_out, feat
