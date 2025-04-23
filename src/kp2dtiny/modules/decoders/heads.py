from torch.nn import Module, Conv2d, PixelShuffle, Dropout2d
from ..base import AnnotatedConvBnReLUModel, TransposedConvUpsampleModel
import torch
from torch import quantization


class SimpleTaskHead(Module):
    def __init__(self, c_in, c_hidden,c_out, bn_momentum=0.1, with_drop=False,leaky_relu=True):
        super().__init__()
        self.convDa = AnnotatedConvBnReLUModel(c_in, c_hidden, kernel_size=3, stride=1, padding=1, bias=False, bn_momentum=bn_momentum,leaky_relu=leaky_relu)
        self.convDb = Conv2d(c_hidden, c_out, kernel_size=3, stride=1, padding=1)
        self.with_drop = with_drop
        self.dropout = Dropout2d(0.2)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
    def forward(self, x):

        x = self.convDa(x)
        if self.with_drop:
            x = self.dropout(x)
        x = self.quant(x)
        x = self.convDb(x)
        x = self.dequant(x)
        return x

class UpscaleHead(Module):
    def __init__(self, c0, c1,c2, c3, c4, c5, with_drop, bn_momentum=0.1, upscale_method='pixelshuffle', leaky_relu=True):
        super().__init__()
        if upscale_method == 'pixelshuffle':
            self.upsample = PixelShuffle(upscale_factor=2)
        elif upscale_method == 'convtranspose':
            self.upsample = TransposedConvUpsampleModel(c2, leaky_relu=leaky_relu)
        else:
            raise NotImplementedError("Upscale method not implemented")
        
        self.convA = AnnotatedConvBnReLUModel(c0, c1, kernel_size=3, stride=1, padding=1, bias=False, bn_momentum=bn_momentum,leaky_relu=leaky_relu)
        # self.convB = AnnotatedConvBnReLUModel(c1, c2, kernel_size=3, stride=1, padding=1, bias=False,
        #                                       bn_momentum=bn_momentum, leaky_relu=leaky_relu)
        self.convB = Conv2d(c1, c2, kernel_size=3, stride=1, padding=1) #TODO change to annotatedconvbnrelu
        self.confAa = AnnotatedConvBnReLUModel(c3, c4, kernel_size=3, stride=1, padding=1, bias=False, bn_momentum=bn_momentum,leaky_relu=leaky_relu)
        self.confBb = Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.dropout = Dropout2d(0.2)
        self.with_drop = with_drop
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
    def forward(self, x, skip):
        x = self.convA(x)
        if self.with_drop:
            x = self.dropout(x)
        x = self.quant(x)
        x = self.convB(x)
        x = self.dequant(x)
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.confAa(x)
        x = self.quant(x)
        x = self.confBb(x)
        x = self.dequant(x)
        return x
