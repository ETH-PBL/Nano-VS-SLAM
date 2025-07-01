import torch.nn.functional as F
from torch import nn, quantization


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2.0, dim=self.dim)


class AnnotatedConvBnReLUModel(nn.Module):
    def __init__(
        self,
        c0,
        c1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        bn_momentum=0.1,
        inplace=False,
        leaky_relu=True,
    ):
        super(AnnotatedConvBnReLUModel, self).__init__()
        self.conv = nn.Conv2d(
            c0, c1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(c1, momentum=bn_momentum)
        if leaky_relu:
            self.relu = nn.LeakyReLU(inplace=inplace)
        else:
            self.relu = nn.ReLU(inplace=inplace)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = x.contiguous()
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x


class ConvBnReLUModel(nn.Module):
    def __init__(
        self,
        c0,
        c1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        bn_momentum=0.1,
        inplace=True,
        leaky_relu=True,
    ):
        super(ConvBnReLUModel, self).__init__()
        self.conv = nn.Conv2d(
            c0, c1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(c1, momentum=bn_momentum)
        if leaky_relu:
            self.relu = nn.LeakyReLU(inplace=inplace)
        else:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = x.contiguous()
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransposedConvUpsampleModel(nn.Module):
    def __init__(
        self,
        c,
        kernel_size=3,
        stride=2,
        padding=2,
        bias=False,
        bn_momentum=0.1,
        inplace=True,
        leaky_relu=True,
    ):
        super(TransposedConvUpsampleModel, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            c,
            c // 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(c // 4, momentum=bn_momentum)
        if leaky_relu:
            self.relu = nn.LeakyReLU(inplace=inplace)
        else:
            self.relu = nn.ReLU(inplace=inplace)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = x.contiguous()
        x = self.quant(x)
        x = self.transposed_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x


class CustomPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(CustomPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        channels_div = channels // (self.upscale_factor**2)

        # Reshape to (batch_size, channels_div, r, r, in_height, in_width)
        x = x.view(
            batch_size,
            channels_div,
            self.upscale_factor,
            self.upscale_factor,
            in_height,
            in_width,
        )

        # Transpose to swap (r, in_height) and (r, in_width)
        # Resulting in (batch_size, channels_div, in_height, r, in_width, r)
        x = x.transpose(2, 4).transpose(3, 5)

        # Reshape to merge the upscale dimensions with the spatial dimensions
        return x.reshape(
            batch_size,
            channels_div,
            in_height * self.upscale_factor,
            in_width * self.upscale_factor,
        )
