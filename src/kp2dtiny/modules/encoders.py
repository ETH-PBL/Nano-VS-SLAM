from .base import AnnotatedConvBnReLUModel
from torch.nn import Module, MaxPool2d, Dropout2d


class BackBone(Module):
    def __init__(
        self,
        c0,
        c1,
        c2,
        c3,
        c4,
        downsample,
        with_drop,
        bn_momentum=0.1,
        leaky_relu=True,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.conv1a = AnnotatedConvBnReLUModel(
            c0,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv1b = AnnotatedConvBnReLUModel(
            c1,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv2a = AnnotatedConvBnReLUModel(
            c2,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv2b = AnnotatedConvBnReLUModel(
            c2,
            c3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv3a = AnnotatedConvBnReLUModel(
            c3,
            c3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv3b = AnnotatedConvBnReLUModel(
            c3,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv4a = AnnotatedConvBnReLUModel(
            c4,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.conv4b = AnnotatedConvBnReLUModel(
            c4,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            bn_momentum=self.bn_momentum,
            leaky_relu=leaky_relu,
        )
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.dropout = Dropout2d(0.2)
        self.with_drop = with_drop
        self.downsample = downsample

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 2:
            x = self.pool(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        if self.with_drop:
            x = self.dropout(x)
        if self.downsample >= 3:
            x = self.pool(x)
        x = self.conv3a(x)
        skip = self.conv3b(x)
        if self.with_drop:
            skip = self.dropout(skip)
        if self.downsample >= 1:
            x = self.pool(skip)

        x = self.conv4a(x)
        x = self.conv4b(x)
        if self.with_drop:
            x = self.dropout(x)
        return x, skip
