import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://amaarora.github.io/posts/2020-08-30-gempool.html
class GeM(nn.Module):
    def __init__(self, c, p=3, eps=1e-6, unshuffle=4):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = unshuffle * unshuffle
        if unshuffle > 1:
            self.unshuffle = nn.PixelUnshuffle(unshuffle)
        else:
            self.unshuffle = None

    def get_factor(self):
        return self.f

    def forward(self, x):
        if self.unshuffle is not None:
            x = self.unshuffle(x)
        x = self.gem(x, p=self.p, eps=self.eps)
        x = x.flatten(1)
        return x

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
