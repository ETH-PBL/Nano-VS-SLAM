import torch
from torch import nn, einsum, quantization

from einops import rearrange

# https://github.com/lucidrains/segformer-pytorch/blob/main/segformer_pytorch/segformer_pytorch.py
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def rearrange_torch(q: torch.Tensor, heads: int):
    b, hc, x, y = q.size()
    b_new = b * heads
    hc_new = hc // heads
    q_reshaped = q.view(b, heads, hc_new, x, y)
    q_permuted = q_reshaped.permute(0, 1, 3, 4, 2)
    q_rearranged = q_permuted.contiguous().view(b_new, x * y, hc_new)
    return q_rearranged
def rearrange_torch_inverse(out: torch.Tensor, heads: int, h: int, w: int) -> torch.Tensor:
    b = out.shape[0] // heads  # Calculating 'b' based on the total number of elements and 'heads'
    c = out.shape[-1]  # Assuming 'c' is the size of the last dimension as per the input pattern
    out = out.view(b, heads, h, w, c)  # Correcting view to match expected dimensions
    out = out.permute(0, 1, 4, 2, 3)  # b, heads, c, h, w
    out = out.contiguous().view(b, heads * c, h, w)  # Flattening heads and channels
    return out

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads
        x = self.quant(x)

        q = self.to_q(x)
        kv = self.to_kv(x)

        q = self.dequant(q)
        kv = self.dequant(kv).chunk(2, dim=1)
        k, v = kv[0], kv[1]

        #q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q = rearrange_torch(q, heads)
        k = rearrange_torch(k, heads)
        v = rearrange_torch(v, heads)

        # q = rearrange(q, 'b (h c) x y -> (b h) (x y) c', h=heads)
        # k = rearrange(k, 'b (h c) x y -> (b h) (x y) c', h=heads)
        # v = rearrange(v, 'b (h c) x y -> (b h) (x y) c', h=heads)
        # q = rearrange_helper(q, heads)
        # k = rearrange_helper(k, heads)
        # v = rearrange_helper(v, heads)

        #q_3, k_3, v_3 = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        #sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1))
        sim = sim * self.scale

        attn = sim.softmax(dim = -1)
        out = torch.matmul(attn, v)
        #out = einsum('b i j, b j d -> b i d', attn, v)
        out = self.quant(out)
        #out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        out = rearrange_torch_inverse(out, heads, h, w)
        out = self.to_out(out)
        out = self.dequant(out)
        return out

class EfficientSelfAttentionLegacy(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads
        x = self.quant(x)

        q = self.to_q(x)
        kv = self.to_kv(x)

        q = self.dequant(q)
        kv = self.dequant(kv).chunk(2, dim=1)
        k, v = kv[0], kv[1]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = self.quant(out)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        out = self.to_out(out)
        out = self.dequant(out)
        return out


class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor,
        activation = "gelu"
    ):
        if activation == "gelu":
            _act = nn.GELU()
        elif activation == "relu":
            _act = nn.ReLU()
        else:
            raise NotImplementedError("activation not implemented")
        
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            _act,
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

class SegFormerAttentionModule(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = PreNorm(c, EfficientSelfAttention(dim = c, heads = 4, reduction_ratio = 2))
        self.mff =  PreNorm(c, MixFeedForward(dim = c, expansion_factor = 2, activation='gelu'))

    def forward(self,x):
        x = self.att(x)
        x = self.mff(x)
        return x
