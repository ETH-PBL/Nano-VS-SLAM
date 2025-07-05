import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# form https://github.com/lyakaap/NetVLAD-pytorch
class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(
                valid_positive_dist, dim=1, keepdim=True
            )

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                1.0 - mask_anchor_negative
            )
            hardest_negative_dist, _ = torch.min(
                anchor_negative_dist, dim=1, keepdim=True
            )

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = labels.device

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = labels.device

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels  # Combine the two masks

    return mask


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/17cbfe0b68148d129a3ddaa227696496
    @author: wassname
    """
    intersection = (y_true * y_pred).abs().sum(dim=-1)
    sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# From https://github.com/zju3dv/deltar/blob/main/src/loss.py
class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = "SILog"

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


# From https://github.com/haofengac/MonoDepth-FPN-PyTorch/blob/master/main_fpn.py
import matplotlib.pyplot as plt


class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode="bilinear")
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(real) - torch.log(fake)) ** 2))
        return loss


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode="bilinear")
        loss = torch.mean(torch.abs(10.0 * real - 10.0 * fake))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode="bilinear")
        loss = torch.mean(torch.abs(torch.log(real) - torch.log(fake)))
        return loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, real, fake):
        mask = real > 0
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode="bilinear")
        fake = fake * mask
        diff = torch.abs(real - fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.0)
        part2 = F.threshold(diff**2 - delta**2, 0.0, -(delta**2.0)) + delta**2
        part2 = part2 / (2.0 * delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode="bilinear")
        loss = torch.sqrt(torch.mean(torch.abs(10.0 * real - 10.0 * fake) ** 2))
        return loss


class Grad(nn.Module):
    def __init__(self):
        super(Grad, self).__init__()
        fx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        fy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        weight = nn.Parameter(torch.stack([fy, fx]).float().unsqueeze(1))
        self.conv1.weight = weight

    # L1 norm
    def forward(self, fake, real):
        mask = torch.cat(
            [
                real.view(real.shape[0], 1, -1).gt(0.0),
                real.view(real.shape[0], 1, -1).gt(0.0),
            ],
            dim=1,
        )
        grad_real = self.conv1(real).view(real.shape[0], 2, -1)
        grad_fake = self.conv1(fake).view(fake.shape[0], 2, -1)
        return grad_fake, grad_real, mask


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real, mask=None):
        if mask is not None:
            grad_fake = grad_fake[mask]
            grad_real = grad_real[mask]
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (
            (grad_fake[:, :, None, :] @ grad_real[:, :, :, None])
            .squeeze(-1)
            .squeeze(-1)
        )
        fake_norm = torch.sqrt(torch.sum(grad_fake**2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real**2, dim=-1))

        return 1 - torch.mean(prod / (fake_norm * real_norm))


def get_coords(b, h, w):
    i_range = Variable(
        torch.arange(0, h).view(1, h, 1).expand(b, 1, h, w)
    )  # [B, 1, H, W]
    j_range = Variable(
        torch.arange(0, w).view(1, 1, w).expand(b, 1, h, w)
    )  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w, h]).view(1, 2, 1, 1))
    coords = coords * 2.0 / norm - 1.0
    coords = coords.permute(0, 2, 3, 1)

    return coords


def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode="bilinear", padding_mode="zeros")


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)

    weight = weight.to(img.device)
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    weight = weight.to(img.device)
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x


def imgrad_2(img):
    img = torch.mean(img, 1, True)
    fx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    fy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
    weight = nn.Parameter(torch.stack([fy, fx]).float().unsqueeze(1))
    conv1.weight = weight
    return conv1(img).view(img.shape[0], 2, -1)


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx) / 255.0)
