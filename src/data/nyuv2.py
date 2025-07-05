import h5py
import torch
import torchgeometry as tgm
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from datasets.utils import sample_homography
import torchvision.transforms.v2 as v2
import numpy as np


def get_nyuv2_transforms(
    im_h, im_w, d_f=4, n_classes=13, max_depth=5000, min_depth=0.0, val=False
):
    unsqueeze = v2.Lambda(lambda img: img.unsqueeze(0))
    transform_pre = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize([im_h, im_w]),
            v2.RandomGrayscale(0.2),
            v2.RandomEqualize(0.2),
            unsqueeze,
        ]
    )
    transform_pre_seg = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze,
    ]

    transform_pre_seg = v2.Compose(transform_pre_seg)
    transform_post_seg = v2.Compose(
        [
            v2.Resize(
                [im_h // d_f, im_w // d_f], interpolation=v2.InterpolationMode.NEAREST
            )
        ]
    )

    transforms_post = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
    )

    warper = tgm.HomographyWarper(im_h, im_w, mode="nearest")

    if val:

        def transforms(examples):
            # Transforms for coco dataset
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            seg_mask = examples["annotation"]
            depth = examples["depth"]

            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(
                torch.tensor(np.array(seg_mask)).unsqueeze(0)
            ).float()
            depth_t = transform_pre_seg(
                torch.tensor(np.array(depth)).unsqueeze(0)
            ).float()
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)
            warped_depth = warper(depth_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = image_t.mul(2.0).sub(1.0)
            warped_image = warped_image.mul(2.0).sub(1.0)

            depth_t = transform_post_seg(depth_t)
            warped_depth = transform_post_seg(warped_depth)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)

            out["homography"] = homography.squeeze(0)

            return out
    else:

        def transforms(examples):
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            seg_mask = examples["annotation"]
            depth = examples["depth"]

            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(
                torch.tensor(np.array(seg_mask)).unsqueeze(0)
            ).float()
            depth_t = transform_pre_seg(
                torch.tensor(np.array(depth)).unsqueeze(0)
            ).float()
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)
            warped_depth = warper(depth_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = transforms_post(image_t).mul(2.0).sub(1.0)
            warped_image = transforms_post(warped_image).mul(2.0).sub(1.0)

            depth_t = transform_post_seg(depth_t)
            warped_depth = transform_post_seg(warped_depth)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)

            out["homography"] = homography.squeeze(0)

            return out

    return transforms


def get_nyuv2_depth_transforms(
    im_h, im_w, d_f=4, n_classes=13, max_depth=5.0, min_depth=0.0, val=False
):
    unsqueeze = v2.Lambda(lambda img: img.unsqueeze(0))
    transform_pre = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize([im_h, im_w]),
            v2.RandomGrayscale(0.2),
            v2.RandomEqualize(0.2),
            unsqueeze,
        ]
    )
    transform_pre_seg = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze,
    ]

    transform_pre_seg = v2.Compose(transform_pre_seg)
    transform_post_seg = v2.Compose(
        [
            v2.Resize(
                [im_h // d_f, im_w // d_f], interpolation=v2.InterpolationMode.NEAREST
            )
        ]
    )

    transforms_post = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
    )

    warper = tgm.HomographyWarper(im_h, im_w, mode="nearest")

    if val:

        def transforms(examples):
            # Transforms for coco dataset
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            depth = examples["depth"]

            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            depth_t = transform_pre_seg(
                torch.tensor(np.array(depth)).unsqueeze(0)
            ).float()
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)

            warped_image = warper(image_t, homography)
            warped_depth = warper(depth_t, homography)

            image_t = image_t.mul(2.0).sub(1.0)
            warped_image = warped_image.mul(2.0).sub(1.0)

            depth_t = transform_post_seg(depth_t)
            warped_depth = transform_post_seg(warped_depth)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)

            out["homography"] = homography.squeeze(0)

            return out
    else:

        def transforms(examples):
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            depth = examples["depth"]

            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            depth_t = transform_pre_seg(
                torch.tensor(np.array(depth)).unsqueeze(0)
            ).float()
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)

            warped_image = warper(image_t, homography)
            warped_depth = warper(depth_t, homography)

            image_t = transforms_post(image_t).mul(2.0).sub(1.0)
            warped_image = transforms_post(warped_image).mul(2.0).sub(1.0)

            depth_t = transform_post_seg(depth_t)
            warped_depth = transform_post_seg(warped_depth)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)

            out["homography"] = homography.squeeze(0)

            return out

    return transforms


#
# class NYUv2Dataset(Dataset):
#     def __init__(self, path, data_transform, device="conda"):
#         super(NYUv2Dataset, self).__init__()
#         self.path = path
#         self.transform = data_transform
#
#         self.data = h5py.File(self.path, 'r')
#         self.n_classes = self.data['names'].shape[1]
#         self.images = self.data['images']
#         self.depths = self.data['depths']
#         self.labels = self.data['labels']
#
#     def info(self):
#         return {
#             'n_classes': self.n_classes,
#             'n_samples': len(self.images),
#             'type': 'RGB, Depth, Segmentation',
#
#
#         }
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#
#         sample = {
#             'image': torch.tensor(self.images[index], dtype=torch.uint8).unsqueeze(0),
#             #'depth': self.depths[index],
#             'annotation': torch.tensor(self.labels[index].astype('int64'), dtype=torch.long).unsqueeze(0).unsqueeze(0)}
#         sample = self.transform(sample)
#
#         return sample
from datasets import load_from_disk, load_dataset
import shutil


class NYUv2Dataset(Dataset):
    info = {
        "description": "NYUv2 Depth",
        "url": "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2",
        "tasks": ["depth", "visloc", "keypoints"],
    }

    def __init__(self, path, n_classes=13, data_transform=None, split="train"):
        super(NYUv2Dataset, self).__init__()
        assert split in ("train", "validation")
        assert n_classes in (13, 40)
        self.path = Path(path) / split
        if self.path.exists():
            self.dataset = load_from_disk(self.path)
            print("Loaded from disk", self.path)
        else:
            self.dataset = load_dataset(
                "sayakpaul/nyu_depth_v2", split=split, cache_dir=Path(path) / "cache"
            )
            print("Loaded from huggingface", self.path)
            print("Saving to disk", self.path)
            self.dataset.save_to_disk(self.path)
            shutil.rmtree(Path(path) / "cache")
            self.dataset = load_from_disk(self.path)
            print("saved to disk", self.path)

        self.transform = data_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["depth"] = sample.pop("depth_map")
        sample = self.transform(sample)
        return sample


class NYUv2Dataset_extracted(Dataset):
    info = {
        "description": "NYUv2",
        "url": "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
        "tasks": ["depth", "segmentation", "visloc", "keypoints"],
    }

    def __init__(self, path, n_classes=13, data_transform=None, split="train"):
        super(NYUv2Dataset_extracted, self).__init__()
        assert split in ("train", "test")
        assert n_classes in (13, 40)
        self.path = Path(path)
        image_path = self.path / "image" / split
        depth_path = self.path / "depth" / split
        label_path = self.path / ("seg%d" % n_classes) / split

        self.images = sorted(list(image_path.glob("*.png")))
        self.depths = sorted(list(depth_path.glob("*.png")))
        self.labels = sorted(list(label_path.glob("*.png")))
        self.transform = data_transform
        self.n_classes = n_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {
            "image": Image.open(self.images[index]),
            "depth": Image.open(self.depths[index]),
            "annotation": Image.open(self.labels[index]),
        }
        sample = self.transform(sample)
        return sample


def apply_random_noise(img, noise_level=0.1, p=0.5):
    device = img.device
    return torch.clip(
        img + (torch.rand(1, device=device) > p) * noise_level * torch.randn_like(img),
        0,
        1.0,
    )


def get_dataset_nyuv2(path, split="train", device="cpu", size=(256, 256), n_classes=40):
    """Get NYUv2 dataset
    Parameters
    ----------
    split: str
        Dataset split, either 'train' or 'test'
    device: str
        Device to use for data
    size: tuple (W,H)
        Image size
    Returns
    -------
    NYUv2Dataset
        NYUv2 dataset

    """

    if split == "validation":
        split = "test"

    transform_pre_seg = v2.Compose(
        [v2.PILToTensor(), v2.Resize(size, interpolation=v2.InterpolationMode.NEAREST)]
    )
    transform_pre = v2.Compose([v2.PILToTensor(), v2.Resize(size)])

    norm = v2.Lambda(lambda img: img.div(255.0))
    RGB_augmentations = v2.Compose(
        [
            v2.RandomGrayscale(0.2),
            # v2.RandomEqualize(0.2),
            norm,
        ]
    )
    RGB_augmentations_post_warp = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
    )

    def transforms(examples):
        warper = tgm.HomographyWarper(size[0], size[1], mode="nearest")
        out = {}
        image = examples["image"]
        seg_mask = examples["annotation"]
        homography = (
            torch.tensor(sample_homography(size)).view(1, 3, 3).float().to(device)
        )

        image = image.convert("RGB")
        image_t = transform_pre(image).float().to(device)
        image_t = RGB_augmentations(image_t).unsqueeze(0)

        seg_mask_t = (
            transform_pre_seg(seg_mask).to(device, dtype=torch.uint8).unsqueeze(0) + 1
        ).float()

        warped_image = warper(image_t, homography)
        warped_seg_mask = warper(seg_mask_t, homography)

        image_t = apply_random_noise(image_t)
        warped_image = apply_random_noise(warped_image)

        image_t = RGB_augmentations_post_warp(image_t)
        warped_image = RGB_augmentations_post_warp(warped_image)

        out["image"] = image_t.squeeze(0).mul(2.0).sub(1.0)
        out["image_aug"] = warped_image.squeeze(0).mul(2.0).sub(1.0)

        out["seg"] = seg_mask_t.squeeze(0)
        out["seg_aug"] = warped_seg_mask.squeeze(0)

        out["homography"] = homography.squeeze(0)
        out["depth"] = transform_pre_seg(examples["depth"]).float().to(device)
        return out

    dataset = NYUv2Dataset_extracted(
        path, split=split, transform=transforms, n_classes=n_classes
    )
    return dataset
