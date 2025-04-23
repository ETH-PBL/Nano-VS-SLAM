# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import v2
import torchgeometry as tgm
import torch
from utils.utils import load_json
from datasets.utils import sample_homography
from torchvision.datasets.cityscapes import Cityscapes
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# IMPORTANT:
# classes are from 0 to 181 with an additional 182 class for stuff
# unlabeled classes are denoted by 255


def map_classes(mask, mapping):
    # Flatten the mask to 1D for mapping
    flat_mask = mask.flatten()

    # Use torch.take to apply the mapping
    mapped_flat_mask = torch.take(mapping.long(), flat_mask.long())

    # Reshape back to the original mask shape
    new_mask = mapped_flat_mask.view_as(mask)
    return new_mask


def get_coco_class_transform(mapping_path="./data/cocostuff_mapping.json"):
    new_class_mapping = load_json(mapping_path)
    new_class_indices = torch.full((256,), 0, dtype=torch.uint8)

    for original_class, new_class_index in new_class_mapping.items():
        new_class_indices[int(original_class)] = new_class_index

    return v2.Lambda(lambda mask: map_classes(mask, new_class_indices))


def get_coco_transforms(im_h, im_w, d_f=4, val=False, load_depth=False,load_segmentation = False, min_depth=10, max_depth=65000):


    unsqueeze = v2.Lambda(lambda img: img.unsqueeze(0))
    transform_pre = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize([im_h, im_w]),
            v2.RandomGrayscale(0.2),
            v2.RandomEqualize(0.2),
            unsqueeze
        ]

    )
    transform_pre_seg = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze
    ]

    transform_pre_depth = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze
    ]

    transform_pre_depth = v2.Compose(transform_pre_depth)
    transform_pre_seg = v2.Compose(transform_pre_seg)
    transform_post_seg = v2.Compose(
        [
            v2.Resize([im_h // d_f, im_w // d_f], interpolation=v2.InterpolationMode.NEAREST)

        ]

    )

    transforms_post = v2.Compose([
        v2.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.GaussianBlur(3, sigma=(0.1, 1.0)),

    ])

    warper = tgm.HomographyWarper(im_h, im_w, mode='nearest')

    def transforms(examples):
        # Transforms for coco dataset
        out = {}

        image = examples["image"]
        homography = torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
        image_t = transform_pre(image).float()
        warped_image = warper(image_t, homography)

        if not val:
            image_t = transforms_post(image_t)
            warped_image = transforms_post(warped_image)
        image_t = image_t.mul(2.).sub(1.)
        warped_image = warped_image.mul(2.).sub(1.)

        out["image"] = image_t.squeeze(0)
        out["image_aug"] = warped_image.squeeze(0)
        out["homography"] = homography.squeeze(0)

        if load_segmentation:
            seg_mask = examples["annotation"]
            seg_mask_t = transform_pre_seg(torch.tensor(np.array(seg_mask)).unsqueeze(0)).float()

            warped_seg_mask = warper(seg_mask_t, homography)
            warped_seg_mask = transform_post_seg(warped_seg_mask)

            seg_mask_t = transform_post_seg(seg_mask_t)
            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

        if load_depth:
            depth = examples["depth"]
            depth_t = transform_pre_depth(torch.tensor(np.array(depth)).unsqueeze(0)).float()
            # clamp depth values and normalize
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)
            depth_t = transform_post_seg(depth_t)

            warped_depth = warper(depth_t, homography)
            warped_depth = transform_post_seg(warped_depth)

            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)



        return out
    return transforms


class SimpleDataset(Dataset):
    """
    Simple dataset class for loading images from a directory
    Should have the following st
    - Root:
    --- images
      ---img1.jpg
      ...
    --- segmentation
      ---seg_img1.jpg
      ...
    --- depth
        ---depth_img1.jpg
        ...

    """
    info = {
        "description": "Simple Dataset",
        "tasks": ["segmentation", "visloc", "keypoints"],
    }

    def __init__(self, root_dir, data_transform=None, split="train", depth=False, segmentation=False):
        assert split in ["train", "val"]

        super().__init__()
        self.depth = depth
        self.segmentation = segmentation

        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir /"images"

        if self.segmentation:
            self.segmentation_dir = self.root_dir / "segmentation"

        if self.depth:
            self.depth_dir = self.root_dir / "depth"

        assert self.root_dir.exists(), f"Error: {self.root_dir} does not exist"
        assert self.image_dir.exists(), f"Error: {self.image_dir} does not exist"

        if self.segmentation:
            assert self.segmentation_dir.exists(), f"Error: {self.segmentation_dir} does not exist"
        if self.depth:
            assert self.depth_dir.exists(), f"Error: {self.depth_dir} does not exist"


        self.files = []
        self.seg_masks = []
        self.depths = []

        for filename in self.image_dir.glob('*.jpg'):
            self.files.append(filename)
            if self.segmentation:
                self.seg_masks.append(self.segmentation_dir /("seg_"+  filename.stem + '.png'))
            if depth:
                self.depths.append(self.depth_dir / ("depth_"+ filename.stem + '.png'))
        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _read_rgb_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, idx):
        
        filename = self.files[idx]
        image = self._read_rgb_file(filename)


        if image.mode == 'L':
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {'image': image_new, 'idx': idx}
        else:
            sample = {'image': image, 'idx': idx}

        if self.segmentation:
            annotation = self.seg_masks[idx]
            annotation_img = self._read_rgb_file(annotation)
            sample['segmentation'] = annotation_img
        if self.depth:
            depth = self.depths[idx]
            depth_img = self._read_rgb_file(depth)
            sample['depth'] = depth_img

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
