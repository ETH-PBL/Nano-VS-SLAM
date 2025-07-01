from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import v2
import torchgeometry as tgm
import torch
from utils.utils import load_json
from .dataset_utils import sample_homography
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


def get_coco_class_transform(
    mapping_path="src/data/cocostuff_mapping.json",
):  # TODO: remove hardcoding
    new_class_mapping = load_json(mapping_path)
    new_class_indices = torch.full((256,), 0, dtype=torch.uint8)

    for original_class, new_class_index in new_class_mapping.items():
        new_class_indices[int(original_class)] = new_class_index

    return v2.Lambda(lambda mask: map_classes(mask, new_class_indices))


def get_coco_transforms(
    im_h,
    im_w,
    d_f=4,
    n_classes=183,
    val=False,
    load_depth=False,
    min_depth=10,
    max_depth=65000,
):
    # norm = v2.Lambda(lambda img: img.div(127.5)-1.)$
    assert n_classes in [183, 28]

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
    if val:
        transform_pre = v2.Compose([v2.ToTensor(), v2.Resize([im_h, im_w]), unsqueeze])
    transform_pre_seg = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze,
    ]
    if n_classes == 28:
        transform_pre_seg.append(get_coco_class_transform())

    transform_pre_depth = [v2.Resize([im_h, im_w]), unsqueeze]
    transform_pre_depth = v2.Compose(transform_pre_depth)
    transform_pre_seg = v2.Compose(transform_pre_seg)
    transform_post_seg = v2.Compose(
        [
            v2.Resize(
                [im_h // d_f, im_w // d_f], interpolation=v2.InterpolationMode.NEAREST
            )
        ]
    )

    transform_post_depth = v2.Compose([v2.Resize([im_h // d_f, im_w // d_f])])

    transforms_post = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
    )

    warper = tgm.HomographyWarper(im_h, im_w, mode="nearest")

    def transforms(examples):
        # Transforms for coco dataset
        out = {}
        image = examples["image"]
        seg_mask = examples["annotation"]
        homography = torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()

        # image = image.convert('RGB')
        image_t = transform_pre(image).float()
        seg_mask_t = transform_pre_seg(
            torch.tensor(np.array(seg_mask)).unsqueeze(0)
        ).float()

        warped_image = warper(image_t, homography)
        warped_seg_mask = warper(seg_mask_t, homography)

        warped_seg_mask = transform_post_seg(warped_seg_mask)
        seg_mask_t = transform_post_seg(seg_mask_t)
        if not val:
            image_t = transforms_post(image_t)
            warped_image = transforms_post(warped_image)

        if load_depth:
            depth = examples["depth"]
            depth_t = transform_pre_depth(
                torch.tensor(np.array(depth)).unsqueeze(0)
            ).float()
            depth_t = torch.clamp(depth_t, min_depth, max_depth).div(max_depth)
            depth_t = transform_post_seg(depth_t)
            warped_depth = warper(depth_t, homography)
            warped_depth = transform_post_seg(warped_depth)
            out["depth"] = depth_t.squeeze(0)
            out["depth_aug"] = warped_depth.squeeze(0)
        image_t = image_t.mul(2.0).sub(1.0)
        warped_image = warped_image.mul(2.0).sub(1.0)

        out["image"] = image_t.squeeze(0)
        out["image_aug"] = warped_image.squeeze(0)

        out["seg"] = seg_mask_t.squeeze(0)
        out["seg_aug"] = warped_seg_mask.squeeze(0)

        out["homography"] = homography.squeeze(0)
        return out

    return transforms


class COCOLoader(Dataset):
    """
    Coco-stuff dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """

    info = {
        "description": "COCO-Stuff dataset 2017",
        "url": "https://cocodataset.org",
        "tasks": ["segmentation", "visloc", "keypoints"],
    }

    def __init__(self, root_dir, data_transform=None, split="train", depth=False):
        assert split in ["train", "val"]

        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images" / (split + "2017")
        self.annotation_dir = self.root_dir / "annotations" / (split + "2017")
        if depth:
            self.depth_dir = self.root_dir / "depth" / (split + "2017")

        self.depth = depth
        assert self.image_dir.exists()
        assert self.annotation_dir.exists()
        if depth:
            assert self.depth_dir.exists()

        self.files = []
        self.annotations = []
        self.depths = []

        for filename in self.image_dir.glob("*.jpg"):
            self.files.append(filename)
            self.annotations.append(self.annotation_dir / (filename.stem + ".png"))
            if depth:
                self.depths.append(self.depth_dir / ("depth_" + filename.stem + ".png"))
        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _read_rgb_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, idx):
        filename = self.files[idx]
        annotation = self.annotations[idx]
        image = self._read_rgb_file(filename)
        annotation_img = self._read_rgb_file(annotation)

        if image.mode == "L":
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {"image": image_new, "idx": idx}
        else:
            sample = {"image": image, "idx": idx}

        sample["annotation"] = annotation_img
        if self.depth:
            depth = self.depths[idx]
            depth_img = self._read_rgb_file(depth)
            sample["depth"] = depth_img

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
