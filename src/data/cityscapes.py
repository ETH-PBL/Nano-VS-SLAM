from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision.transforms import v2
import torch
import torchgeometry as tgm
from .dataset_utils import sample_homography

def get_city_scapes_class_transform():
    classes = Cityscapes.classes

    mapping = {label.id: label.train_id for label in classes}
    new_class_indices = torch.full((256,), 0, dtype=torch.uint8)

    for original_class, new_class_index in mapping.items():
        new_class_indices[int(original_class)] = new_class_index

    return v2.Lambda(lambda mask: map_classes(mask, new_class_indices))

def get_cityscapes_transforms(im_h, im_w, d_f=4, val=False):
    # norm = v2.Lambda(lambda img: img.div(127.5)-1.)$

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
        unsqueeze,
        get_city_scapes_class_transform()
    ]

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
    if val:
        def transforms(examples):
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            seg_mask = examples["annotation"]
            homography = torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(torch.tensor(np.array(seg_mask)).unsqueeze(0)).float()

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = transforms_post(image_t).mul(2.).sub(1.)
            warped_image = transforms_post(warped_image).mul(2.).sub(1.)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

            out["homography"] = homography.squeeze(0)
            return out
    else:
        def transforms(examples):
            # Transforms for coco dataset
            out = {}
            image = examples["image"]
            seg_mask = examples["annotation"]
            homography = torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(torch.tensor(np.array(seg_mask)).unsqueeze(0)).float()

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = transforms_post(image_t).mul(2.).sub(1.)
            warped_image = transforms_post(warped_image).mul(2.).sub(1.)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

            out["homography"] = homography.squeeze(0)
            return out

    return transforms



def map_classes(mask, mapping):
    # Flatten the mask to 1D for mapping
    flat_mask = mask.flatten()

    # Use torch.take to apply the mapping
    mapped_flat_mask = torch.take(mapping.long(), flat_mask.long())

    # Reshape back to the original mask shape
    new_mask = mapped_flat_mask.view_as(mask)
    return new_mask
class CityScapeLoader(Dataset):
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
        "description": "Cityscapes",
        "url": "https://cityscapes-dataset.com/",
        "tasks": ["segmentation", "visloc", "keypoints"],
    }
    def __init__(self, root_dir, data_transform=None, split="train"):
        assert split in ["train", "val"]

        super().__init__()
        self._dataset = Cityscapes(root_dir, split=split, target_type='semantic')
        self.data_transform = data_transform
        self._map_fun = self._class_mapping()

        #self.mapping[-1] = 255

    def __len__(self):
        return len(self._dataset)

    def get_n_classes(self):
        return 19

    def __getitem__(self, idx):
        
        image, annotation = self._dataset[idx]

        if image.mode == 'L':
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {'image': image_new, 'idx': idx}
        else:
            sample = {'image': image, 'idx': idx}

        sample['annotation'] = annotation

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

    def _class_mapping(self):
        classes = self._dataset.classes

        mapping = {label.id: label.train_id for label in classes}
        new_class_indices = torch.full((256,), 0, dtype=torch.uint8)

        for original_class, new_class_index in mapping.items():
            new_class_indices[int(original_class)] = new_class_index

        return  v2.Lambda(lambda mask: map_classes(mask, new_class_indices))
