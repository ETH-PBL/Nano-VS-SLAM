# Copyright 2020 Toyota Research Institute.  All rights reserved.

from pathlib import Path

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

from torch.utils.data import DataLoader
def normalize_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.
    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to
    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """

    sample['image'] = torch.sub(sample['image'], 0.5)
    sample['image'] = torch.mul(sample['image'], 2.0)

    sample['image_aug'] = torch.sub(sample['image_aug'], 0.5)
    sample['image_aug'] = torch.mul(sample['image_aug'], 2.0)
    return sample
class PatchesDataset(Dataset):
    """
    HPatches dataset class.
    Note: output_shape = (H, W)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    type: str
        Dataset subset to return from ['i', 'v', 'all']: 
        i - illumination sequences
        v - viewpoint sequences
        all - all sequences
    """
    def __init__(self, root_dir, use_color=True, data_transform=None, output_shape=None, type='all', mode='quantized_default'):

        super().__init__()
        self.type = type
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.output_shape = output_shape[::-1] # convert from H,W to W,H
        self.use_color = use_color

        base_path = Path(root_dir)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if self.type == 'i' and path.stem[0] != 'i':
                continue
            if self.type == 'v' and path.stem[0] != 'v':
                continue
            num_images = 5
            file_ext = '.ppm'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}

    @staticmethod
    def scale_homography(homography, original_scale, new_scale, pre):
        scales = np.divide(new_scale, original_scale)
        if pre:
            s = np.diag(np.append(scales, 1.))
            homography = np.matmul(s, homography)
        else:
            sinv = np.diag(np.append(1. / scales, 1.))
            homography = np.matmul(homography, sinv)
        return homography

    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):

        image = self._read_rgb_file(self.files['image_paths'][idx])
        warped_image = self._read_rgb_file(self.files['warped_image_paths'][idx])

        homography = np.array(self.files['homography'][idx])
        sample = {'image': image, 'image_aug': warped_image, 'homography': homography, 'index' : idx}

        # Apply transformations
        if self.output_shape is not None:
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         np.array(sample['image']).shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=False)
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         np.array(sample['image_aug']).shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=True)
        if self.data_transform:
            sample = self.data_transform(sample)
        return sample


    def _read_rgb_file(self, filename):
        return Image.open(filename)

def get_patches_dataset(val_path, img_shape, augmentation_mode='default', quantize=False, n_workers=0):
    """Prepare datasets for training, validation and test."""

    data_transforms = image_transforms_eval(augmentation_mode, img_shape, quantize=quantize)

    hp_dataset = PatchesDataset(root_dir=val_path, use_color=True,
                                output_shape=img_shape,
                                data_transform=data_transforms,
                                mode=augmentation_mode,
                                type='a')

    data_loader = DataLoader(hp_dataset,
                             batch_size=1,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=n_workers,
                             worker_init_fn=None,
                             sampler=None,
                             drop_last=True)
    return hp_dataset, data_loader

class image_transforms_eval():
    def __init__(self, mode, image_shape, quantize=False):
        self.transform = getattr(self, "_" + mode)
        self.image_shape = image_shape  # [H,W]
        self.quantize = quantize


    def _default(self, sample):
        image_transforms = transforms.Compose([transforms.Resize(self.image_shape),
                                               transforms.ToTensor()])
        for key in ['image', 'image_aug']:
            sample[key] = image_transforms(sample[key])
        sample = normalize_sample(sample)
        return sample

    def __call__(self, sample):
        return self.transform(sample)


