import numpy as np
import datasets

from math import pi
import torch
from torchvision.transforms import v2
import torchgeometry as tgm
from .scene_parse_mapping import get_mapping


def to_tensor_sample(sample, tensor_type="torch.FloatTensor"):
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
    transform = v2.ToTensor()
    sample["image"] = transform(sample["image"]).type(tensor_type)
    return sample


def spatial_augment_sample(sample):
    """Apply spatial augmentation to an image (flipping and random affine transformation)."""
    augment_image = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )
    sample["image"] = augment_image(sample["image"])

    return sample


def sample_homography(
    shape,
    perspective=True,
    scaling=True,
    rotation=True,
    translation=True,
    n_scales=100,
    n_angles=100,
    scaling_amplitude=0.2,
    perspective_amplitude=0.2,
    patch_ratio=0.7,
    max_angle=pi / 2,
):
    """Sample a random homography that includes perspective, scale, translation and rotation operations."""

    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:, 1] *= hw_ratio

    if perspective:
        perspective_amplitude_x = np.random.normal(0.0, perspective_amplitude / 2, (2))
        perspective_amplitude_y = np.random.normal(
            0.0, hw_ratio * perspective_amplitude / 2, (2)
        )

        perspective_amplitude_x = np.clip(
            perspective_amplitude_x,
            -perspective_amplitude / 2,
            perspective_amplitude / 2,
        )
        perspective_amplitude_y = np.clip(
            perspective_amplitude_y,
            hw_ratio * -perspective_amplitude / 2,
            hw_ratio * perspective_amplitude / 2,
        )

        pts2[0, 0] -= perspective_amplitude_x[1]
        pts2[0, 1] -= perspective_amplitude_y[1]

        pts2[1, 0] -= perspective_amplitude_x[0]
        pts2[1, 1] += perspective_amplitude_y[1]

        pts2[2, 0] += perspective_amplitude_x[1]
        pts2[2, 1] -= perspective_amplitude_y[0]

        pts2[3, 0] += perspective_amplitude_x[0]
        pts2[3, 1] += perspective_amplitude_y[0]

    if scaling:
        random_scales = np.random.normal(1, scaling_amplitude / 2, (n_scales))
        random_scales = np.clip(
            random_scales, 1 - scaling_amplitude / 2, 1 + scaling_amplitude / 2
        )

        scales = np.concatenate([[1.0], random_scales], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (
            np.expand_dims(pts2 - center, axis=0)
            * np.expand_dims(np.expand_dims(scales, 1), 1)
            + center
        )
        valid = np.arange(n_scales)  # all scales are valid except scale=1
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = (
            np.min(pts2 - [-1.0, -hw_ratio], axis=0),
            np.min([1.0, hw_ratio] - pts2, axis=0),
        )
        pts2 += np.expand_dims(
            np.stack(
                [
                    np.random.uniform(-t_min[0], t_max[0]),
                    np.random.uniform(-t_min[1], t_max[1]),
                ]
            ),
            axis=0,
        )

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.0], angles], axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                axis=1,
            ),
            [-1, 2, 2],
        )
        rotated = (
            np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat,
            )
            + center
        )

        valid = np.where(
            np.all(
                (rotated >= [-1.0, -hw_ratio]) & (rotated < [1.0, hw_ratio]),
                axis=(1, 2),
            )
        )[0]

        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = rotated[idx]

    pts2[:, 1] /= hw_ratio

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
    )

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.0]]).reshape(3, 3)
    return homography


def apply_random_noise(img, noise_level=0.1, p=0.5):
    device = img.device
    return torch.clip(
        img + (torch.rand(1, device=device) > p) * noise_level * torch.randn_like(img),
        0,
        1.0,
    )


def get_dataset(
    path,
    size,
    split="train",
    device="cpu",
    n_classes=7,
):
    assert n_classes in (150, 7)
    assert split in ("train", "validation")

    dataset = datasets.load_dataset("scene_parse_150", split=split, cache_dir=path)

    if n_classes == 7:
        new_class_mapping = get_mapping()

        def map_classes(mask, mapping):
            # Flatten the mask to 1D for mapping
            flat_mask = mask.flatten()

            # Use torch.take to apply the mapping
            mapped_flat_mask = torch.take(mapping.long(), flat_mask.long())

            # Reshape back to the original mask shape
            new_mask = mapped_flat_mask.view_as(mask)
            return new_mask

        class_transform = v2.Lambda(lambda mask: map_classes(mask, new_class_mapping))

    else:
        class_transform = None

    transforms_post = v2.Compose(
        [
            v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
    )
    to_tensor = v2.ToTensor()
    resize = v2.Resize(size, interpolation=v2.InterpolationMode.NEAREST)
    augment_image = v2.Compose(
        [
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )

    def transforms(examples):
        (
            transformed_images,
            transformed_masks,
            warped_images,
            warped_masks,
            homographies,
        ) = [], [], [], [], []
        warper = tgm.HomographyWarper(size[0], size[1], mode="nearest")
        out = {}
        for image, seg_mask in zip(examples["image"], examples["annotation"]):
            homography = torch.tensor(sample_homography(size)).view(1, 3, 3).float()

            image = image.convert("RGB")

            # transformed = geo_transform(image=np.array(image), mask=np.array(seg_mask))
            # image_t = to_tensor(transformed["image"]).unsqueeze(0)
            image_t = to_tensor(image).unsqueeze(0)
            image_t = resize(image_t)
            # seg_mask_t =  torch.Tensor(transformed["mask"])
            seg_mask_t = torch.Tensor(np.array(seg_mask))
            seg_mask_t = resize(seg_mask_t.unsqueeze(0).unsqueeze(0).float())
            image_t, seg_mask_t = augment_image(image_t, seg_mask_t)

            if class_transform:
                seg_mask_t = class_transform(seg_mask_t)
            seg_mask_t = seg_mask_t.float()

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)

            image_t = apply_random_noise(transforms_post(image_t)).mul(2.0).sub(1.0)
            warped_image = (
                apply_random_noise(transforms_post(warped_image)).mul(2.0).sub(1.0)
            )

            transformed_images.append(image_t.squeeze(0).to(device))
            transformed_masks.append(seg_mask_t.squeeze(0).to(device))

            warped_images.append(warped_image.squeeze(0).to(device))
            warped_masks.append(warped_seg_mask.squeeze(0).to(device))
            homographies.append(homography.squeeze(0).to(device))

        out["image"] = transformed_images
        out["image_aug"] = warped_images

        out["seg"] = transformed_masks
        out["seg_aug"] = warped_masks

        out["homography"] = homographies
        return out

    dataset.set_transform(transforms)

    return dataset
