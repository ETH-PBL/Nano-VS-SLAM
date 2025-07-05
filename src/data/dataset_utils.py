import numpy as np
from math import pi
from torchvision.transforms import v2
import torchgeometry as tgm
import torch
from utils.utils import load_json


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


def get_transforms(im_h, im_w, d_f=4, n_classes=183, val=False):
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
    transform_pre_seg = [
        v2.Resize([im_h, im_w], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze,
    ]
    if n_classes == 28:
        transform_pre_seg.append(get_coco_class_transform())

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
            out = {}
            image = examples["image"]
            seg_mask = examples["annotation"]
            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(
                torch.tensor(np.array(seg_mask)).unsqueeze(0)
            ).float()

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = image_t.mul(2.0).sub(1.0)
            warped_image = warped_image.mul(2.0).sub(1.0)

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
            homography = (
                torch.tensor(sample_homography([im_h, im_w])).view(1, 3, 3).float()
            )

            # image = image.convert('RGB')
            image_t = transform_pre(image).float()
            seg_mask_t = transform_pre_seg(
                torch.tensor(np.array(seg_mask)).unsqueeze(0)
            ).float()

            warped_image = warper(image_t, homography)
            warped_seg_mask = warper(seg_mask_t, homography)

            warped_seg_mask = transform_post_seg(warped_seg_mask)
            seg_mask_t = transform_post_seg(seg_mask_t)

            image_t = transforms_post(image_t).mul(2.0).sub(1.0)
            warped_image = transforms_post(warped_image).mul(2.0).sub(1.0)

            out["image"] = image_t.squeeze(0)
            out["image_aug"] = warped_image.squeeze(0)

            out["seg"] = seg_mask_t.squeeze(0)
            out["seg_aug"] = warped_seg_mask.squeeze(0)

            out["homography"] = homography.squeeze(0)
            return out

    return transforms
