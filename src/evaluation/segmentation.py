import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import cv2


def evaluate_segmentation(model, dataloader, n_classes, debug=False):
    """Evaluate a segmentation model on a dataset
    Args:
        model (nn.Module): the segmentation model to evaluate
        dataloader (DataLoader): the dataloader to use for evaluation
    Returns:
        float: the mean IoU score over the dataset
    """
    iou_mean = 0
    iou_macro_mean = 0
    acc_mean = 0
    f1_mean = 0
    model.eval()
    model.training = False

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader, 0),
            unit=" images",
            unit_scale=4,
            total=len(dataloader),
            smoothing=0,
            disable=False,
        )

        for i, sample in pbar:
            img = sample["image"].to(model.device)
            seg_gt = sample["seg"].to(model.device)
            b, c, H, W = img.shape

            out = model(img)
            out = model.post_processing(out, H, W)
            seg = out["seg"]

            tp, fp, fn, tn = smp.metrics.get_stats(
                seg.int(),
                seg_gt.int(),
                mode="multiclass",
                num_classes=n_classes,
                ignore_index=255,
            )

            iou_score = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )
            iou_score_macro = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="macro-imagewise"
            )
            acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
            f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou_mean += iou_score
            iou_macro_mean += iou_score_macro
            acc_mean += acc
            f1_mean += f1
            if debug:
                segmentation_debug_pred = (
                    seg[0].cpu().permute(1, 2, 0).numpy().astype(np.float32) / n_classes
                )
                segmentation_debug = (
                    seg_gt[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                segmentation_debug = segmentation_debug / n_classes

                concatenated_seg = np.hstack(
                    (segmentation_debug, segmentation_debug_pred)
                )
                cv2.imshow("Segmentation", concatenated_seg)
                cv2.waitKey(1)
            pbar.set_description(
                "Eval [ IoU {:.4f}, IoU_AVG {:.4f} Acc {:.4f}, Acc_AVG {:.4f}]".format(
                    float(iou_score),
                    float(iou_mean / i),
                    float(acc),
                    float(acc_mean / i),
                )
            )
    results = {
        "IoU": float(iou_mean / i),
        "accuracy": float(acc_mean / i),
        "f1": float(f1_mean / i),
        "IoU_macro": float(iou_macro_mean / i),
    }

    return results
