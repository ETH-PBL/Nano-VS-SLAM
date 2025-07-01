import torch
from tqdm import tqdm
import numpy as np
import cv2

# From https://github.com/zju3dv/deltar/blob/main/src/utils/metrics.py
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

# written with ChatGPT
def compute_errors_torch(gt, pred):
    gt = gt.float()  # Ensure floating point for division
    pred = pred.float()

    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    # Compute log10 manually since torch does not have a direct log10 function
    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()

    return dict(a1=a1.item(), a2=a2.item(), a3=a3.item(), abs_rel=abs_rel.item(), sq_rel=sq_rel.item(),
                rmse=rmse.item(), rmse_log=rmse_log.item(), silog=silog.item(), log_10=log_10.item())

def evaluate_depth_estimation(model, dataloader, debug=False):
    """Evaluate a segmentation model on a dataset
    Args:
        model (nn.Module): the segmentation model to evaluate
        dataloader (DataLoader): the dataloader to use for evaluation
    Returns:
        float: the mean IoU score over the dataset
    """
    metrics = RunningAverageDict()
    model.eval()
    model.training = False

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader, 0),
                    unit=' images',
                    unit_scale=4,
                    total=len(dataloader),
                    smoothing=0,
                    disable=False)

        for (i, sample) in pbar:
            img = sample['image'].to(model.device)
            depth_gt = sample['depth'].to(model.device)
            b, c, H, W = img.shape

            out = model(img)
            out = model.post_processing(out, H, W)
            depth = out['depth']
            metrics.update(compute_errors_torch(depth_gt, depth))

            m = metrics.get_value()
            pbar.set_description('Eval [AbsRel {:.4f}, RMSE {:.4f}, RMSE_log {:.4f}, Silog {:.4f}, log10 {:.4f}]'.format(m['abs_rel'], m['rmse'], m['rmse_log'], m['silog'], m['log_10']))

            if debug:
                depth_debug = depth[0].detach().cpu().numpy().squeeze()
                depth_debug = cv2.resize(depth_debug, (W, H))
                depth_gt_debug = depth_gt[0].cpu().numpy().squeeze()
                depth_gt_debug = cv2.resize(depth_gt_debug, (W, H))
                concatenated_depth = np.hstack((depth_debug, depth_gt_debug))
                cv2.imshow('Depth', concatenated_depth)
                cv2.waitKey(1)
    return metrics.get_value()
