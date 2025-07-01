from src.kp2dtiny.models.kp2dtiny import tiny_factory
from evaluation.visual_odometry import (
    demo,
    samsung_params,
)
import torch

weights_path = r"./demo_data/V3_S_A_p_best.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = tiny_factory("S_A", 28, v3=True).cpu()
model.load_state_dict(
    torch.load(weights_path, map_location=torch.device("cpu"))["state_dict"],
    strict=False,
)
model = model.to(device)
model.eval()
model.training = False

video_path = r"./demo_data/8.mp4"
out_path = r"./demo_data"

new_size = (240, 320)
cam_params = samsung_params(new_size)
demo(
    model,
    video_path,
    cam_params,
    device,
    new_size,
    plot=True,
    out_path=out_path,
    track=False,
)

# KITTI DEMO
# datasets_config = load_json(r"./datasets.json")
# kitti_path = datasets_config["kitti_path"]
# gt_path = datasets_config["kitti_gt_path"]
# video_name = datasets_config["kitti_video_path"]
# new_size = (240, 320)
# error = evaluate_visual_odometry(model, kitti_path, gt_path, video_name, device,new_size, plot=True)
