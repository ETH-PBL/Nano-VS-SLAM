import cv2
import torch
import os
from tqdm import tqdm

from pathlib import Path
import argparse
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def load_model(model_type):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform, device

def get_image(filename, transform):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    return input_batch, img.shape[:2]

def save_image(output, filename):
    output_image =(1- (output - output.min()) / (output.max() - output.min()) )* 65535
    output_image = output_image.astype('uint16')
    cv2.imwrite(filename, output_image)

def process_images(directory, out_directory, model_type):
    midas, transform, device = load_model(model_type)
    pbar = tqdm(directory.glob("*.jpg"))
    for filepath in  pbar:
        filename =  Path(filepath).stem + ".png"
        out_file = out_directory / f"depth_{filename}"
        if out_file.exists():
            continue
        input_batch, img_shape = get_image(str(filepath), transform)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        save_image(output,str( out_file))
        pbar.set_description(str(out_file))

def create_depth_coco(root_dir, model_type =  "DPT_Large"):
    data_transform = None
    root_dir = Path(root_dir)
    train_dir = root_dir / "images" / "train2017"
    val_dir = root_dir  / "images" / "val2017"

    train_dir_depth = root_dir / "depth" / "train2017"
    val_dir_depth = root_dir / "depth" / "val2017"
    try:
        os.mkdir(root_dir / "depth")
    except:
        pass
    try:
        os.mkdir(train_dir_depth)
    except:
        pass
    try:
        os.mkdir(val_dir_depth)
    except:
        pass
    print("Processing train images")

    process_images(train_dir, train_dir_depth, model_type)
    process_images(val_dir,val_dir_depth, model_type)
# Example usage

def arg_parse():
    parser = argparse.ArgumentParser(description='Depth Estimation')
    parser.add_argument("--root_dir", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--dataset", type=str, default="cocostuff", help="Dataset to use")
    parser.add_argument("--model_type", type=str, default="DPT_Large", help="Model type")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    if args.dataset == "cocostuff":
        create_depth_coco(args.root_dir, args.model_type)
    else:
        print("Dataset not supported")
