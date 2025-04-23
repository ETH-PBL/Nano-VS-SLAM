from PIL import Image
import torch
import os
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import argparse
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_segformer():
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

    class SegformerForDepthEstimation(nn.Module):
        def __init__(self, model, feature_extractor):
            super().__init__()
            self.model = model
            self.feature_extractor = feature_extractor
            self.softmax = nn.Softmax2d()

        def forward(self, image):
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

            return self.softmax(logits).argmax(dim=1)

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
    model.eval()

    return SegformerForDepthEstimation(model, feature_extractor)

def get_image(filename):
    img = Image.open(filename)
    print(img.size)
    return img

def save_image(output, filename):
    output_image = output
    output_image = output_image.astype('uint8')[0]
    Image.fromarray(output_image).save(filename)

def process_images(directory, out_directory, model_type):
    segformer = load_segformer()
    pbar = tqdm(directory.glob("*.jpg"))
    for filepath in  pbar:
        filename =  Path(filepath).stem + ".png"
        out_file = out_directory / f"depth_{filename}"
        if out_file.exists():
            continue
        input_batch = get_image(str(filepath))
        print(input_batch.size)
        with torch.no_grad():
            prediction = segformer(input_batch)
        print(prediction.shape)
        output = prediction.cpu().numpy()
        save_image(output,str( out_file))
        pbar.set_description(str(out_file))

def create_depth_coco(root_dir, model_type =  "DPT_Large"):
    data_transform = None
    root_dir = Path(root_dir)
    train_dir = root_dir / "images" / "train2017"
    val_dir = root_dir  / "images" / "val2017"

    train_dir_depth = root_dir / "seg" / "train2017"
    val_dir_depth = root_dir / "seg" / "val2017"
    try:
        os.mkdir(root_dir / "seg")
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
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument("--root_dir", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--dataset", type=str, default="folder", help="Dataset to use")
    parser.add_argument("--model_type", type=str, default="DPT_Large", help="Model type")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    if args.dataset == "cocostuff":
        create_depth_coco(args.root_dir, args.model_type)
    elif args.dataset == "folder":
        out_path = Path(args.root_dir) / "segmentation"
        out_path.mkdir(exist_ok=True)
        process_images(Path(args.root_dir), out_path, args.model_type)
    else:
        print("Dataset not supported")

