import torch
from utils.utils import load_checkpoint, save_checkpoint, set_seed, load_json
from datasets.utils import get_transforms
from datasets.coco import COCOLoader
from datasets.scene_parse_150 import get_dataset
from kp2dtiny.models.kp2dtiny import KP2DTinyV2, get_config
from quantize import quantize, save
import os
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Export quantized model')
    parser.add_argument('--dataset_name', type=str, default='cocostuff', help='Dataset name')
    parser.add_argument('--im_h', type=int, default=120, help='Image height')
    parser.add_argument('--im_w', type=int, default=160, help='Image width')
    parser.add_argument('--n_classes', type=int, default=28, help='Number of classes')
    parser.add_argument('--dataset_config', type=str, default='datasets.json', help='Path to dataset config file')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='S', help='Model config [S, F, A]')
    parser.add_argument('--backend', type=str, default='onednn', help='Quantization backend')
    parser.add_argument('--onnx', action='store_true', help='Export to onnx')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    conf = get_config(args.config)
    dataset_config = load_json(args.dataset_config)
    size = (args.im_h, args.im_w)
    d_f = 2 ** (conf['downsample'] - 1)
    cell = 2 ** (conf['downsample'] )

    if args.dataset_name == "scene_parse":
        dataset_val = get_dataset(dataset_config["scene_parse_data_path"], size, device="cpu", split="validation", n_classes=args.n_classes)
    elif args.dataset_name == "cocostuff":
        dataset_val = COCOLoader(dataset_config["coco_data_path"], data_transform=get_transforms(args.im_h ,args.im_w ,d_f=d_f, n_classes=args.n_classes), split='val')
    else:
        raise NotImplementedError("Dataset not implemented")

    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, drop_last=True,
                                                 num_workers=0)

    temp_model = KP2DTinyV2(**conf, nClasses=args.n_classes)
    state_dict, _, _ = load_checkpoint(args.model_path, optimizer_key="optimizer")
    temp_model.load_state_dict(state_dict, strict=False)
    model = quantize(temp_model, dataloader_val, backend=args.backend)
    # append q_ at the beginning of the name and replace .ckpt with .pth keep in mind that the path is absolute

    if args.onnx:
        out_path = os.path.join(os.path.dirname(args.model_path),
                                f"q_{os.path.basename(args.model_path).replace('.ckpt', '.onnx')}")

        torch.onnx.export(model, torch.randn(1,3,args.im_h, args.im_w), out_path , verbose=True, opset_version=17, input_names=["image"], output_names=["score", "coord", "desc", "vlad","seg"])
    else:
        out_path = os.path.join(os.path.dirname(args.model_path),
                                f"q_{os.path.basename(args.model_path).replace('.ckpt', '.pth')}")

        save(model, out_path)
    print("Quantized model saved to {}".format(out_path))