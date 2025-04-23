from src.kp2dtiny.models.kp2dtiny import KP2DTinyV2, KP2DTinyV3, get_config
from src.kp2dtiny.models.kp2d_former import KeypointFormer, KEYPOINTFORMER_TINY_CONFIG, KEYPOINTFORMER_DEFAULT_CONFIG
import torch
from pathlib import Path

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Export ONNX model')
    parser.add_argument('--config', type=str, default='S', help='Model config')
    parser.add_argument('--im_h', type=int, default=120, help='Image height')
    parser.add_argument('--im_w', type=int, default=160, help='Image width')
    parser.add_argument('--n_classes', type=int, default=28, help='Number of classes')
    parser.add_argument('--model_type', type=str, default='KP2Dtiny', help='Model type')
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='Model path')
    parser.add_argument('--weight_path', default= None,type=str, help='Weight path')
    parser.add_argument('--to_mcu', default=True, help='Export for MCU')    
    parser.add_argument('--to_export', default=True, help='Removes netvlad layer for export to MCU')
    parser.add_argument('--depth', action='store_true', help='Use depth')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    name = args.model_type
    
    to_mcu = args.to_mcu
    to_export = args.to_export
        
    # if to_mcu:
    #     name += "_mcu"
    # if to_export:
    #     name += "_export"
        
    out_path = Path(args.model_path)
    if args.model_type == "KP2Dtiny":
        model = KP2DTinyV2(**get_config(args.config, to_mcu=to_mcu, to_export=to_export), nClasses=args.n_classes)
        name += "_" + args.config
    elif args.model_type == "KP2DtinyV3":
        model = KP2DTinyV3(**get_config(args.config, v3=True, to_mcu=to_mcu, to_export=to_export), nClasses=args.n_classes, depth=args.depth)
        name += "_" + args.config
    elif args.model_type == "KeypointFormer":
        model = KeypointFormer(**KEYPOINTFORMER_DEFAULT_CONFIG, nClasses=args.n_classes)
    else:
        raise ValueError("Model type not supported")
    
    if args.weight_path is not None:
        model.load_state_dict(torch.load(args.weight_path, map_location=torch.device('cpu'))['state_dict'])
    torch_input = torch.randn(1, 3, args.im_h, args.im_w)

    torch.onnx.export(model, torch_input, out_path / (name + ".onnx"), 
                      verbose=False, opset_version=16, input_names=["image"], 
                      output_names=["score", "coord", "desc", "vlad","seg"], do_constant_folding=False)

    print(f"Model exported to {out_path / (name + '.onnx')}")