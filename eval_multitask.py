import copy
from pathlib import Path
from datetime import datetime
import os
import argparse

import torch
import wandb

from datasets.pittsburgh import get_whole_val_set
from datasets.scene_parse_150 import get_dataset
#from data.nyuv2 import get_dataset_nyuv2
from datasets.patches_dataset import get_patches_dataset
from datasets.cityscapes import CityScapeLoader, get_cityscapes_transforms
from datasets.coco import COCOLoader, get_coco_transforms
from datasets.nyuv2 import NYUv2Dataset_extracted, get_nyuv2_transforms

from src.evaluation.evaluate_keypoints import evaluate_keypoint_net
from src.evaluation.evaluate_segmentation import evaluate_segmentation
from src.evaluation.global_descriptor_evaluation import evaluate_global_descriptor
from src.evaluation.visual_odometry_evaluation import evaluate_visual_odometry
from src.evaluation.evaluate_depth_estimation import evaluate_depth_estimation

from src.kp2dtiny.models.kp2dtiny import KP2DTinyV2, get_config, KP2DTinyV3

from quantize import quantize
from utils.utils import load_checkpoint, save_checkpoint, set_seed, load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multitask model')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--dataset_config', type=str, default="datasets.json", help='Path to dataset config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to use for dataloading')
    parser.add_argument('--seed', type=int, default=42069, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=28, help='Number of classes')
    parser.add_argument('--model_type', type=str, default="KeypointNet", help='Type of keypoint net')
    parser.add_argument('--wandb_project', type=str, default="MT-Evaluation-Seg", help='Wandb project name')
    parser.add_argument('--dataset_name', type=str, default="cocostuff", help='Dataset name')
    parser.add_argument('--config', type=str, default="S", help='Model config [S, F, A, STM]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--quantized', action='store_true', help='Use quantized model')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--keypoints', action='store_true', help='Evaluate keypoints')
    parser.add_argument('--visloc', action='store_true', help='Evaluate visual localization')
    parser.add_argument('--segmentation', action='store_true', help='Evaluate segmentation')
    parser.add_argument('--depth', action='store_true', help='Evaluate depth estimation')
    parser.add_argument('--load_depth', action='store_true', help='Load depth model')
    parser.add_argument('--vo', action='store_true', help='Evaluate visual odometry')
    parser.add_argument('--backend', type=str, default="x86", help='Quantization backend')
    parser.add_argument('--v3', action='store_true', help='Use KP2DTinyV3')
    parser.add_argument('--result_dir', type=str, default="results", help='Result directory')

    return parser.parse_args()

def evaluate_local_keypoint_detection(model, dataloader,size,debug = False, top_k_eval=300):
    r, loc, c1, c3, c5, m, auc = evaluate_keypoint_net(dataloader, model, output_shape=size, top_k=top_k_eval, debug=debug)
    print("Repeatability: {} Loc: {} C1: {} C3: {} C5: {} MScore: {}".format(r, loc, c1, c3, c5, m))
    print(auc)

    return {"repeatability": r, "localization": loc, "c1": c1, "c3": c3, "c5": c5, "mscore": m, "auc": auc}

global_desc_dim = {"S": 8192, "F": 16384, "A": 8192, "STM": 8192}

def main(args):
    resolutions = [(240, 320)]#, (360, 640)]
    TOP_K = [300, 1000]
    set_seed(args.seed)
    info = {}

    model_path = Path(args.model_path)
    name = model_path.stem
    state_dict, optimizer_state, history = load_checkpoint(args.model_path, optimizer_key="optimizer")
    epoch = history.get("epoch", None)
    #config = history.get("config", None)
    config = None
    info["epoch"] = epoch
    info["quantized"] = args.quantized
    dataset_config = load_json(args.dataset_config)

    conf = get_config(args.config, v3=args.v3)
    dataset_name = args.dataset_name

    info["config"] = conf
    info["dataset"] = dataset_name

    d_f = 2 ** (conf['downsample'] - 1)
    cell = 2 ** (conf['downsample'] )
    all_results = {}
    if config is None:
        if args.v3:
            model = KP2DTinyV3(**conf, nClasses=args.n_classes, depth=args.load_depth)
        else:
            model = KP2DTinyV2(**conf, nClasses=args.n_classes, depth=args.load_depth)
    else:
        if conf['version']=='V3':
            model = KP2DTinyV3(**conf)
        else:
            model = KP2DTinyV2(**conf)



    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print("Error loading model state dict")
        print(e)
        print("Trying to load model state dict with strict=False")
        model.load_state_dict(state_dict, strict=False)



    if args.quantized:
        q_ds = COCOLoader(dataset_config["coco_data_path"],
                                 data_transform=get_coco_transforms(120, 160, d_f=d_f, n_classes=args.n_classes,
                                                                    val=True, load_depth=args.depth), split='val',
                                 depth=args.depth)
        q_dl = torch.utils.data.DataLoader(q_ds, batch_size=args.batch_size, shuffle=False,
                                                     drop_last=True, num_workers=args.num_workers)
        # copy model
        temp_model = copy.deepcopy(model)
        model = quantize(temp_model, q_dl, backend=args.backend)
        #model = KP2DTinyV2_Quantized(model=temp_model, device=args.device, cell=cell,
         #                            global_desc_dim=model.get_global_desc_dim())
    model.eval()
    model.training = False
    model.to(args.device)
    model.device = args.device



    for size in resolutions:
        print("Evaluating model on size: ", size)
        # Datasets
        if dataset_name == "scene_parse":
            dataset_val = get_dataset(dataset_config["scene_parse_data_path"], size, device=args.device, split="validation", n_classes=args.n_classes)
        elif dataset_name == "cocostuff":
            dataset_val = COCOLoader(dataset_config["coco_data_path"], data_transform=get_coco_transforms(size[0],size[1],d_f=d_f, n_classes=args.n_classes, val=True, load_depth=args.depth), split='val', depth=args.depth)
        elif dataset_name == "cityscapes":
            dataset_val = CityScapeLoader(dataset_config['cityscapes_data_path'], data_transform=get_cityscapes_transforms(size[0],size[1],d_f=d_f, val=True), split='val')
        elif dataset_name == "nyuv2":
            dataset_val = NYUv2Dataset_extracted(dataset_config['nyuv2_data_path'],data_transform=get_nyuv2_transforms(size[0],size[1],d_f=d_f, val=True),split='test')
        else:
            raise NotImplementedError("Dataset not implemented")

        pittsburgh_dataset = get_whole_val_set(dataset_config["pittsburgh_data_path"], size)

        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=args.num_workers)
        _, patches_dataloader = get_patches_dataset(dataset_config["hpatches_data_path"], size, augmentation_mode="default", n_workers=args.num_workers)


        results = {}
        if args.keypoints:
            for k in TOP_K:
                try:
                    keypoint_results = evaluate_local_keypoint_detection(model, patches_dataloader, size, debug=args.debug, top_k_eval=k)
                    results["keypoints_top" + str(k)] = keypoint_results.copy()
                    print(keypoint_results)
                except:
                    print("Error in keypoint evaluation")

        if args.visloc and size[0]<400:
            try:
                visloc_results = evaluate_global_descriptor(model, pittsburgh_dataset, device=args.device,
                                                            num_workers=args.num_workers)
                results["visloc"] = visloc_results
                print(visloc_results)
            except Exception as e:
                print("Error in visual localization evaluation", e)

        if args.segmentation and size[0]:
            try:
                segmentation_results = evaluate_segmentation(model, dataloader_val, n_classes=args.n_classes, debug=args.debug)
                results["segmentation"] = segmentation_results
                print(segmentation_results)
            except:
                print("Error in segmentation evaluation")

        if args.depth and size[0]<400:
            try:
                depth_results = evaluate_depth_estimation(model, dataloader_val, debug=args.debug)
                results["depth"] = depth_results
                print(depth_results)
            except:
                print("Error in depth evaluation")
        print(results)
        resolution_key = str(size[0]) + "x" + str(size[1])
        all_results[resolution_key] = results
    if args.vo:
        for size in [(128,256),(128,512),(256,1024)]:
            try:
                print("Evaluating visual odometry on size: ", size)
                vo_results = evaluate_visual_odometry(model, dataset_config["kitti_path"],
                                                         dataset_config["kitti_gt_path"],
                                                         dataset_config["kitti_video_path"],
                                                         args.device,
                                                         size, plot=args.debug)
                resolution_key ="visual_odometry_"+ str(size[0]) + "x" + str(size[1])
                all_results[resolution_key] = vo_results
                print(vo_results)
            except:
                print("Error in visual odometry evaluation")

    config = {"input_args": vars(args),
              "dataset_config": dataset_config,
              "size": resolutions,
              "script": os.path.basename(__file__),
              "model_info": model.gather_info()}

    if args.wandb:
        wandb.init(config=config, project=args.wandb_project, entity="thomacdebabo")


    if args.wandb:
        wandb.log({"val/": all_results})
    final_dict = {"results": all_results, "info": info}
    result_path = Path(args.result_dir) / (name + date_file_name())
    os.mkdir(result_path)
    save_json(final_dict, result_path / ("results_"+ name +".json"))
if __name__ == "__main__":

    args = parse_args()
    main(args)


