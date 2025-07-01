import argparse
import os

import torch
from tqdm import tqdm
import wandb

from src.kp2dtiny.models.KeypointNetwithIOLoss import KeypointNetwithIOLoss
from src.data.patches_dataset import get_patches_dataset
from src.data.pittsburgh import get_whole_val_set
from evaluation.keypoints import evaluate_keypoint_net
from evaluation.segmentation import evaluate_segmentation
from evaluation.global_descriptor import evaluate_global_descriptor
from evaluation.visual_odometry import evaluate_visual_odometry
from evaluation.depth_estimation import evaluate_depth_estimation
from utils.utils import load_checkpoint, save_checkpoint, set_seed, load_json
from src.data.scene_parse_150 import get_dataset
from src.data.cityscapes import CityScapeLoader, get_cityscapes_transforms
from src.data.coco import COCOLoader, get_coco_transforms

TRAIN_CONFIG = {
    "device": "cuda",
    "lr": 0.0005,
    "batch_size": 4,
    "model_path": None,
    "out_model_path": "model.ckpt",
    "dataset_config": "datasets.json",
    "debug": False,
    "num_workers": 0,
    "seed": 42069,
    "n_classes": 28,
    "freeze_backbone": False,
    "model_type": "KP2DtinyV2",
    "im_h": 120,
    "im_w": 160,
    "wandb_project": "Masterthesis-test",
    "hard_triplet_loss": True,
    "dataset_name": "cocostuff",
    "n_epochs": 100,
    "initial_evaluation": False,
    "config": "A",
    "qat": False,
    "start_qat_epoch": 0,
    "wandb": False,
    "disable_multi": False,
    "optimizer": "adam",
    "lr_scheduler": "cosine",
    "disable_strict_load": False,
    "to_mcu": False,
}

LOSS_WEIGHTS = {
    "keypoint_loss": 0.5,
    "loc_loss": 1.0,
    "io_loss": 1.0,
    "score_loss": 1.0,
    "descriptor_loss": 2.0,
    "segmentation_loss": 2.0,
    "vlad_loss": 1.0,
    "depth_loss": 0.5,
    "huber_loss": 1.0,
}

LOSS_WEIGHTS_SCHEDULE = {
    5: {
        "keypoint_loss": 1.4,
        "score_loss": 1.4,
        "descriptor_loss": 2.0,
        "segmentation_loss": 0.5,
        "vlad_loss": 1.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    }
}

LOSS_WEIGHTS_SCHEDULE_REFINED = {
    0: {
        "keypoint_loss": 2.0,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.0,
        "descriptor_loss": 2.0,
        "segmentation_loss": 5.0,
        "vlad_loss": 1.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    3: {
        "keypoint_loss": 0.1,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.0,
        "descriptor_loss": 2.0,
        "segmentation_loss": 4.0,
        "vlad_loss": 0.1,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    50: {
        "keypoint_loss": 0.2,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.0,
        "descriptor_loss": 2.0,
        "segmentation_loss": 3.0,
        "vlad_loss": 0.3,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    75: {
        "keypoint_loss": 0.5,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.5,
        "descriptor_loss": 2.0,
        "segmentation_loss": 2.0,
        "vlad_loss": 1.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    90: {
        "keypoint_loss": 0.7,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.5,
        "descriptor_loss": 2.0,
        "segmentation_loss": 1.5,
        "vlad_loss": 2.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    95: {
        "keypoint_loss": 0.3,
        "loc_loss": 1.0,
        "io_loss": 1.0,
        "score_loss": 1.5,
        "descriptor_loss": 2.0,
        "segmentation_loss": 1.5,
        "vlad_loss": 1.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
}

LOSS_WEIGHTS_SCHEDULE_D = {
    10: {
        "keypoint_loss": 1.4,
        "score_loss": 1.4,
        "descriptor_loss": 2.0,
        "segmentation_loss": 0.5,
        "vlad_loss": 3.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    25: {
        "keypoint_loss": 1.0,
        "score_loss": 1.0,
        "descriptor_loss": 2.5,
        "segmentation_loss": 2.0,
        "vlad_loss": 2.0,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
    30: {
        "keypoint_loss": 1.2,
        "score_loss": 1.2,
        "descriptor_loss": 2.0,
        "segmentation_loss": 1.0,
        "vlad_loss": 1.5,
        "depth_loss": 0.5,
        "huber_loss": 1.0,
    },
}

TRAIN_FLAGS = {"segmentation": True, "keypoints": True, "visloc": True, "depth": False}

TRAIN_SEGMENTATION = {
    "segmentation": True,
    "keypoints": False,
    "visloc": False,
    "depth": False,
}

TRAIN_KEYPOINTS = {
    "segmentation": False,
    "keypoints": True,
    "visloc": False,
    "depth": False,
}

COCOSTUFF_CONFIG = {
    "lr": 0.0005,
    "n_classes": 28,
    "im_h": 120,
    "im_w": 160,
    "n_epochs": 20,
    "initial_evaluation": False,
    "optimizer": "adam",
    "lr_scheduler": "cosine",
    "freeze_backbone": False,
}

CITYSCAPES_CONFIG = {
    "lr": 0.001,
    "n_classes": 19,
    "im_h": 120,
    "im_w": 160,
    "n_epochs": 20,
    "initial_evaluation": True,
    "optimizer": "adam",
    "lr_scheduler": "cosine",
    "freeze_backbone": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Scene Parse 150")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--out_model_path",
        type=str,
        default="model.ckpt",
        help="Path to save model checkpoint",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="datasets.json",
        help="Path to dataset config file",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for dataloading",
    )
    parser.add_argument("--seed", type=int, default=42069, help="Random seed")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DD",
        help="Type of model [KeypointFormer, DD or DF]",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="NanoVSSLAM", help="Wandb project name"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cocostuff", help="Dataset name"
    )
    parser.add_argument("--config", type=str, default="S_A", help="Model config")
    parser.add_argument("--wandb", action="store_true", help="Use wandb")
    parser.add_argument(
        "--disable_strict_load", action="store_true", help="Disable strict load"
    )
    parser.add_argument("--to_mcu", action="store_true", help="Train for MCU")
    parser.add_argument(
        "--ignore_seg_head",
        action="store_true",
        help="Do not load segmentation head from checkpoint",
    )
    parser.add_argument("--freeze_seg", action="store_true", help="Freeze seg head")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (experimental)",
    )
    parser.add_argument(
        "--full_eval", default=3, type=int, help="Full evaluation every n epochs"
    )
    parser.add_argument("--top_k", default=300, type=int, help="Top k for evaluation")
    parser.add_argument("--depth", action="store_true", help="Load depth")
    parser.add_argument(
        "--loss_weight_schedule", action="store_true", help="Use loss weight schedule"
    )
    parser.add_argument(
        "--only_segmentation", action="store_true", help="Only train segmentation"
    )
    parser.add_argument(
        "--only_keypoints", action="store_true", help="Only train keypoints"
    )
    parser.add_argument("--no_vpr", action="store_true", help="No VPR training")
    parser.add_argument("--start_epoch", default=0, type=int, help="Start epoch")
    return parser.parse_args()


def evaluate_local_keypoint_detection(model, dataloader, size):
    r, loc, c1, c3, c5, m, auc = evaluate_keypoint_net(
        dataloader, model, output_shape=size, top_k=300
    )
    print(f"Repeatability: {r} Loc: {loc} C1: {c1} C3: {c3} C5: {c5} MScore: {m}")
    return {
        "repeatability": r,
        "localization": loc,
        "c1": c1,
        "c3": c3,
        "c5": c5,
        "mscore": m,
        "auc": auc,
    }


def filter_statedict(state_dict, mode=None):
    if mode is None:
        return state_dict
    if mode == "seg":
        return {k: v for k, v in state_dict.items() if "seg_head" not in k}
    elif mode == "vlad":
        return {k: v for k, v in state_dict.items() if "vlad_head" not in k}
    elif mode == "seg_last":
        try:
            state_dict.pop("seg_head.convs.8.weight")
            state_dict.pop("seg_head.convs.8.bias")
        except KeyError:
            state_dict.pop("seg_head.convs.7.weight")
            state_dict.pop("seg_head.convs.7.bias")
        return state_dict
    else:
        raise NotImplementedError("Mode not implemented")


def main(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise ValueError("CUDA not available")
    # This is a bit of a mess, TODO: refactor
    loss_weights = LOSS_WEIGHTS.copy()
    if args.dataset_name == "cocostuff":
        train_config = COCOSTUFF_CONFIG.copy()
        train_flags = TRAIN_FLAGS.copy()
    elif args.dataset_name == "cityscapes":
        train_config = CITYSCAPES_CONFIG.copy()
        train_flags = TRAIN_FLAGS.copy()
    else:
        raise NotImplementedError("Dataset not implemented")

    if args.only_segmentation:
        train_flags = TRAIN_SEGMENTATION.copy()
    elif args.only_keypoints:
        train_flags = TRAIN_KEYPOINTS.copy()

    if args.no_vpr:
        train_flags["visloc"] = False
        loss_weights["vlad_loss"] = 0.0
        print("Disabling VPR training")

    size = (train_config["im_h"], train_config["im_w"])
    if args.dataset_name == "cityscapes":
        train_flags["depth"] = False
        print("Disabling depth training for cityscapes")
    ######################
    set_seed(args.seed)
    dataset_config = load_json(args.dataset_config)
    model = KeypointNetwithIOLoss(
        loss_weights=loss_weights,
        debug=args.debug,
        device=args.device,
        n_classes=train_config["n_classes"],
        keypoint_net_type=args.model_type,
        config=args.config,
        to_mcu=args.to_mcu,
        load_depth=args.depth,
        top_k=args.top_k,
    )

    optimizer = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }.get(train_config["optimizer"], None)

    if optimizer is None:
        raise NotImplementedError("Optimizer not implemented")

    optimizer = optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), lr=train_config["lr"]
    )

    scheduler = {
        "step": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        ),
        "cosine": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, eta_min=0
        ),
        "none": None,
    }.get(train_config["lr_scheduler"], None)

    if scheduler is None and train_config["lr_scheduler"] != "none":
        raise NotImplementedError("Scheduler not implemented")

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    config = {
        "input_args": vars(args),
        "dataset_config": dataset_config,
        "train_config": train_config,
        "size": size,
        "script": os.path.basename(__file__),
        "model_info": model.gather_info(),
        "loss_weights": loss_weights,
    }

    dataset, dataset_val = get_datasets(
        args, dataset_config, train_config, model, train_flags, size
    )
    pittsburgh_dataset = get_whole_val_set(dataset_config["pittsburgh_data_path"], size)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    _, patches_dataloader = get_patches_dataset(
        dataset_config["hpatches_data_path"],
        size,
        augmentation_mode="default",
        n_workers=args.num_workers,
    )

    history = None
    start_results = {}

    if args.model_path is not None:
        state_dict, optimizer_state, history = load_checkpoint(
            args.model_path, optimizer_key="optimizer"
        )
        mode = "seg_last" if args.ignore_seg_head else None
        state_dict = filter_statedict(state_dict, mode=mode)
        model.keypoint_net.load_state_dict(
            state_dict, strict=not args.disable_strict_load
        )
        print(f"Model loaded from {args.model_path}")
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Optimizer state not loaded: {e}")

    model.init_warper(
        size
    )  # initialize Homography warper in the model (for homography augmentation)
    # NOTE
    # .train() is the method call for pytorch
    # .training is so that the model does not sample the descriptors at the end of the forward pass
    # .set_train_flags is a method to set training flags taskwise
    model.train()
    model.training = True
    model.set_train_flags(train_flags)

    if train_config["freeze_backbone"]:
        model.keypoint_net.freeze_backbone()
    if args.freeze_seg:
        model.keypoint_net.freeze_segmentation(True)
    if args.wandb:
        initialize_wandb(args, config, model, dataset, dataset_val, train_flags)

    log_freq_loss = len(dataloader) // 10

    if train_config["initial_evaluation"]:
        start_results = initial_evaluation(
            args,
            model,
            dataset_config,
            dataloader_val,
            patches_dataloader,
            pittsburgh_dataset,
            train_config,
        )
        print(start_results)
        if args.wandb:
            wandb.log({"val/": start_results})

    lw_schedule = get_loss_weight_schedule(args)

    for epoch in range(args.start_epoch, train_config["n_epochs"]):
        model.training = True
        model.train()

        if epoch in lw_schedule:
            model.set_loss_weights(lw_schedule[epoch])
            print(f"Setting new loss weights: {lw_schedule[epoch]}")

        if train_config["freeze_backbone"]:
            model.keypoint_net.freeze_backbone()

        pbar = tqdm(
            enumerate(dataloader, 0),
            unit=" images",
            unit_scale=args.batch_size,
            total=len(dataloader),
            smoothing=0,
            disable=False,
        )
        iters = len(dataloader)
        for i, sample in pbar:
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.use_amp
            ):
                loss_2d, loss_dict, _ = model(sample)
            scaler.scale(loss_2d).backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step(float(epoch + i / iters))

            pbar.set_description(
                f"Train [ E {epoch} L {loss_2d:.4f}, S_L {loss_dict['seg_loss']:.4f}, V_L {loss_dict['vlad_loss']:.4f}]"
            )
            if i % log_freq_loss == 0 and args.wandb:
                wandb.log({"loss/": loss_dict})
                if scheduler is not None:
                    wandb.log({"scheduler/": {"lr": float(scheduler.get_last_lr()[0])}})

        eval_model = model.keypoint_net
        results = evaluate_model(
            args,
            eval_model,
            dataloader_val,
            patches_dataloader,
            pittsburgh_dataset,
            train_flags,
            train_config,
            epoch,
            dataset_config,
        )

        if args.wandb:
            wandb.log({"val/": results})
            wandb.log({"loss_weights": model.get_loss_weights()})
            if train_flags["depth"]:
                log_depth_examples(args, model, dataset, dataset_val)

        print(results)

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.keypoint_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "start_results": start_results,
            "current_results": results,
        }

        save_checkpoint(checkpoint, args.out_model_path)


def get_datasets(args, dataset_config, train_config, model, train_flags, size):
    if args.dataset_name == "scene_parse":
        dataset = get_dataset(
            dataset_config["scene_parse_data_path"],
            size,
            device=args.device,
            split="train",
            n_classes=train_config["n_classes"],
        )
        dataset_val = get_dataset(
            dataset_config["scene_parse_data_path"],
            size,
            device=args.device,
            split="validation",
            n_classes=train_config["n_classes"],
        )
    elif args.dataset_name == "cocostuff":
        dataset = COCOLoader(
            dataset_config["coco_data_path"],
            data_transform=get_coco_transforms(
                train_config["im_h"],
                train_config["im_w"],
                d_f=model.d_f,
                n_classes=train_config["n_classes"],
                load_depth=train_flags["depth"],
            ),
            split="train",
            depth=train_flags["depth"],
        )
        dataset_val = COCOLoader(
            dataset_config["coco_data_path"],
            data_transform=get_coco_transforms(
                train_config["im_h"],
                train_config["im_w"],
                d_f=model.d_f,
                n_classes=train_config["n_classes"],
                val=True,
                load_depth=train_flags["depth"],
            ),
            split="val",
            depth=train_flags["depth"],
        )
    elif args.dataset_name == "cityscapes":
        dataset = CityScapeLoader(
            dataset_config["cityscapes_data_path"],
            data_transform=get_cityscapes_transforms(
                train_config["im_h"], train_config["im_w"], d_f=model.d_f
            ),
            split="train",
        )
        dataset_val = CityScapeLoader(
            dataset_config["cityscapes_data_path"],
            data_transform=get_cityscapes_transforms(
                train_config["im_h"], train_config["im_w"], d_f=model.d_f, val=True
            ),
            split="val",
        )
    else:
        raise NotImplementedError("Dataset not implemented")
    return dataset, dataset_val


def initialize_wandb(args, config, model, dataset, dataset_val, train_flags):
    wandb.init(config=config, project=args.wandb_project, entity="thomacdebabo")
    wandb.watch(model.keypoint_net, log_freq=1000, log="all")
    if train_flags["depth"]:
        wandb_batch_val = torch.stack(
            [dataset_val.__getitem__(i)["image"] for i in range(4)]
        )
        wandb_batch_train = torch.stack(
            [dataset.__getitem__(i)["image"] for i in range(4)]
        )
        wandb_gt_val = torch.stack(
            [dataset_val.__getitem__(i)["depth"] for i in range(4)]
        )
        wandb_gt_train = torch.stack(
            [dataset.__getitem__(i)["depth"] for i in range(4)]
        )
        wandb_img_val = wandb.Image(wandb_gt_val, caption="depth")
        wandb_img_train = wandb.Image(wandb_gt_train, caption="depth")
        wandb.log({"val_gt": wandb_img_val, "train_gt": wandb_img_train})


def initial_evaluation(
    args,
    model,
    dataset_config,
    dataloader_val,
    patches_dataloader,
    pittsburgh_dataset,
    train_config,
):
    eval_model = model.keypoint_net
    eval_model.device = args.device
    eval_model.to(args.device)

    kitti_results = evaluate_visual_odometry(
        eval_model,
        dataset_config["kitti_path"],
        dataset_config["kitti_gt_path"],
        dataset_config["kitti_video_path"],
        args.device,
        (256, 1024),
        plot=args.debug,
    )
    keypoint_results = evaluate_local_keypoint_detection(
        eval_model, patches_dataloader, (train_config["im_h"], train_config["im_w"])
    )
    visloc_results = evaluate_global_descriptor(
        eval_model,
        pittsburgh_dataset,
        device=eval_model.device,
        num_workers=args.num_workers,
    )
    segmentation_results = evaluate_segmentation(
        eval_model, dataloader_val, n_classes=train_config["n_classes"]
    )

    return {
        "keypoints": keypoint_results,
        "visloc": visloc_results,
        "segmentation": segmentation_results,
        "vo": kitti_results,
    }


def get_loss_weight_schedule(args):
    if args.loss_weight_schedule:
        print("Using loss weight schedule")
        if args.config in ["D", "D_A"]:
            return LOSS_WEIGHTS_SCHEDULE_D
        else:
            return LOSS_WEIGHTS_SCHEDULE_REFINED
    return {}


def evaluate_model(
    args,
    eval_model,
    dataloader_val,
    patches_dataloader,
    pittsburgh_dataset,
    train_flags,
    train_config,
    epoch,
    dataset_config,
):
    results = {}
    if train_flags["segmentation"]:
        segmentation_results = evaluate_segmentation(
            eval_model, dataloader_val, n_classes=train_config["n_classes"]
        )
        results["segmentation"] = segmentation_results

    if train_flags["depth"]:
        depth_results = evaluate_depth_estimation(
            eval_model, dataloader_val, debug=args.debug
        )
        results["depth"] = depth_results

    if epoch % args.full_eval == 0:
        if train_flags["visloc"]:
            vlad_recalls = evaluate_global_descriptor(
                eval_model,
                pittsburgh_dataset,
                device=eval_model.device,
                num_workers=args.num_workers,
            )
            results["visloc"] = vlad_recalls
        if train_flags["keypoints"]:
            keypoint_results = evaluate_local_keypoint_detection(
                eval_model,
                patches_dataloader,
                (train_config["im_h"], train_config["im_w"]),
            )
            kitti_results = evaluate_visual_odometry(
                eval_model,
                dataset_config["kitti_path"],
                dataset_config["kitti_gt_path"],
                dataset_config["kitti_video_path"],
                args.device,
                (256, 1024),
                plot=args.debug,
            )
            results["keypoints"] = keypoint_results
            results["vo"] = kitti_results
    return results


def log_depth_examples(args, model, dataset, dataset_val):
    wandb_out = model.keypoint_net(
        torch.stack([dataset_val.__getitem__(i)["image"] for i in range(4)]).to(
            args.device
        )
    )
    depth_image = wandb_out["depth"].detach().cpu()
    wandb_img = wandb.Image(depth_image, caption="depth")
    wandb.log({"depth examples val": wandb_img})

    wandb_out = model.keypoint_net(
        torch.stack([dataset.__getitem__(i)["image"] for i in range(4)]).to(args.device)
    )
    depth_image = wandb_out["depth"].detach().cpu()
    wandb_img = wandb.Image(depth_image, caption="depth")
    wandb.log({"depth examples train": wandb_img})


if __name__ == "__main__":
    args = parse_args()
    main(args)
