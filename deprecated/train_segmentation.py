# FILEPATH: train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import segmentation
from torchvision.transforms import v2
from kp2dtiny.models.kp2d_former import KeypointFormer
from datasets.coco import COCOLoader
import segmentation_models_pytorch as smp
from kp2dtiny.models.keypoint_net_vlad import KeypointNet, VGG16_DEFAULT
from kp2dtiny.models.tiny_keypoint_net import KeypointNetRaw, KP2D_TINY
from kp2dtiny.models.vgg16 import VGG16Net
from utils.utils import load_checkpoint, save_checkpoint, set_seed, load_json
import argparse
import numpy as np
from tqdm import tqdm
from evaluation.evaluate_segmentation import evaluate_segmentation
import wandb
import os
from utils.losses import jaccard_distance_loss

new_class_mapping = load_json("./data/cocostuff_mapping.json")
new_class_indices = torch.full((256,), 0, dtype=torch.uint8)

for original_class, new_class_index in new_class_mapping.items():
    # Subtract 1 from the original class because class indices are 1-based in your dataset
    new_class_indices[int(original_class)] = new_class_index

def map_classes(mask, mapping):
    # Flatten the mask to 1D for mapping
    flat_mask = mask.flatten()

    # Use torch.take to apply the mapping
    mapped_flat_mask = torch.take(mapping.long(), flat_mask.long())

    # Reshape back to the original mask shape
    new_mask = mapped_flat_mask.view_as(mask)
    return new_mask
class_transform =  v2.Lambda(lambda mask:  map_classes(mask, new_class_indices))

def parse_args():
    parser = argparse.ArgumentParser(description='Train Scene Parse 150')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--out_model_path', type=str, default='model.ckpt', help='Path to save model checkpoint')
    parser.add_argument('--dataset_config', type=str, default="datasets.json", help='Path to dataset config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to use for dataloading')
    parser.add_argument('--seed', type=int, default=42069, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
    parser.add_argument('--model_type', type=str, default="KeypointNet", help='Type of keypoint net')
    parser.add_argument('--im_h', type=int, default=256, help='Image height')
    parser.add_argument('--im_w', type=int, default=256, help='Image width')
    parser.add_argument('--wandb_project', type=str, default="Masterthesis-test", help='Wandb project name')
    parser.add_argument('--hard_triplet_loss', action='store_true', help='Use hard triplet loss')
    parser.add_argument('--tiny_v2', action='store_true', help='Use tiny v2 which has a different segmentation head')
    parser.add_argument('--large_netvlad', action='store_true', help='larger netvlad head')
    parser.add_argument('--dataset_name', type=str, default="scene_parse", help='Dataset name')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--use_attention', action='store_true', help='Only applies to kp2dtiny which can use attention when tiny_v2 is set')
    return parser.parse_args()

# Define your custom semantic segmentation model
args = parse_args()

dice_weight = 0.5
focal_weight = 0.2
cross_entropy_weight = 0.5
lovasz_weight = 0.1

weights = {"dice": dice_weight, "focal": focal_weight, "cross_entropy": cross_entropy_weight, "lovasz": lovasz_weight}

n_classes = args.n_classes
dataset_config = load_json(args.dataset_config)

set_seed(args.seed)
# Set up your training data and dataloader

#norm = v2.Lambda(lambda img: img.div(127.5)-1.)


    
size = (args.im_h,args.im_w)
# Initialize your model
if args.model_type == "KeypointNet":
    model = KeypointNet(nClasses=n_classes)
    d_f = 4
elif args.model_type == "KP2Dtiny":
    model = KeypointNetRaw(**KP2D_TINY, v2_seg=args.tiny_v2, nClasses=args.n_classes, use_attention=args.use_attention)
    d_f = 2
elif args.model_type == "vgg16":
    model = KeypointNet(**VGG16_DEFAULT, nClasses=n_classes)
    d_f = 4
elif args.model_type == "KeypointFormer":
    model = KeypointFormer( num_classes=n_classes)
    d_f = 4

model = model.to(args.device)
config = {"input_args": vars(args),
            "dataset_config": dataset_config,
            "size": size,
            "script": os.path.basename(__file__),
            "model_info": model.gather_info(),
            "weights": weights}


unsqueeze = v2.Lambda(lambda img: img.unsqueeze(0))
transform_pre = v2.Compose(
    [
        v2.ToTensor(),
        v2.Resize([args.im_h, args.im_w]),
        v2.RandomGrayscale(0.2),
        v2.RandomEqualize(0.2),
        unsqueeze
    ]

)
transform_pre_seg =     [
        v2.Resize([args.im_h//d_f, args.im_w//d_f], interpolation=v2.InterpolationMode.NEAREST),
        unsqueeze
    ]
if args.n_classes == 28:
    transform_pre_seg.append(class_transform)
transform_pre_seg = v2.Compose(transform_pre_seg)

transforms_post = v2.Compose([
    v2.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    v2.GaussianBlur(3, sigma=(0.1, 1.0)),

])
def transforms(examples):
    out = {}
    image   = examples["image"]
    seg_mask =  examples["annotation"]

    #image = image.convert('RGB')
    image_t = transform_pre(image).float()
    seg_mask_t = transform_pre_seg(torch.tensor(np.array(seg_mask)).unsqueeze(0)).float()


    image_t = transforms_post(image_t).mul(2.).sub(1.)

    out["image"] = image_t.squeeze(0)

    out["seg"] = seg_mask_t.squeeze(0)

    return out

dataset = COCOLoader(dataset_config["coco_data_path"], data_transform=transforms)
dataset_val = COCOLoader(dataset_config["coco_data_path"], split='val', data_transform=transforms)
  

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,drop_last=True)
    
# Define your loss function
criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(args.device)
dice_loss = smp.losses.dice.DiceLoss(mode="multiclass", ignore_index=255).to(args.device)
focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=255).to(args.device)
lovasz_loss = smp.losses.LovaszLoss(mode="multiclass", ignore_index=255).to(args.device)

# Define your optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

wandb.init(config=config, project=args.wandb_project, entity="thomacdebabo")
wandb.watch(model, log_freq=1000)

log_freq_loss = len(dataloader) // 10
# Training loop
for epoch in range(args.n_epochs):
    model.train()
    model.training = True
    pbar = tqdm(enumerate(dataloader, 0),
                    unit=' images',
                    unit_scale=args.batch_size,
                    total=len(dataloader),
                    smoothing=0,
                    disable=False)
    for (i, samples) in pbar:
        # Forward pass
        image   = samples["image"].to(args.device)
        seg_mask = samples['seg'].clone().long().to(args.device)
        
        _,_,_,_,seg_pred = model(image)
        
        loss = criterion(seg_pred, seg_mask.squeeze())*cross_entropy_weight
        d_loss = dice_loss(seg_pred, seg_mask.squeeze())*dice_weight
        lov_loss = lovasz_loss(seg_pred, seg_mask.squeeze())*lovasz_weight
        foc_loss = focal_loss(seg_pred, seg_mask.squeeze())*focal_weight
        #foc_loss = 0
        
        
        total_loss =  d_loss + lov_loss + loss + foc_loss
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        pbar.set_description('Train [ E {} CE {:.4f}Lovasz {:.4f} Dice {:.4f} Focal{:.4f} Total {:.4f}]'.format(epoch, float(loss),float(lov_loss), float(d_loss), float(foc_loss), float(total_loss)))
        if i % log_freq_loss == 0:
            wandb.log({"loss/":{"segmentation": total_loss, "dice": d_loss, "lovasz_loss": lov_loss, "focal_loss": foc_loss, "cross_entropy": loss}})
        # Print training progress
    segmentation_results = evaluate_segmentation(model, dataloader_val, n_classes)
    wandb.log({"val/": {"segmentation": segmentation_results}})
    print(segmentation_results)
    # Save the trained model
    torch.save(model.state_dict(), args.out_model_path)
