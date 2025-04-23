# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
from matplotlib.cm import get_cmap
import torchgeometry as tgm
import segmentation_models_pytorch as smp

from .inlier_net import InlierNet
from ..utils.keypoints import draw_keypoints
import inspect
from .kp2d_former import KeypointFormer
from .kp2dtiny import KP2DTinyV2, get_config, KP2DTinyV3
from ..utils.losses import HardTripletLoss, SILogLoss, RMSE_log, GradLoss, NormalLoss, imgrad_yx, Grad

def build_descriptor_loss(source_des, target_des, source_points, tar_points, tar_points_un, keypoint_mask=None, relax_field=8,epsilon=1e-8, eval_only=False):
    """Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf..
    Parameters
    ----------
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points: torch.Tensor (B,H/8,W/8,2)
        Source image keypoints
    tar_points: torch.Tensor (B,H/8,W/8,2)
        Target image keypoints
    tar_points_un: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """

    batch_size, C, _, _ = source_des.shape
    loss, recall = 0., 0.
    margins = 0.2

    for cur_ind in range(batch_size):

        if keypoint_mask is None:
            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze().view(C, -1)
            tar_points_raw = tar_points_un[cur_ind].view(2, -1)
        else:
            keypoint_mask_ind = keypoint_mask[cur_ind].squeeze()

            n_feat = keypoint_mask_ind.sum().item()
            if n_feat < 20:
                continue

            ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_points_raw = tar_points_un[cur_ind][:, keypoint_mask_ind]

        # Compute dense descriptor distance matrix and find nearest neighbor
        ref_desc = ref_desc.div(torch.norm(ref_desc+epsilon, p=2, dim=0)+epsilon)
        tar_desc = tar_desc.div(torch.norm(tar_desc+epsilon, p=2, dim=0)+epsilon)
        dmat = torch.mm(ref_desc.t(), tar_desc)
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1)+epsilon)

        # Sort distance matrix
        dmat_sorted, idx = torch.sort(dmat, dim=1)

        # Compute triplet loss and recall
        candidates = idx.t() # Candidates, sorted by descriptor distance

        # Get corresponding keypoint positions for each candidate descriptor
        match_k_x = tar_points_raw[0, candidates]
        match_k_y = tar_points_raw[1, candidates]

        # True keypoint coordinates
        true_x = tar_points_raw[0]
        true_y = tar_points_raw[1]

        # Compute recall as the number of correct matches, i.e. the first match is the correct one
        correct_matches = (abs(match_k_x[0]-true_x) == 0) & (abs(match_k_y[0]-true_y) == 0)
        recall += float(1.0 / batch_size) * (float(correct_matches.float().sum()) / float( ref_desc.size(1)))

        if eval_only:
            continue

        # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
        correct_idx = (abs(match_k_x - true_x) <= relax_field) & (abs(match_k_y - true_y) <= relax_field)
        # Get hardest negative example as an incorrect match and with the smallest descriptor distance 
        incorrect_first = dmat_sorted.t()
        incorrect_first[correct_idx] = 2.0 # largest distance is at most 2
        incorrect_first = torch.argmin(incorrect_first, dim=0)
        incorrect_first_index = candidates.gather(0, incorrect_first.unsqueeze(0)).squeeze()

        anchor_var = ref_desc
        pos_var    = tar_desc
        neg_var    = tar_desc[:, incorrect_first_index]

        loss += float(1.0 / batch_size) * torch.nn.functional.triplet_margin_loss(anchor_var.t(), pos_var.t(), neg_var.t(), margin=margins)

    return loss, recall

class KeypointNetwithIOLoss(torch.nn.Module):
    """
    Model class encapsulating the KeypointNet and the IONet.

    Parameters
    ----------
    keypoint_loss_weight: float
        Keypoint loss weight.
    descriptor_loss_weight: float
        Descriptor loss weight.
    score_loss_weight: float
        Score loss weight.
    keypoint_net_learning_rate: float
        Keypoint net learning rate.
    with_io:
        Use IONet.
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    do_cross: bool
        Predict keypoints outside cell borders.
    with_drop : bool
        Use dropout.
    descriptor_loss: bool
        Use descriptor loss.
    kwargs : dict
        Extra parameters
    """
    def __init__(
        self, loss_weights,
        keypoint_net_learning_rate=0.001, keypoint_net_type='KeypointNet',device='cpu',
            debug = False, n_classes=8, config = "S", to_mcu=False,load_depth=False, top_k = 300, **kwargs):

        super().__init__()
        self.device = device
        self.debug = debug
        self.use_color = True
        self.keypoint_net_learning_rate = keypoint_net_learning_rate
        self.optim_params = []
        self.n_classes = n_classes
        self.cell = 8 # Size of each output cell. Keep this fixed.
        self.border_remove = 4 # Remove points this close to the border.
        self.top_k2 = top_k
        self.relax_field = 4

        # Loss weights
        self.keypoint_loss_weight = loss_weights['keypoint_loss']
        self.descriptor_loss_weight = loss_weights['descriptor_loss']
        self.score_loss_weight = loss_weights['score_loss']
        self.segmentation_loss_weight = loss_weights['segmentation_loss']
        self.vlad_loss_weight = loss_weights['vlad_loss']
        self.depth_loss_weight = loss_weights['depth_loss']
        self.io_loss_weight = loss_weights['io_loss']
        self.loc_loss_weight = loss_weights['loc_loss']

        # Losses
        self.descriptor_loss = True
        self.use_mse_loss = False
        self.with_io = True

        # Training flags
        self.train_keypoints = True
        self.train_segmentation = True
        self.train_visloc = True
        self.train_depth = True

        self.kp2dtiny_config = {}

        # Initialize Losses
        self.segmentation_criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)


        self.RMSE_log = RMSE_log().to(device)
        self.GradLoss = GradLoss().to(device)
        self.NormalLoss = NormalLoss().to(device)
        self.Grad = Grad().to(device)
        self.grad_factor = 0.0
        self.normal_factor = 0.0
        self.MSE_Loss = torch.nn.MSELoss()
        self.L1_Loss = torch.nn.L1Loss()
        self.SILog_Loss = SILogLoss()
        self.HuberLoss = torch.nn.HuberLoss()
        self.huber_loss_factor = 1.0
        self.dice_loss = smp.losses.dice.DiceLoss(mode="multiclass", ignore_index=255).to(device)
        self.vlad_criterion = HardTripletLoss(margin=0.5, hardest=True, squared=False).to(device)


        self.downsample = 3
        self.d_f = 2**(self.downsample-1)
        
        # Initialize KeypointNet

        if keypoint_net_type == 'KeypointFormer':
            self.keypoint_net = KeypointFormer(num_classes=self.n_classes, device=self.device)
        elif keypoint_net_type == 'KP2DtinyV2' or keypoint_net_type == 'DD':
            
            conf = get_config(config, to_mcu=to_mcu, v3=False)
            self.keypoint_net = KP2DTinyV2(**conf, nClasses=self.n_classes, device=self.device, depth=load_depth)
            self.downsample = conf['downsample']
            self.d_f = 2**(conf['downsample']-1)
            self.kp2dtiny_config = conf
            self.kp2dtiny_config['nClasses'] = self.n_classes
            self.kp2dtiny_config['device'] = self.device
            self.kp2dtiny_config['name'] = config
            self.kp2dtiny_config['version'] = "V2"

        elif keypoint_net_type == 'KP2DtinyV3' or keypoint_net_type == 'DF':

            conf = get_config(config, to_mcu=to_mcu, v3=True)

            self.keypoint_net = KP2DTinyV3(**conf, nClasses=self.n_classes, device=self.device, depth=load_depth)
            self.downsample = conf['downsample']
            self.d_f = 2 ** (conf['downsample'] - 1)
            self.kp2dtiny_config = conf
            self.kp2dtiny_config['nClasses'] = self.n_classes
            self.kp2dtiny_config['device'] = self.device
            self.kp2dtiny_config['name'] = config
            self.kp2dtiny_config['version'] = "V3"

        else:
            raise NotImplemented('Keypoint net type not supported {}'.format(keypoint_net_type))


        self.keypoint_net = self.keypoint_net.to(self.device)
        self.add_optimizer_params('KeypointNet', self.keypoint_net.parameters(), keypoint_net_learning_rate)


        self.io_net = None
        if self.with_io:
            self.io_net = InlierNet(blocks=4)
            self.io_net = self.io_net.to(self.device)
            self.add_optimizer_params('InlierNet', self.io_net.parameters(),  keypoint_net_learning_rate)



        self.train_metrics = {}
        self.vis = {}
    def init_warper(self, size):
        self.warper = tgm.HomographyWarper(size[0]//self.d_f, size[1]//self.d_f, mode='nearest')

    def get_loss_weights(self):
        return {
            'keypoint_loss': self.keypoint_loss_weight,
            'descriptor_loss': self.descriptor_loss_weight,
            'score_loss': self.score_loss_weight,
            'segmentation_loss': self.segmentation_loss_weight,
            'vlad_loss': self.vlad_loss_weight,
            'depth_loss': self.depth_loss_weight,
            'huber_loss': self.huber_loss_factor,
        }
    def set_loss_weights(self, loss_weights):
        self.keypoint_loss_weight = loss_weights['keypoint_loss']
        self.descriptor_loss_weight = loss_weights['descriptor_loss']
        self.score_loss_weight = loss_weights['score_loss']
        self.io_loss_weight = loss_weights['io_loss']
        self.loc_loss_weight = loss_weights['loc_loss']

        self.segmentation_loss_weight = loss_weights['segmentation_loss']
        
        self.vlad_loss_weight = loss_weights['vlad_loss']
        self.depth_loss_weight = loss_weights['depth_loss']
        self.huber_loss_factor = loss_weights['huber_loss']
        
    def gather_info(model):
        init_signature = inspect.signature(model.__init__)
        init_params = init_signature.parameters
        init_args = {name: getattr(model, name) for name in init_params.keys() if hasattr(model, name)}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'init_args': init_args,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'keypoint_net_info': model.keypoint_net.gather_info(),
            'kp2d_config': model.kp2dtiny_config,
        }

        return info
    
    def add_optimizer_params(self, name, params, lr):
        self.optim_params.append(
            {'name': name, 'lr': lr, 'original_lr': lr,
             'params': filter(lambda p: p.requires_grad, params)})
    def init_qat(self):
        self.keypoint_net.qconfig = torch.quantization.get_default_qat_qconfig('x86')
        self.keypoint_net.eval()
        self.keypoint_net.fuse()
        self.keypoint_net = torch.quantization.prepare_qat(self.keypoint_net.train())

    def set_train_flags(self, flags):
        self.train_keypoints = flags['keypoints']
        self.train_segmentation = flags['segmentation']
        self.train_visloc = flags['visloc']
        self.train_depth = flags['depth']

    def forward(self, data, debug = False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch.
        debug : bool
            True if to compute debug data (stored in self.vis).

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """

        loss_2d = torch.tensor(0).float().to(self.device)
        recall_2d = 0
        loss_dict = {}
        B, _, H, W = data['image'].shape

        input_img = data['image'].clone().to(self.device)
        input_img_aug = data['image_aug'].clone().to(self.device)

        homography = data['homography'].to(self.device)

        # Get network outputs

        out_aug =  self.keypoint_net(input_img_aug)
        out_aug = self.keypoint_net.post_processing(out_aug, H, W)
        out = self.keypoint_net(input_img)
        out = self.keypoint_net.post_processing(out, H, W)

        if self.train_keypoints and self.keypoint_loss_weight > 0:
            keypoint_loss = torch.tensor(0).float().to(self.device)
            source_score = out_aug['score']
            source_uv_pred = out_aug['coord']
            source_feat = out_aug['feat']

            target_score = out['score']
            target_uv_pred = out['coord']
            target_feat = out['feat']


            _, _, Hc, Wc = target_score.shape

            # Normalize uv coordinates
            target_uv_norm = _normalize_uv_coordinates(target_uv_pred, H, W)
            source_uv_norm = _normalize_uv_coordinates(source_uv_pred, H, W)

            source_uv_warped_norm = self._warp_homography_batch(source_uv_norm, homography)
            source_uv_warped = _denormalize_uv_coordinates(source_uv_warped_norm, H, W)

            # Border mask
            border_mask = _create_border_mask(B, Hc, Wc)
            border_mask = border_mask.gt(1e-3).to(self.device)

            d_uv_l2_min, d_uv_l2_min_index = _min_l2_norm(source_uv_warped, target_uv_pred, B)
            dist_norm_valid_mask = d_uv_l2_min.lt(4) & border_mask.view(B,Hc*Wc)

            # Keypoint loss
            loc_loss = self.loc_loss_weight *d_uv_l2_min[dist_norm_valid_mask].mean()
            keypoint_loss +=  loc_loss

            loss_dict['loc_loss'] = loc_loss

            #Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.
            if self.descriptor_loss:
                metric_loss, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm.detach(), source_uv_warped_norm.detach(), source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field)
                keypoint_loss +=  self.descriptor_loss_weight * 2 * metric_loss 

                loss_dict['metric_loss'] = metric_loss
            else:
                _, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm, source_uv_warped_norm, source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field, eval_only=True)

            #Score Head Loss
            target_score_associated = target_score.view(B,Hc*Wc).gather(1, d_uv_l2_min_index).view(B,Hc,Wc).unsqueeze(1)
            dist_norm_valid_mask = dist_norm_valid_mask.view(B,Hc,Wc).unsqueeze(1) & border_mask.unsqueeze(1)
            d_uv_l2_min = d_uv_l2_min.view(B,Hc,Wc).unsqueeze(1)
            loc_err = d_uv_l2_min[dist_norm_valid_mask]

            usp_loss = (target_score_associated[dist_norm_valid_mask] + source_score[dist_norm_valid_mask]) * (loc_err - loc_err.mean())
            keypoint_loss += self.score_loss_weight * usp_loss.mean()

            loss_dict['usp_loss'] = self.score_loss_weight * usp_loss.mean()

            target_score_resampled = torch.nn.functional.grid_sample(target_score, source_uv_warped_norm.detach(), mode='bilinear', align_corners=True)

            keypoint_loss += self.score_loss_weight * torch.nn.functional.mse_loss(target_score_resampled[border_mask.unsqueeze(1)],
                                                                                source_score[border_mask.unsqueeze(1)]).mean() * 2
            if self.with_io:
                # Compute IO loss
                io_loss = self._compute_io_loss(source_score, source_feat, target_feat, target_score,
                         B, Hc, Wc, H, W, source_uv_norm, target_uv_norm, source_uv_warped_norm,
                         self.device)

                keypoint_loss +=  self.io_loss_weight * io_loss
                loss_dict['io_loss'] = self.io_loss_weight * io_loss
            loss_2d += self.keypoint_loss_weight *keypoint_loss
        else:
            loss_dict['loc_loss'] = 0.
            loss_dict['metric_loss'] = 0.
            loss_dict['usp_loss'] = 0.
            loss_dict['io_loss'] = 0.

        if self.train_segmentation and self.segmentation_loss_weight > 0:
            seg_aug = out_aug['seg']
            seg = out['seg']

            segmentation_gt = data['seg'].clone().long().to(self.device)
            segmentation_gt_aug = data['seg_aug'].clone().long().to(self.device)

            seg_loss = (self._compute_segmentation_loss(seg, segmentation_gt))*0.5
            seg_loss += (self._compute_segmentation_loss(seg_aug, segmentation_gt_aug))*0.5

            if self.use_mse_loss:
                seg_loss +=  (self.MSE_Loss(seg_aug.softmax(1), self.warper(seg,homography).softmax(1)))*1000

            loss_dict['seg_loss'] = seg_loss*self.segmentation_loss_weight
            loss_2d += seg_loss*self.segmentation_loss_weight
        else:
            loss_dict['seg_loss'] = 0.

        if self.train_visloc and self.vlad_loss_weight > 0:
            vlad_aug = out_aug['vlad']
            vlad = out['vlad']

            vlad_loss = self._hard_global_descriptor_loss(vlad, vlad_aug)
            # if self.use_hard_triplet_loss:
            #     vlad_loss = self._hard_global_descriptor_loss(vlad, vlad_aug)
            # else:
            #     vlad_loss = self._global_descriptor_loss(vlad, vlad_aug)*0.05
            vlad_loss = vlad_loss*self.vlad_loss_weight
            loss_2d += vlad_loss
            loss_dict['vlad_loss'] = vlad_loss
        else:
            loss_dict['vlad_loss'] = 0.


        if self.train_depth and self.depth_loss_weight > 0:
            depth_aug = out_aug['depth']
            depth = out['depth']

            depth_gt = data['depth'].clone().to(self.device)
            depth_gt_aug = data['depth_aug'].clone().to(self.device)

            depth_loss = self._compute_depth_loss(depth, depth_gt)
            depth_loss += self._compute_depth_loss(depth_aug, depth_gt_aug)

            depth_loss += (self.MSE_Loss(depth_aug, self.warper(depth, homography)))* 0.5

            depth_loss = depth_loss * self.depth_loss_weight
            loss_dict['depth_loss'] = depth_loss
            loss_2d += depth_loss
        else:
            loss_dict['depth_loss'] = 0.

        if debug or self.debug:
            if self.train_keypoints:
                # Generate visualization data
                vis_ori = (input_img[0].permute(1, 2, 0).detach().cpu().clone().squeeze() )
                vis_ori -= vis_ori.min()
                vis_ori /= vis_ori.max()
                vis_ori = (vis_ori* 255).numpy().astype(np.uint8)

                vis_aug = (input_img_aug[0].permute(1, 2, 0).detach().cpu().clone().squeeze() )
                vis_aug -= vis_aug.min()
                vis_aug /= vis_aug.max()
                vis_aug = (vis_aug*255).numpy().astype(np.uint8)

                #print(segmentation_debug.min())
                if self.use_color is False:
                    vis_ori = cv2.cvtColor(vis_ori, cv2.COLOR_GRAY2BGR)

                _, top_k = target_score.view(B,-1).topk(self.top_k2, dim=1) #JT: Target frame keypoints
                vis_ori = draw_keypoints(vis_ori, target_uv_pred.view(B,2,-1)[:,:,top_k[0].squeeze()],(0,0,255))

                _, top_k = source_score.view(B,-1).topk(self.top_k2, dim=1) #JT: Warped Source frame keypoints
                vis_ori = draw_keypoints(vis_ori, source_uv_warped.view(B,2,-1)[:,:,top_k[0].squeeze()],(255,0,255))

                _, top_k = source_score.view(B,-1).topk(self.top_k2, dim=1) #JT: Target frame keypoints
                vis_aug = draw_keypoints(vis_aug, source_uv_pred.view(B,2,-1)[:,:,top_k[0].squeeze()],(0,0,255))


                cm = get_cmap('plasma')
                heatmap = target_score[0].detach().cpu().clone().numpy().squeeze()
                heatmap -= heatmap.min()
                heatmap /= heatmap.max()
                heatmap = cv2.resize(heatmap, (W, H))
                heatmap = cm(heatmap)[:, :, :3]

                heatmap_aug = source_score[0].detach().cpu().clone().numpy().squeeze()
                heatmap_aug -= heatmap_aug.min()
                heatmap_aug /= heatmap_aug.max()
                heatmap_aug = cv2.resize(heatmap_aug, (W, H))
                heatmap_aug = cm(heatmap_aug)[:, :, :3]

                vis_ori = np.clip(vis_ori, 0, 255) / 255.
                vis_aug = np.clip(vis_aug, 0, 255) / 255.
                heatmap = np.clip(heatmap * 255, 0, 255) / 255.
                heatmap_aug = np.clip(heatmap_aug * 255, 0, 255) / 255.


                # cv2.imshow('org',vis_ori)
                # cv2.imshow('heatmap',heatmap)
                # cv2.imshow('aug',vis_aug)

                concatenated_seg = np.vstack(
                    (np.hstack((heatmap, heatmap_aug)),
                     np.hstack((vis_ori, vis_aug))))

                cv2.imshow('Keypoints', concatenated_seg)

            if self.train_segmentation:

                seg_deb_warped = self.warper(data['seg'].clone().float().to(self.device), homography).squeeze(0).long()[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                seg_deb_warped = cv2.resize(seg_deb_warped, (W, H))/self.n_classes

                segmentation_debug = segmentation_gt[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                segmentation_debug = cv2.resize(segmentation_debug, (W, H))/self.n_classes

                segmentation_debug_aug = segmentation_gt_aug[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                segmentation_debug_aug = cv2.resize(segmentation_debug_aug, (W, H))/self.n_classes

                segmentation_debug_pred = seg[0].cpu().argmax(0).unsqueeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                segmentation_debug_pred = cv2.resize(segmentation_debug_pred, (W, H))/self.n_classes

                segmentation_debug_aug_pred = seg_aug[0].cpu().argmax(0).unsqueeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                segmentation_debug_aug_pred = cv2.resize(segmentation_debug_aug_pred, (W, H))/self.n_classes
                concatenated_seg = np.vstack(
                    (np.hstack((segmentation_debug, segmentation_debug_aug)), np.hstack((segmentation_debug_pred, segmentation_debug_aug_pred))))

                # Display the stitched image
                cv2.imshow('Segmentation', concatenated_seg)
                # cv2.imshow('seg_warped',(seg_deb_warped-segmentation_debug_aug))
                # cv2.imshow('seg',segmentation_debug/self.n_classes)
                # cv2.imshow('seg_pred',segmentation_debug_pred/self.n_classes)
                # cv2.imshow('seg_pred_aug',np.clip(segmentation_debug_aug_pred/self.n_classes, 0, 1))

            if self.train_depth:
                depth_debug = depth[0].detach().cpu().numpy().squeeze().astype(np.float32)
                depth_debug = cv2.resize(depth_debug, (W, H))
                depth_debug_aug = depth_aug[0].detach().cpu().numpy().squeeze().astype(np.float32)
                depth_debug_aug = cv2.resize(depth_debug_aug, (W, H))
                depth_gt = data['depth'][0].cpu().numpy().squeeze()
                depth_gt = cv2.resize(depth_gt, (W, H))
                depth_gt_aug = data['depth_aug'][0].cpu().numpy().squeeze()
                depth_gt_aug = cv2.resize(depth_gt_aug, (W, H))

                concatenated_depth = np.vstack((np.hstack((depth_gt, depth_gt_aug)),np.hstack((depth_debug, depth_debug_aug))))

                # Display the stitched image
                cv2.imshow('Depth', concatenated_depth)
                # cv2.imshow('depth',depth_debug/depth_debug.max())
                # cv2.imshow('depth_aug',depth_debug_aug/depth_debug_aug.max())
                # cv2.imshow('depth_gt',depth_gt/depth_gt.max())
                # cv2.imshow('depth_gt_aug',depth_gt_aug/depth_gt_aug.max())
            cv2.waitKey(1)
        loss_dict['total_loss'] = loss_2d
        return loss_2d, loss_dict, recall_2d

    def _compute_io_loss(self, source_score,source_feat,target_feat, target_score,
                         B, Hc, Wc, H, W,
                         source_uv_norm, target_uv_norm, source_uv_warped_norm,
                         device, epsilon = 1e-8):

        top_k_score1, top_k_indice1 = source_score.view(B, Hc * Wc).topk(self.top_k2, dim=1, largest=False)
        top_k_mask1 = torch.zeros(B, Hc * Wc).to(device)
        top_k_mask1.scatter_(1, top_k_indice1, value=1)
        top_k_mask1 = top_k_mask1.gt(1e-3).view(B, Hc, Wc)

        top_k_score2, top_k_indice2 = target_score.view(B, Hc * Wc).topk(self.top_k2, dim=1, largest=False)
        top_k_mask2 = torch.zeros(B, Hc * Wc).to(device)
        top_k_mask2.scatter_(1, top_k_indice2, value=1)
        top_k_mask2 = top_k_mask2.gt(1e-3).view(B, Hc, Wc)

        source_uv_norm_topk = source_uv_norm[top_k_mask1].view(B, self.top_k2, 2)
        target_uv_norm_topk = target_uv_norm[top_k_mask2].view(B, self.top_k2, 2)
        source_uv_warped_norm_topk = source_uv_warped_norm[top_k_mask1].view(B, self.top_k2, 2)

        source_feat_topk = torch.nn.functional.grid_sample(source_feat, source_uv_norm_topk.unsqueeze(1),
                                                           align_corners=True).squeeze(2)
        target_feat_topk = torch.nn.functional.grid_sample(target_feat, target_uv_norm_topk.unsqueeze(1),
                                                           align_corners=True).squeeze(2)

        source_feat_topk = source_feat_topk.div(torch.norm(source_feat_topk, p=2, dim=1).unsqueeze(1)+epsilon)
        target_feat_topk = target_feat_topk.div(torch.norm(target_feat_topk, p=2, dim=1).unsqueeze(1)+epsilon)

        dmat = torch.bmm(source_feat_topk.permute(0, 2, 1), target_feat_topk)
        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1)+epsilon)
        dmat_min, dmat_min_indice = torch.min(dmat, dim=2)

        target_uv_norm_topk_associated = target_uv_norm_topk.gather(1, dmat_min_indice.unsqueeze(2).repeat(1, 1, 2))
        point_pair = torch.cat([source_uv_norm_topk, target_uv_norm_topk_associated, dmat_min.unsqueeze(2)], 2)

        inlier_pred = self.io_net(point_pair.permute(0, 2, 1).unsqueeze(3)).squeeze()

        target_uv_norm_topk_associated_raw = target_uv_norm_topk_associated.clone()
        target_uv_norm_topk_associated_raw[:, :, 0] = (target_uv_norm_topk_associated_raw[:, :, 0] + 1) * (
                    float(W - 1) / 2.)
        target_uv_norm_topk_associated_raw[:, :, 1] = (target_uv_norm_topk_associated_raw[:, :, 1] + 1) * (
                    float(H - 1) / 2.)

        source_uv_warped_norm_topk_raw = source_uv_warped_norm_topk.clone()
        source_uv_warped_norm_topk_raw[:, :, 0] = (source_uv_warped_norm_topk_raw[:, :, 0] + 1) * (float(W - 1) / 2.)
        source_uv_warped_norm_topk_raw[:, :, 1] = (source_uv_warped_norm_topk_raw[:, :, 1] + 1) * (float(H - 1) / 2.)

        matching_score = torch.norm(target_uv_norm_topk_associated_raw - source_uv_warped_norm_topk_raw, p=2, dim=2)
        inlier_mask = matching_score.lt(4)
        inlier_gt = 2 * inlier_mask.float() - 1

        return int(inlier_mask.sum() > 10 )*torch.nn.functional.mse_loss(inlier_pred, inlier_gt)

    def _compute_segmentation_loss(self, pred, gt):
        return self.segmentation_criterion(pred, gt.squeeze())*0.5 + self.dice_loss(pred, gt.squeeze())*1.5

    def _global_descriptor_loss(self, pred, pred_aug):
        N,_ = pred.shape
        vlad_loss = torch.tensor(0).float().to(self.device)
        for i in range(N):
            for j in range(N):
                if not i == j:
                    vlad_loss += self.vlad_criterion(pred[i:i+1], pred_aug[i:i+1], pred[j:j+1])
                    vlad_loss += self.vlad_criterion(pred[i:i+1], pred_aug[i:i+1], pred_aug[j:j+1])
                    vlad_loss += self.vlad_criterion(pred_aug[i:i+1], pred[i:i+1], pred[j:j+1])
                    vlad_loss += self.vlad_criterion(pred_aug[i:i+1], pred[i:i+1], pred_aug[j:j+1])

        return vlad_loss.div(N*(N-1)*4)

    def _compute_depth_loss(self, pred, gt):

        #rmse_log = self.RMSE_log(pred, gt)
        mask = gt.gt(0.)
        #grad_fake, grad_real,grad_mask = self.Grad(pred,gt)
        #grad_loss = self.GradLoss(grad_fake, grad_real, mask = grad_mask)* self.grad_factor
        #normal_loss = self.NormalLoss(grad_fake, grad_real)*self.normal_factor
        return  self.SILog_Loss(pred,gt, mask=mask) + self.HuberLoss(pred[mask],gt[mask])*self.huber_loss_factor # + grad_loss + normal_loss
        #return self.MSE_Loss(pred[mask],gt[mask])
    def _hard_global_descriptor_loss(self, pred, pred_aug):
        N,_ = pred.shape
        label = torch.arange(N).to(self.device)
        embeds = torch.cat([pred, pred_aug], dim=0).to(self.device)
        labels = torch.cat([label, label], dim=0).to(self.device)
        vlad_loss = self.vlad_criterion(embeds, labels)
        
        return vlad_loss
    
    def _warp_homography_batch(self, sources, homographies):
        """Batch warp keypoints given homographies.

        Parameters
        ----------
        sources: torch.Tensor (B,H,W,C)
            Keypoints vector.
        homographies: torch.Tensor (B,3,3)
            Homographies.

        Returns
        -------
        warped_sources: torch.Tensor (B,H,W,C)
            Warped keypoints vector.
        """
        B, H, W, _ = sources.shape
        warped_sources = []

        for b in range(B):
            source = sources[b].clone()
            source = source.view(-1, 2)

            source = torch.addmm(homographies[b, :, 2], source, homographies[b, :, :2].t())
            source.mul_(1 / source[:, 2].unsqueeze(1))

            source = source[:, :2].contiguous().view(H, W, 2)
            warped_sources.append(source)
        return torch.stack(warped_sources, dim=0)

def _normalize_uv_coordinates(uv_pred, H, W):
    uv_norm = uv_pred.clone()
    uv_norm[:, 0] = (uv_norm[:, 0] / (float(W - 1) / 2.)) - 1.
    uv_norm[:, 1] = (uv_norm[:, 1] / (float(H - 1) / 2.)) - 1.
    uv_norm = uv_norm.permute(0, 2, 3, 1)
    return uv_norm

def _denormalize_uv_coordinates(uv_norm, H, W):
    uv_pred = uv_norm.clone()
    uv_pred[:, :, :, 0] = (uv_pred[:, :, :, 0] + 1) * (float(W - 1) / 2.)
    uv_pred[:, :, :, 1] = (uv_pred[:, :, :, 1] + 1) * (float(H - 1) / 2.)
    uv_pred = uv_pred.permute(0, 3, 1, 2)
    return uv_pred

def _create_border_mask(B, Hc, Wc):
    border_mask_ori = torch.ones(B, Hc, Wc)
    border_mask_ori[:, 0] = 0
    border_mask_ori[:, Hc - 1] = 0
    border_mask_ori[:, :, 0] = 0
    border_mask_ori[:, :, Wc - 1] = 0
    return border_mask_ori


def _min_l2_norm(source_uv_warped, target_uv_pred, B):
    d_uv_mat_abs = torch.abs(source_uv_warped.view(B, 2, -1).unsqueeze(3) - target_uv_pred.view(B, 2, -1).unsqueeze(2))
    d_uv_l2_mat = torch.norm(d_uv_mat_abs, p=2, dim=1)
    return d_uv_l2_mat.min(dim=2)
