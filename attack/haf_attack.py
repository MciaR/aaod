import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mmcv

from attack import BaseAttack
from PIL import Image

class HAFAttack(BaseAttack):
    """High Activation Featuremap Attack."""
    def __init__(self, 
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 stage: list = [0, 1], # attack stage of backbone. `(0, 1, 2, 3)` for resnet. 看起来0,3时效果最好。ssd和fr_vgg16就取0
                 p: int = 2, # attack param
                 alpha: float = 0.25,  # attack param, factor of distance loss. 0.125 for ssd300, 0.25 for fr
                 lr: float = 0.05, # default 0.05
                 M: int = 300, # attack param, max step of generating perbutaion. 300 for fr, 1000 for ssd.
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, attack_params=dict(p=p, alpha=alpha, stage=stage, M=M, lr=lr))

    def get_topk_info(
            self,
            input: torch.Tensor,
            k: int = 10,
            largest: bool = True,
            ):
        flatten_tensor = input.flatten()
        values, topk_indices = torch.topk(input=flatten_tensor, k=k, dim=0, largest=largest)
        assert len(input.shape) == 2, \
            f' featmap tensor must be shape (H, W)'
        H, W = input.shape

        h_indices = topk_indices // W
        w_indices = topk_indices % W

        indices = torch.stack((h_indices, w_indices), dim=1)

        return values, indices
    
    def modify_featmap(
            self,
            featmap: torch.Tensor,
            modify_percent: float = 0.7,
            scale_factor: float = 0.01):
        """Modify topk value in each featmap (H, W).
        Args:
            featmap (torch.Tensor): shape `(N, C, H, W)`
            mean_featmap
            scale_factor (float): miniumize factor
        """
        N, C, H, W = featmap.shape
        for sample_ind in range(N):
            sample_featmap = featmap[sample_ind]
            k = int(H * W * modify_percent)
            mean_featmap = torch.mean(sample_featmap, dim=0)
            _, topk_indices = self.get_topk_info(input=mean_featmap, k=k, largest=True)

            # scale indices value in each featmap
            featmap[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] = featmap[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] * scale_factor

        return featmap

    def reverse_augment(self, x, datasample):
        """Reverse tensor to input image."""
        ori_shape = datasample.ori_shape
        pad_shape = datasample.pad_shape

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375] # for fr
        # std = [1, 1, 1] # for ssd
        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)

        # revert normorlize
        ori_pic = x * std_t + mean_t
        # revert bgr_to_rgb
        ori_pic = ori_pic[[2, 1, 0], ...]
        # revert pad
        ori_pic = ori_pic[:, :datasample.img_shape[0], :datasample.img_shape[1]]

        # (c, h, w) to (h, w, c)
        ori_pic = ori_pic.permute(1, 2, 0)

        ori_pic = ori_pic.detach().cpu().numpy()

        # for fr
        ori_pic, _ = mmcv.imrescale(
                            ori_pic,
                            ori_shape,
                            interpolation='bilinear',
                            return_scale=True,
                            backend='cv2')
        # ori_pic, _, _ = mmcv.imresize(
        #                     ori_pic,
        #                     (ori_shape[1], ori_shape[0]),
        #                     interpolation='bilinear',
        #                     return_scale=True,
        #                     backend='cv2')
        
        return ori_pic

    # def generate_adv_samples(self, x, stage, eplison, p, M):
    #     """Attack method to generate adversarial image.
    #     Args:
    #         x (str): clean image path.
    #         eplison (float): niose strength.    
    #         p (int): default `2`, p-norm to calculate distance between clean and adv image.
    #     Return:
    #         noise (np.ndarray | torch.Tensor): niose which add to clean image.
    #         adv (np.ndarray | torch.Tensor): adversarial image.
    #     """
    #     # get feature map of clean img.
    #     bb_outs = self._forward(img=x, stage='backbone')
    #     # target featmap
    #     target_fm = bb_outs[stage]
    #     # featmap that the attack should be generated
    #     attack_gt_featmap = self.modify_featmap(target_fm)

    #     # initialize r
    #     data = self.get_data_from_img(img=x)
    #     clean_image = data['inputs']
    #     r = torch.randn(clean_image.shape, requires_grad=True, device=self.device)
    #     r = r.detach()
    #     r.requires_grad = True
    #     gt_r = torch.zeros_like(r)
        
    #     # params
    #     step = 0
    #     optimizer = torch.optim.Adam(params=[r], lr=0.01)
    #     loss_pertub = torch.nn.MSELoss()
    #     loss_distance = torch.nn.MSELoss()
    #     alpha = 0.1
        
    #     while step < M:
    #         # calculate output featmap
    #         pertub_bb_output = self.model.backbone(clean_image + r)
    #         pertub_featmap = pertub_bb_output[stage]

    #         l1 = loss_pertub(pertub_featmap, attack_gt_featmap) 
    #         l2 = loss_distance(r, gt_r)
    #         loss = l1 + alpha * l2

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         step += 1

    #         if step % 10 == 0:
    #             print("Train step [{}/{}], loss: {}, pertub_loss: {}, distance_loss: {}.".format(step, M, loss, l1, l2))

    #     print("Generate adv compeleted!")

    #     pertub = self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0])
    #     adv_image = self.reverse_augment(x=clean_image.squeeze(), datasample=data['data_samples'][0]) + pertub

    #     return pertub, adv_image

    def generate_adv_samples(self, x, stage, alpha, p, M, lr):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image path.
            eplison (float): niose strength.    
            p (int): default `2`, p-norm to calculate distance between clean and adv image.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        # get feature map of clean img.
        bb_outs = self._forward(img=x, stage='backbone')
        # target featmap
        target_fm = [bb_outs[i] for i in stage]
        # featmap that the attack should be generated
        attack_gt_featmap = [self.modify_featmap(fm) for fm in target_fm]

        # initialize r
        data = self.get_data_from_img(img=x)
        clean_image = data['inputs']
        r = torch.randn(clean_image.shape, requires_grad=True, device=self.device) # for fr
        # r = clean_image.clone() + torch.randn(clean_image.shape, requires_grad=True, device=self.device) # for ssd
        # r.retain_grad()
        
        # params
        step = 0
        optimizer = torch.optim.Adam(params=[r], lr=lr)
        loss_pertub = torch.nn.MSELoss()
        loss_distance = torch.nn.MSELoss()

        while step < M:
            # calculate output featmap
            pertub_bb_output = self.model.backbone(r)
            pertub_featmap = [pertub_bb_output[i] for i in stage]

            l1 = 0
            for p_fm, gt_fm in zip(pertub_featmap, attack_gt_featmap):
                l1 += (1 / len(stage) * loss_pertub(p_fm, gt_fm)) 
            l2 = loss_distance(r, clean_image)
            loss = l1 + alpha * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0:
                print("Train step [{}/{}], loss: {}, pertub_loss: {}, distance_loss: {}.".format(step, M, loss, l1, l2))

        # print("Generate adv compeleted!")

        pertub = self.reverse_augment(x=(r.squeeze() - clean_image.squeeze()), datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0])

        return pertub, adv_image
