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
                 stage: int = 3, # attack stage of backbone. `(0, 1, 2, 3)` for resnet.
                 p: int = 2, # attack param
                 eplison: float = 0.01,  # attack param
                 M: int = 10000, # attack param, max step of generating perbutaion.
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, attack_params=dict(p=p, eplison=eplison, stage=stage, M=M))

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
        std = [58.395, 57.12, 57.375]
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

        ori_pic, _ = mmcv.imrescale(
                            ori_pic,
                            ori_shape,
                            interpolation='bilinear',
                            return_scale=True,
                            backend='cv2')
        
        return ori_pic

    def generate_adv_samples(self, x, stage, eplison, p, M):
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
        target_fm = bb_outs[stage]
        # featmap that the attack should be generated
        attack_gt_featmap = self.modify_featmap(target_fm)

        # initialize r
        data = self.get_data_from_img(img=x)
        clean_image = data['inputs']
        r = torch.randn(clean_image.shape, requires_grad=True, device=self.device)
        r = r.detach()
        r.requires_grad = True
        norm_thr = torch.norm(clean_image * eplison, p=2)
        
        # params
        step = 0
        optimizer = torch.optim.Adam(params=[r], lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        while step < M:
            # calculate output featmap
            pertub_bb_output = self.model.backbone(clean_image + r)
            pertub_featmap = pertub_bb_output[stage]

            loss = loss_fn(pertub_featmap, attack_gt_featmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            r_norm = torch.norm(r, p=2) 

            if r_norm < norm_thr and loss < 0.01:
                break

            if step % 10 == 0:
                print("Train step [{}/{}], loss: {}, r_norm: [{}/{}].".format(step, M, loss, r_norm, norm_thr))

        print("Generate adv compeleted!")

        pertub = self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=clean_image.squeeze(), datasample=data['data_samples'][0]) + pertub * 0.01

        return pertub, adv_image
