import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from attack import BaseAttack
from PIL import Image

class HAFAttack(BaseAttack):
    """High Activation Featuremap Attack."""
    def __init__(self, 
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 stage: int = 0, # attack stage of backbone. `(0, 1, 2, 3)` for resnet.
                 p: int = 2, # attack param
                 eplison: float = 0.5,  # attack param
                 M: int = 1000, # attack param, max step of generating perbutaion.
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
    
    def get_normalize_distance(self, x, adv, p: int = 2):
        """Get normailize distance bettween clean image and adv image.
        Args:
            x (np.ndarray | torch.Tensor): clean image.
            adv (np.ndarray | torch.Tensor): adversarial sample.
            p (int): p-norm, default 2.
        Return:
            eplison (float): p-norm distance bettween clean and adv image. 
        """
        pass

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
        img = Image.open(x)
        clean_img = np.array(img)
        r = torch.rand(clean_img.shape) * 10
        r = r.cuda()
        r.requires_grad = True

        # params
        step = 0
        optimizer = torch.optim.SGD([r], lr=0.01, momentum=0.9)
        loss_fn = torch.nn.MSELoss()
        
        while step < M:
            # calculate output featmap
            pertub_image = clean_img + r.detach().cpu().numpy()
            pertub_bb_output = self._forward(img=pertub_image, stage='backbone')
            pertub_featmap = pertub_bb_output[stage]

            loss = loss_fn(pertub_featmap, attack_gt_featmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0:
                print("Train step [{}/{}], loss: {}.".format(step, M, loss))

        print("Generate adv compeleted!")

        pertub = r.detach().cpu().numpy()
        adv_image = clean_img + pertub

        return pertub, adv_image
