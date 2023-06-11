import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from attack import BaseAttack


class HAFAttack(BaseAttack):
    """High Activation Featuremap Attack."""
    def __init__(self, 
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 p: int = 2,
                 eplison: float = 0.5, 
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device, attack_params=dict(p=p, eplison=eplison))

    def attack_method(self, bb_outputs, topk = 1):
        """ Find mean featmap max and min activate value pixel, and switch them."""
        attack_result = []
        out_len = len(bb_outputs)
        for i in range(out_len):
            featmap = bb_outputs[i]
            # feat_maps: (N, C, H, W)
            attack_result.append(self.modify_featmap(featmap=featmap))
        
        return attack_result

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

    def generate_adv_samples(self, x, eplison, p):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image path.
            eplison (float): niose strength.    
            p (int): default `2`, p-norm to calculate distance between clean and adv image.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        pass
