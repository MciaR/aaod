import torch
import torch.nn.functional as F
import numpy as np
import mmcv

from attack import BaseAttack
from PIL import Image
from torch.optim.lr_scheduler import StepLR


class HEFMAAttack(BaseAttack):
    """Head Enhanced Feature Map Adversarial Attack.
    Args:         
        gamma (float): scale factor of normalizing noise `r`.
        M (float): SGD total step, if iter reach the limit or every RP has been attack, the loop ends (for DAG).
    """
    def __init__(self, 
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 gamma=0.5,
                 M=500,
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, 
                         attack_params=dict(gamma=gamma, M=M))
        
    def generate_adv_samples(self, x, log_info=True):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image path.
            log_info (bool): if print the train information.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """

        r = torch.rand_like(x, requires_grad=True, device=self.device)
        



