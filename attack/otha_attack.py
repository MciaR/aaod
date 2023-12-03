import torch
import torch.nn.functional as F
import numpy as np
import mmcv

from attack import BaseAttack
from PIL import Image
from torch.optim.lr_scheduler import StepLR


class OTHAAttack(BaseAttack):
    """Object-wise Top-k High Activation Attack.
    Args:         
        eplison (float): niose strength.    
        p (int): default `2`, p-norm to calculate distance between clean and adv image.
        adv_type (str): 
            - default `residual`, that means only optimize the noise added to image. 
            - `direct`, that means optimize the whole adversarial sample.
        constrain (str):
            - default `consine_sim`, that means use consine similarity to comput loss.
            - `distance`, that means use distance function to comput loss.
        channel_mean (bool):
            - default `False`, means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
            - `True`, calculate each point mean by channel-wise, the featmap shape is (B, H, W).
    """
    def __init__(self, 
                 modify_percent=0.7,
                 scale_factor=0.01,
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 feature_type = 'backbone', # `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
                 channel_mean=False, # means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
                 stages: list = [4], # attack stage of backbone. `(0, 1, 2, 3)` for resnet. 看起来0,3时效果最好。ssd和fr_vgg16就取0
                 p: int = 2, # attack param
                 alpha: float = 5,  # attack param, factor of distance loss. 0.125 for ssd300, 0.25 for fr
                 lr: float = 0.005, # default 0.05
                 M: int = 1000, # attack param, max step of generating perbutaion. 300 for fr, 1000 for ssd.
                 adv_type='direct', # `direct` or `residual`, `direct` means cal pertub noise as whole image directly, `residual` means only cal pertub noise.
                 constrain='consine_sim', #  - default `consine_sim`, that means use consine similarity to comput loss. `distance`, that means use distance function to comput loss.
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, 
                         attack_params=dict(modify_percent=modify_percent, scale_factor=scale_factor, p=p, alpha=alpha, stages=stages, M=M, lr=lr, feature_type=feature_type, adv_type=adv_type, constrain=constrain, channel_mean=channel_mean))
