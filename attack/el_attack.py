import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from visualizer import AAVisualizer


class ELAttack():
    def __init__(self,
                 cfg_file, 
                 ckpt_file, ) -> None:
        self.visualizer = AAVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
        self.model = self.visualizer.model
        self.device = self.visualizer.device

    def _forward(self, model, img):
        return self.visualizer._forward(model=model, img=img)
    
    def attack(self, img, attack_method='dcn_attack'):
        """Call different attack method to generate adversarial sample."""
        ad_result = None
        save_dir = 'ad_result/' + attack_method
        img_name = img.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ad_img_path = os.path.join(save_dir, img_name)

        if attack_method == 'dcn_attack':
            ad_result = self.dcn_attack(img=img)
            cv2.imwrite(ad_img_path, ad_result)
        
        assert os.path.exists(ad_img_path), \
            f'`{ad_img_path}` does not save successfully!.'
        
        return ad_img_path

    def dcn_attack(self, img, out_ind=3, alpha=1):
        """Simply use dcn output as noise to add to ori img to generate adversarial sample.
        
        Args:
            img (torch.Tensor | np.Numpy): ori image.
            out_ind (int): index of out_indices of backbone, coressponding to `out_indices` in config file.
            alpha (float): attack power factor.

        Return:
            ad_result (torch.Tensor | np.Numpy): adversarial sample.
        """
        result = self.visualizer._forward(model=self.model.backbone, img=img)
        image = cv2.imread(img)

        featmap = F.interpolate(
            result[out_ind].squeeze()[None],
            image.shape[:2],
            mode='bilinear',
            align_corners=False)[0]
        mean_featmap = torch.mean(featmap, dim=0)

        mean_featmap = mean_featmap.detach().cpu().numpy()
        perturbed_image = np.stack((mean_featmap,) * 3, axis=-1)
        adv_example = perturbed_image*alpha + image

        return adv_example


