import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from visualizer import AAVisualizer
from mmdet.apis import init_detector


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
        pertub_img_path = os.path.join(save_dir, 'pertub_' + img_name)

        if attack_method == 'dcn_attack':
            per_image, ad_result = self.dcn_attack(img=img)
            cv2.imwrite(ad_img_path, ad_result)
            cv2.imwrite(pertub_img_path, per_image)
        
        assert os.path.exists(ad_img_path), \
            f'`{ad_img_path}` does not save successfully!.'
        
        return ad_img_path

    def dcn_attack(self, img, out_ind=[1, 2, 3], alpha=0.35):
        """Simply use dcn output as noise to add to ori img to generate adversarial sample.
        
        Args:
            img (torch.Tensor | np.Numpy): ori image.
            out_ind (int): index of out_indices of backbone, coressponding to `out_indices` in config file.
            alpha (float): attack power factor.

        Return:
            ad_result (torch.Tensor | np.Numpy): adversarial sample.
        """
        dcn_cfg = 'configs/faster_rcnn_r101_dcn_c3_c5_fpn_coco.py'
        dcn_ckpt = 'pretrained/faster_rcnn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth'
        model = init_detector(dcn_cfg, dcn_ckpt, device='cuda:0')

        result = self.visualizer._forward(model=model.backbone, img=img)
        image = cv2.imread(img)
        perturbed_image = np.zeros(image.shape)
        per_factor = [0.05, 0.7, 0.25]

        for i in range(len(out_ind)):
            ind = out_ind[i]
            featmap = F.interpolate(
                result[ind].squeeze()[None],
                image.shape[:2],
                mode='bilinear',
                align_corners=False)[0]
            mean_featmap = torch.mean(featmap, dim=0)

            mean_featmap = mean_featmap.detach().cpu().numpy()
            temp_img = np.zeros(mean_featmap.shape)
            temp_img = cv2.normalize(mean_featmap, temp_img, 0, 255, cv2.NORM_MINMAX)

            temp_img = np.stack((temp_img,) * 3, axis=-1)
            perturbed_image += (per_factor[i] * temp_img)

        perturbed_image = np.asarray(perturbed_image, dtype=np.uint8)

        # perturbed_image = cv2.applyColorMap(perturbed_image, cv2.COLORMAP_JET)
        # perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2RGB)

        adv_example = perturbed_image*alpha + image

        return perturbed_image, adv_example


