import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector
from mmengine.registry import MODELS


class ELAttack():
    """Explainable Location Adversarial Attack."""
    def __init__(self,
                 cfg_file='configs/faster_rcnn_r101_dcn_c3_c5_fpn_coco.py', 
                 ckpt_file='pretrained/faster_rcnn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth',
                 device='cuda:0') -> None:
        self.device = device
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file)
        self.data_preprocessor = self.get_data_preprocess()

    def get_model(self, cfg_file, ckpt_path):
        model = init_detector(cfg_file, ckpt_path, device=self.device)
        return model 
    
    def get_test_pipeline(self):
        """Get data preprocess pipeline"""
        cfg = self.model.cfg
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline = Compose(test_pipeline)

        return test_pipeline
    
    def get_data_preprocess(self):
        """Get data preprocessor"""
        data_preprocessor = self.model.data_preprocessor
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, nn.Module):
            data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)  

        return data_preprocessor   
    
    def get_data_from_img(self, img):
        """Get preprocessed data from img path
        Args:
            img (str): path of img.
        Return:
            data (dict): the data format can forward model.
        """
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        test_pipeline = self.get_test_pipeline()
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]


        data = self.data_preprocessor(data_, False) 

        return data
    
    def _forward(self, stage, img):
        """ Get model output.
        
        Args:
            stage (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
            imgs (str): img path. 
        Return:
            feats (List[Tensor]): List of model output. 
        """
        data = self.get_data_from_img(img=img)
        input_data = data['inputs']

        # forward the model
        with torch.no_grad():
            if stage == 'backbone':
                feat = self.model.backbone(input_data)
            else:
                feat = self.model.backbone(input_data)
                feat = self.model.neck(feat)

        return feat
    
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
        
        return pertub_img_path, ad_img_path

    def dcn_attack(self, img, out_ind=[3, 4], alpha=0.35):
        """Simply use dcn output as noise to add to ori img to generate adversarial sample.
        
        Args:
            img (torch.Tensor | np.Numpy): ori image.
            out_ind (int): index of out_indices of backbone, coressponding to `out_indices` in config file.
            alpha (float): attack power factor.

        Return:
            ad_result (torch.Tensor | np.Numpy): adversarial sample.
        """
        result = self._forward(stage="neck",img = img)
        image = cv2.imread(img)
        perturbed_image = np.zeros(image.shape)
        per_factor = [0.4, 0.6]

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


