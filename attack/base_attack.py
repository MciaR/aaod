import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg, InstanceList
from mmdet.apis import init_detector, inference_detector
from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.structures import SampleList
from mmengine.registry import MODELS


class BaseAttack():
    """Base class for Attack method."""
    def __init__(self,
                 cfg_file, 
                 ckpt_file,
                 device='cuda:0') -> None:
        self.device = device
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file)
        self.data_preprocessor = self.get_data_preprocess()
        self.test_pipeline = self.get_test_pipeline()

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
    
    def get_pred(self, img: str):
        """Get inference results of model.
        
        Args:
            img (str | np.ndarray): img info.
        Return:
            result (torch.Tensor | np.ndarray): result of inference.
        """
        result = inference_detector(self.model, img)
        return result
    
    def get_data_from_img(self, img):
        """Get preprocessed data from img path
        Args:
            img (str | np.ndarray): img info.
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
    
    def attack(self, img):
        """Call attack method to generate adversarial sample."""
        ad_result = None
        attack_name = os.path.basename(__file__).split('.')[0]
        save_dir = 'ad_result/' + attack_name
        img_name = img.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ad_img_path = os.path.join(save_dir, img_name)
        pertub_img_path = os.path.join(save_dir, 'pertub_' + img_name)

        per_image, ad_result = self._attack(img=img)
        cv2.imwrite(ad_img_path, ad_result)
        cv2.imwrite(pertub_img_path, per_image)
        
        assert os.path.exists(ad_img_path), \
            f'`{ad_img_path}` does not save successfully!.'
        
        return pertub_img_path, ad_img_path
    
    def _attack(self, img):
        pass