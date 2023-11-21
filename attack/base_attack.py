import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector, inference_detector
from mmengine.registry import MODELS


class BaseAttack():
    """Base class for Attack method."""
    def __init__(self,
                 cfg_file, 
                 ckpt_file,
                 attack_params,
                 device='cuda:0',) -> None:
        self.device = device
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file)
        self.data_preprocessor = self.get_data_preprocess()
        self.test_pipeline = self.get_test_pipeline()
        self.attack_params = dict(**attack_params)

    def get_model(self, cfg_file, ckpt_path):
        model = init_detector(cfg_file, ckpt_path, device=self.device)
        return model 
    
    def get_test_pipeline(self, load_from_ndarray: bool = False):
        """Get data preprocess pipeline"""
        cfg = self.model.cfg
        test_pipeline = get_test_pipeline_cfg(cfg)
        if load_from_ndarray:
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
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
    
    def attack(self, img, save=False):
        """Get inference results of model.
        Args:
            img (str): img path.
            save (bool): whether save noise and adv.
        Return:
            result (torch.Tensor | np.ndarray): result of inference.
            pertub_path (str): if `save=True`, it will return path of pertub noise, otherwise return np.ndarray of pertub noise.
            adv_path (str):  if `save=True` it will return path of adversarial sample, otherwise return np.ndarray of adv sample.
        """
        # get adversarial samples, kwargs must be implemented.
        pertub, adv = self.generate_adv_samples(x=img, **self.attack_params)
        # forward detector to get pred results.
        result = self.get_pred(img=adv)

        if save:
            attack_name = os.path.basename(__file__).split('.')[0]
            save_dir = 'ad_result/' + attack_name
            img_name = os.path.basename(img)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ad_img_path = os.path.join(save_dir, img_name)
            pertub_img_path = os.path.join(save_dir, 'pertub_' + img_name) 

            adv_image = Image.fromarray(adv.astype(np.uint8))
            pertub_image = Image.fromarray(pertub.astype(np.uint8))

            adv_image.save(ad_img_path)
            pertub_image.save(pertub_img_path)
            
            assert os.path.exists(ad_img_path) and os.path.exists(pertub_img_path), \
                f'`{ad_img_path}` or `{pertub_img_path}` does not save successfully!.'
            
            return result, pertub_img_path, ad_img_path
        else:
            return result, pertub.astype(np.uint8), adv.astype(np.uint8)
    
    def get_data_from_img(self, img):
        """Get preprocessed data from img path
        Args:
            img (str | np.ndarray): img info.
        Return:
            data (dict): the data format can forward model.
        """
        load_from_ndarray = False
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
            load_from_ndarray = True
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        test_pipeline = self.get_test_pipeline(load_from_ndarray=load_from_ndarray)
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
    
    def get_pred(self, img):
        """Get pred result of img
        Args:
            img (str): path of img.
        Returns:
            result (DetDataSample): result of pred.
        """
        result = inference_detector(self.model, img)
        return result
    
    def generate_adv_samples(self, x, **kwargs):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image's path.
            kwargs: other key-value args.    
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        pass