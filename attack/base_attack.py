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
                 cfg_options: dict = None,
                 device='cuda:0',) -> None:
        self.device = device
        self.cfg_file = cfg_file
        self.ckpt_file = ckpt_file
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file, cfg_options=cfg_options)
        self.data_preprocessor = self.get_data_preprocess()
        self.test_pipeline = self.get_test_pipeline()
        self.attack_params = dict(**attack_params)
        self.init_attack_params()

    def init_attack_params(self):
        assert self.attack_params is not None

        for key, value in self.attack_params.items():
            setattr(self, key, value)

    def update_model(self):
        self.model = self.get_model(cfg_file=self.cfg_file, ckpt_path=self.ckpt_file, cfg_options=None)

    def get_model(self, cfg_file, ckpt_path, cfg_options):
        """Initialize the detector with cfg and ckpt file.
        Args:
            cfg_file (str): config file path.
            ckpt_path (str): checkpoint file path.
            cfg_options (dict): a dict will overwirte corresponding cfg k-v pair in cfg_file.
        """
        model = init_detector(cfg_file, ckpt_path, device=self.device, cfg_options=cfg_options)
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
    
    def get_attack_name(self):
        """Get attack name."""
        return type(self).__name__
    
    def attack(self, img, exp_name=None):
        """Get inference results of model.
        Args:
            img (str): img path.
            exp_name (str): exp name.
        Return:
            result (torch.Tensor | np.ndarray): result of inference.
            pertub_path (str): return path of pertub noise.
            adv_path (str):  return path of adversarial sample.
        """
        # get adversarial samples, kwargs must be implemented.
        # NOTE: old version code
        # pertub, adv = self.generate_adv_samples(x=img, **self.attack_params)
        pertub, adv = self.generate_adv_samples(x=img)

        attack_name = self.get_attack_name()
        save_dir = 'records/attack_pics/' + attack_name
        if exp_name is not None:
            save_dir = os.path.join(save_dir, exp_name)

        adv_save_dir = os.path.join(save_dir, 'adv')
        pertub_save_dir = os.path.join(save_dir, 'pertub')
        img_name = os.path.basename(img).split('.')[0] + '.jpg' # 和PNG的攻击效果没有本质区别

        ad_img_path = os.path.join(adv_save_dir, img_name)
        pertub_img_path = os.path.join(pertub_save_dir, img_name) 

        if not os.path.exists(adv_save_dir):
            os.makedirs(adv_save_dir)
        if not os.path.exists(pertub_save_dir):
            os.makedirs(pertub_save_dir)

        adv_image = Image.fromarray(adv.astype(np.uint8))
        pertub_image = Image.fromarray(pertub.astype(np.uint8))

        adv_image.save(ad_img_path)
        pertub_image.save(pertub_img_path)
        
        assert os.path.exists(ad_img_path) and os.path.exists(pertub_img_path), \
            f'`{ad_img_path}` or `{pertub_img_path}` does not save successfully!.'
        
        # re-initialize model to remove modified cfg affection.
        self.update_model()
        
        # forward detector to get pred results.
        result = self.get_pred(img=ad_img_path)

        return result, pertub_img_path, ad_img_path

    
    def get_data_from_img(self, img):
        """Get preprocessed data from img path
        Args:
            img (str | np.ndarray): img info.
        Return:
            data (dict): the data format can forward model.
        """
        # load_from_ndarray = False
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
            # load_from_ndarray = True
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = self.test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        data = self.data_preprocessor(data_, False) 

        return data
    
    def _forward(self, img, feature_type=None):
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
            if feature_type == 'backbone':
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
    
    # NOTE: **kwargs在新版代码中可以不用，依然采用这种方式只是作为示例
    def generate_adv_samples(self, x, log_info=True, **kwargs):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image's path.
            kwargs: other key-value args.    
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        pass
    
    def get_target_feature(
        self,
        ori_features,
        ):
        """Get target features for visualizer."""
        pass