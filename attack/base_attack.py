import os
import cv2
import torch
import mmcv
import torch.nn as nn
import numpy as np

from PIL import Image
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import DATASETS
from mmengine.registry import MODELS


class BaseAttack():
    """Base class for Attack method."""
    def __init__(self,
                 cfg_file, 
                 ckpt_file,
                 attack_params,
                 exp_name=None, 
                 cfg_options: dict = None,
                 device='cuda:0',) -> None:
        self.device = device
        self.cfg_file = cfg_file
        self.ckpt_file = ckpt_file
        self.exp_name = exp_name
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file, cfg_options=cfg_options)
        self.dataset = self.get_dataset()
        self.data_preprocessor = self.get_data_preprocess()
        self.test_pipeline = self.get_test_pipeline()
        self.attack_params = dict(**attack_params)
        self.init_attack_params()

    def init_attack_params(self):
        assert self.attack_params is not None

        for key, value in self.attack_params.items():
            setattr(self, key, value)

    def get_model(self, cfg_file, ckpt_path, cfg_options):
        """Initialize the detector with cfg and ckpt file.
        Args:
            cfg_file (str): config file path.
            ckpt_path (str): checkpoint file path.
            cfg_options (dict): a dict will overwirte corresponding cfg k-v pair in cfg_file.
        """
        model = init_detector(cfg_file, ckpt_path, device=self.device, cfg_options=cfg_options)
        return model 
    
    def get_dataset(self):
        """Get dataset from config."""
        dataset_cfg = self.model.cfg._cfg_dict.test_dataloader.dataset
        dataset = DATASETS.build(dataset_cfg)

        return dataset
    
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
    
    def reverse_augment(self, x, datasample):
        """Reverse tensor to input image."""
        ori_shape = datasample.ori_shape
        pad_shape = datasample.pad_shape

        mean = self.data_preprocessor.mean
        std = self.data_preprocessor.std 

        # revert normorlize
        ori_pic = x * std + mean
        # revert bgr_to_rgb
        # NOTE: dont need to revert bgr_to_rgb, beacuse saving format is RGB if using PIL.Image
        # ori_pic = ori_pic[[2, 1, 0], ...]
        # revert pad
        ori_pic = ori_pic[:, :datasample.img_shape[0], :datasample.img_shape[1]]

        # (c, h, w) to (h, w, c)
        ori_pic = ori_pic.permute(1, 2, 0)
        # cut overflow values
        ori_pic = torch.clamp(ori_pic, 0, 255)

        ori_pic = ori_pic.detach().cpu().numpy()

        keep_ratio = False
        for trans in self.test_pipeline:
            if type(trans).__name__ == 'Resize':
                keep_ratio = trans.keep_ratio

        # for fr
        # cuz fr is keep ratio and ssd just resize any pic to (300, 300)
        if keep_ratio:
            ori_pic, _ = mmcv.imrescale(
                                ori_pic,
                                ori_shape,
                                interpolation='bilinear',
                                return_scale=True,
                                backend='cv2')
        else:
            ori_pic, _, _ = mmcv.imresize(
                                ori_pic,
                                (ori_shape[1], ori_shape[0]),
                                interpolation='bilinear',
                                return_scale=True,
                                backend='cv2')
        
        return ori_pic  

    def attack(self, img, save_root=None, data_sample=None):
        """Get inference results of model.
        Args:
            img (str): img path.
            save_root (str): save root dictrionary, if `None`, results will be saved in `records/attack_pics`. Default `None`.
            data_sample (DetDataSample): contains gt_instances, i.e. gt bboxes and gt labels.
            exp_name (str): exp name.
        Return:
            pertub_path (str): return path of pertub noise.
            adv_path (str):  return path of adversarial sample.
        """
        # get adversarial samples, kwargs must be implemented.
        # NOTE: old version code
        # pertub, adv = self.generate_adv_samples(x=img, **self.attack_params)
        pertub, adv = self.generate_adv_samples(x=img, data_sample=data_sample)

        attack_name = self.get_attack_name()
        if save_root is None:
            save_root = 'records/attack_pics/'
        save_dir = os.path.join(save_root, attack_name)
        
        if self.exp_name is not None:
            save_dir = os.path.join(save_dir, self.exp_name)

        adv_save_dir = os.path.join(save_dir, 'adv')
        pertub_save_dir = os.path.join(save_dir, 'pertub')
        img_name = os.path.basename(img).split('.')[0] + '.png' # jpg会压缩一部分图像，导致攻击损失。

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

        return pertub_img_path, ad_img_path

    
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
                if self.model.with_neck:
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
    def generate_adv_samples(self, x, data_sample=None, log_info=True, **kwargs):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image's path.
            data_sample (DetDataSample): contains gt_instances, i.e. gt bboxes and gt labels. Default `None`.
            log_info (bool): whether print training log. Default `True`.
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