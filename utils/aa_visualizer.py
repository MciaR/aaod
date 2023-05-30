from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mmcv.transforms import Compose
from mmdet.registry import DATASETS
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample


class AAVisualizer(DetLocalVisualizer):
    """Visualizer for Adversarial Attack."""
    def __init__(self, 
                 cfg_file, 
                 ckpt_file, 
                 name: str = 'visualizer', 
                 device='cuda:0'):
        super().__init__(name=name)
        self.device = device
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file)
        self.dataset_meta = self.model.dataset_meta
    
    def get_model(self, cfg_file, ckpt_path):
        model = init_detector(cfg_file, ckpt_path, device=self.device)
        return model 
    
    def get_preprocess(self):
        """Get data preprocess pipeline"""
        cfg = self.model.cfg
        test_pipeline = get_test_pipeline_cfg(cfg)
        test_pipeline = Compose(test_pipeline)

        return test_pipeline
    
    def get_dataset(self):
        """Get dataset from config."""
        dataset_cfg = self.model.cfg._cfg_dict.test_dataloader.dataset
        dataset = DATASETS.build(dataset_cfg)

        return dataset
    
    def get_pred(self, img: str):
        """Get inference results of model.
        
        Args:
            img (str): img path.
        Return:
            result (torch.Tensor | np.ndarray): result of inference.
        """
        result = inference_detector(self.model, img)
        return result
    
    def draw_dt_gt(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        return drawn_img
    
    def _forward(self, model, img):
        """ Get model output.
        
        Args:
            model (nn.Module): such as `model.backbone`, `model.head`.
            imgs (str): img path. 
        Return:
            feats (List[Tensor]): List of model output. 
        """
        preprocess = self.get_preprocess()

        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = preprocess(data_)
        input_data = data_['inputs'].float().unsqueeze(0).to(self.device)

        # forward the model
        with torch.no_grad():
            feat = model(input_data)

        return feat