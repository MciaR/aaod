from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import matplotlib.patches as patches

from mmcv.transforms import Compose
from mmdet.registry import DATASETS
from mmdet.utils import get_test_pipeline_cfg, InstanceList
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample, SampleList
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.registry import MODELS
from mmengine.visualization.utils import img_from_canvas, check_type, tensor2ndarray


class AAVisualizer(DetLocalVisualizer):
    """Visualizer for Adversarial Attack."""
    def __init__(self, 
                 cfg_file, 
                 ckpt_file, 
                 name: str = 'visualizer', 
                 device='cuda:0',
                 fig_fontsize=18,
                 ):
        super().__init__(name=name)
        self.device = device
        self.model = self.get_model(cfg_file=cfg_file, ckpt_path=ckpt_file)
        self.dataset_meta = self.model.dataset_meta
        self.data_preprocessor = self.get_data_preprocess()
        self.fig_fontsize = fig_fontsize
    
    def get_model(self, cfg_file, ckpt_path):
        model = init_detector(cfg_file, ckpt_path, device=self.device)
        return model 
    
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
    
    def get_test_pipeline(self):
        """Get data pipeline"""
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
    
    def unify_featmap(self, index, output):
        """Make each element of output to be same as output[index].
        Just used for `get_multi_level_pred` for hack `fpn` output.
        Args:
            output (List[Tensor]): backbone or neck output.
        Return:
            unify_featmap (List[Tensor]): each element of output will same as output[index].
        """
        unify_featmap = []
        for i in range(len(output)):
            # unify_featmap.append(F.interpolate(
            #     output[index],
            #     output[i].shape[2:],
            #     mode='bilinear',
            #     align_corners=False)
            # )
            unify_featmap.append(output[index])
        return unify_featmap
    
    def rescale_pred(self, data_sample, feature_output, index):
        max_w, max_h = feature_output[0].shape[2:]
        this_w, this_h = feature_output[index].shape[2:]
        scale_factor = ((max_w / this_w) + (max_h / this_h)) / 2
        data_sample.pred_instances.bboxes = data_sample.pred_instances.bboxes * scale_factor

        return data_sample

    
    def get_multi_level_pred(self, index, img, rescale: bool = True):
        """Get multi level pred results."""
        
        data = self.get_data_from_img(img=img)
        data_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        with torch.no_grad():
            feature_output = self.model.backbone(data_inputs)
            if self.model.with_neck:
                feature_output = self.model.neck(feature_output)
        
        # make all output be same, so that get the output[index] featmap.
        x = self.unify_featmap(index=index, output=feature_output)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.model.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.model.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        data_sample = self.rescale_pred(batch_data_samples[0], feature_output, index)

        return data_sample
    
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
            # change type of bboxes to Tensor or Numpy
            data_sample.gt_instances.bboxes = data_sample.gt_instances.bboxes.numpy()
            
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
                with torch.no_grad():
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
    
    def _forward(self, feature_type, img):
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
    
    @staticmethod
    def map2uni_space_normalize(normalize_minmax: Tuple[np.ndarray], src: np.ndarray):
        """Normlize src featmap to uni-value MINMAX space which same as target.
        Args:
            normalize_minmax (Tuple[np.ndarray]): (min, max).
            src (np.ndarray): shape must be (H, W).
        Returns:
            output (np.ndarray): make the minuim and maxuim of src equal to targets'.
        """
        target_min = normalize_minmax[0]
        target_max = normalize_minmax[1]

        src = (src - target_min) / (target_max - target_min)
        output = np.clip(src, 0, 1)
        output = output * 255.
        output = np.asarray(output, dtype=np.uint8)

        return output

    def convert_overlay_heatmap(self,
                                feat_map: Union[np.ndarray, torch.Tensor],
                                img: Optional[np.ndarray] = None,
                                alpha: float = 0.5,
                                grey: bool = False,
                                normalize_minmax: Tuple[torch.Tensor] = None) -> np.ndarray:
        """Convert feat_map to heatmap and overlay on image, if image is not None.

        Args:
            feat_map (np.ndarray, torch.Tensor): The feat_map to convert
                with of shape (H, W), where H is the image height and W is
                the image width.
            img (np.ndarray, optional): The origin image. The format
                should be RGB. Defaults to None.
            alpha (float): The transparency of featmap. Defaults to 0.5.

        Returns:
            np.ndarray: heatmap
        """
        assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                    and feat_map.shape[0] in [1, 3])
        if isinstance(feat_map, torch.Tensor):
            feat_map = feat_map.detach().cpu().numpy()

        if feat_map.ndim == 3:
            feat_map = feat_map.transpose(1, 2, 0)

        if normalize_minmax is not None:
            norm_img = self.map2uni_space_normalize(normalize_minmax, feat_map)
        else:
            norm_img = np.zeros(feat_map.shape)
            norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
            norm_img = np.asarray(norm_img, dtype=np.uint8)

        if grey:
            heat_img = np.stack((norm_img,) * 3, -1)
        else:
            heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        if img is not None:
            heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
        return heat_img
    
    def draw_featmap(self,
                     featmap: torch.Tensor,
                     overlaid_image: Optional[np.ndarray] = None,
                     channel_reduction: Optional[str] = 'squeeze_mean',
                     topk: int = 20,
                     arrangement: Tuple[int, int] = (4, 5),
                     resize_shape: Optional[tuple] = None,
                     grey: bool = False,
                     normalize_minmax: Tuple[np.ndarray] = None,
                     alpha: float = 0.5) -> np.ndarray:
        """Draw featmap.

        - If `overlaid_image` is not None, the final output image will be the
          weighted sum of img and featmap.

        - If `resize_shape` is specified, `featmap` and `overlaid_image`
          are interpolated.

        - If `resize_shape` is None and `overlaid_image` is not None,
          the feature map will be interpolated to the spatial size of the image
          in the case where the spatial dimensions of `overlaid_image` and
          `featmap` are different.

        - If `channel_reduction` is "squeeze_mean" and "select_max",
          it will compress featmap to single channel image and weighted
          sum to `overlaid_image`.

        - If `channel_reduction` is None

          - If topk <= 0, featmap is assert to be one or three
            channel and treated as image and will be weighted sum
            to ``overlaid_image``.
          - If topk > 0, it will select topk channel to show by the sum of
            each channel. At the same time, you can specify the `arrangement`
            to set the window layout.

        Args:
            featmap (torch.Tensor): The featmap to draw which format is
                (C, H, W).
            overlaid_image (np.ndarray, optional): The overlaid image.
                Defaults to None.
            channel_reduction (str, optional): Reduce multiple channels to a
                single channel. The optional value is 'squeeze_mean'
                or 'select_max'. Defaults to 'squeeze_mean'.
            topk (int): If channel_reduction is not None and topk > 0,
                it will select topk channel to show by the sum of each channel.
                if topk <= 0, tensor_chw is assert to be one or three.
                Defaults to 20.
            arrangement (Tuple[int, int]): The arrangement of featmap when
                channel_reduction is not None and topk > 0. Defaults to (4, 5).
            resize_shape (tuple, optional): The shape to scale the feature map.
                Defaults to None.
            normalize_minmax (Tuple[np.ndarray]): map value to target MINMAX.
            alpha (Union[int, List[int]]): The transparency of featmap.
                Defaults to 0.5.

        Returns:
            np.ndarray: RGB image.
        """
        import matplotlib.pyplot as plt
        assert isinstance(featmap,
                          torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                          f' but got {type(featmap)}')
        assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                                  f'but got {featmap.ndim}'
        featmap = featmap.detach().cpu()

        if overlaid_image is not None:
            if overlaid_image.ndim == 2:
                overlaid_image = cv2.cvtColor(overlaid_image,
                                              cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems !')
                if resize_shape is None:
                    featmap = F.interpolate(
                        featmap[None],
                        overlaid_image.shape[:2],
                        mode='bilinear',
                        align_corners=False)[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max'], \
                f'Mode only support "squeeze_mean", "select_max", ' \
                f'but got {channel_reduction}'
            if channel_reduction == 'select_max':
                sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
                _, indices = torch.topk(sum_channel_featmap, 1)
                feat_map = featmap[indices]
            else:
                feat_map = torch.mean(featmap, dim=0)
            
            return self.convert_overlay_heatmap(feat_map, overlaid_image, alpha, grey, normalize_minmax)
        elif topk <= 0:
            featmap_channel = featmap.shape[0]
            assert featmap_channel in [
                1, 3
            ], ('The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                f'dimension you input is {featmap_channel}, you can use the'
                ' channel_reduction parameter or set topk greater than '
                '0 to solve the error')
            return self.convert_overlay_heatmap(featmap, overlaid_image, alpha, grey)
        else:
            row, col = arrangement
            channel, height, width = featmap.shape
            assert row * col >= topk, 'The product of row and col in ' \
                                      'the `arrangement` is less than ' \
                                      'topk, please set the ' \
                                      '`arrangement` correctly'

            # Extract the feature map of topk
            topk = min(channel, topk)
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, topk)
            topk_featmap = featmap[indices]

            fig = plt.figure(frameon=False)
            # Set the window layout
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
                axes.imshow(
                    self.convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                            alpha, grey))
            image = img_from_canvas(fig.canvas)
            plt.close(fig)
            return image
        
    def draw_bboxes_with_patch(self, 
                    bboxes: Union[np.ndarray, torch.Tensor],
                    edge_colors: str,
                    ):
        """Draw single or multiple bboxes.
        
        Args:
            bboxes (Union[np.ndarray| torch.Tensor]): bboxes list, (x1, y1, w, h).
            edge_colors (str): such as `#9400D3`

        Return:
            rect (Object): patches object.
        """

        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))
        bboxes = tensor2ndarray(bboxes)

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()
        rects = []
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=edge_colors, facecolor='none')
            rects.append(rect)

        return rects
    
    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples