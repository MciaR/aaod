import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

from visualizer import AAVisualizer


class AnalysisVisualizer(AAVisualizer):
    """Analysis data for Adversarial Attack.
    Args:
        cfg_file (str):
        ckpt_file (str):  
    """
    def __init__(
            self,
            cfg_file, 
            ckpt_file, 
            save_dir: str = 'records/analysis',
            name: str = 'analysis_visualizer'):
        super().__init__(cfg_file=cfg_file, ckpt_file=ckpt_file, name=name)
        self.save_dir = save_dir
        self.color_panel = [
            '#FFB300', '#803E75', '#FF6800', '#A6BDD7', '#C10020', '#CEA262', '#817066',
            '#007D34', '#F6768E', '#00538A', '#FF7A5C', '#53377A', '#FF8E00', '#B32851',
            '#F4C800', '#7F180D', '#93AA00', '#593315', '#F13A13', '#232C16', '#FF6EFF',
            '#FFFF99', '#FF1CAE', '#FFCABD', '#B0DD16', '#6B440B', '#4D1E01', '#587E0E',
            '#71B2C9', '#AD5D5D', '#92C7C7', '#88FFCC', '#F9E555', '#D2386C', '#AB92BF',
            '#FFFF00', '#C2FFED', '#A1CAF1', '#F99379', '#604E97', '#F6A600', '#B3446C',
            '#DCD300', '#8DB600', '#654522', '#E25822', '#2B3D26', '#F2F3F4', '#222222',
            '#F1E788', '#FFA6C9', '#B2CEEE', '#5DA493', '#FFC800', '#7A89B8', '#E68FAC',
            '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17',
            '#8DB600', '#654522', '#E25822', '#2B3D26', '#F2F3F4', '#222222', '#F1E788',
            '#FFA6C9', '#B2CEEE', '#5DA493', '#FFC800', '#7A89B8', '#E68FAC', '#0067A5',
            '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600',
            '#654522', '#E25822', '#2B3D26', '#F2F3F4', '#222222', '#F1E788', '#FFA6C9',
            '#B2CEEE', '#5DA493', '#FFC800', '#7A89B8', '#E68FAC', '#0067A5', '#F99379',
            '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522'
        ]

        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = self.fig_fontsize

    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
    def bbox_cxcywh_to_xyxy(self, bbox: torch.Tensor) -> torch.Tensor:
        """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

        Args:
            bbox (Tensor): Shape (n, 4) for bboxes.

        Returns:
            Tensor: Converted bboxes.
        """
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
        return torch.cat(bbox_new, dim=-1)

    def rescale_bboxes(self, pred_bboxes, img_meta):
        """Rescale bboxes to original pic scale.
        Args:
            pred_bboxes (torch.Tensor): pred_bboxes, shape is (N, 4).
            img_meta (dict): metainfo of image.
        Returns:
            rescaled_bboxes (torch.Tensor): rescaled bboxes.
        """
        img_shape = img_meta['img_shape']

        det_bboxes = self.bbox_cxcywh_to_xyxy(pred_bboxes)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        det_bboxes /= det_bboxes.new_tensor(
            img_meta['scale_factor']).repeat((1, 2))
        
        return det_bboxes
    
    def analysis_adv_and_clean_mean_activation(
            self,
            feature_type,
            exp_name,
            adv_images_path,
            clean_image_path,
            attack_name=None,
            figure_name=None,):
        """Save channel wise mean activate value.
        Args:
            adv_images_path (str): adv image set path.
            clean_image_path (str): clean image set path.
            attack_name (str): attack method, default `None`.
            exp_name (str): exp name
            feature_type (str): 'backbone' or 'neck'.
        """

        # plt.figure(frameon=False, figsize=(4*col, 4*row), dpi=1200)
        # plt.subplots_adjust(wspace=0.3)

        assert adv_images_path, \
            f'adv_images_path can not be empty.'
        
        # -------------- adv images -----------------
        print('Processing adversarial examples ...')
        adv_mean_activation_list = []
        adv_image_list = os.listdir(adv_images_path)

        for image_name in tqdm(adv_image_list):
            img_path = os.path.join(adv_images_path, image_name)

            outputs = self._forward(feature_type=feature_type, img=img_path) # (stages, 1, C, H, W)
            stage_channel_mean = []
            for i in range(len(outputs)):
                feature_map = outputs[i]
                N, C, H, W = feature_map.shape
                assert N == 1, \
                    f'Batch_size must be 1.'
                
                sample_features = feature_map[0]
                channel_wise_mean = torch.mean(sample_features.view(C, -1), dim=-1) # (C, )
                channel_wise_mean = channel_wise_mean.cpu().detach().numpy()
                stage_channel_mean.append(channel_wise_mean)
            
            adv_mean_activation_list.append(stage_channel_mean)

        # get average mean activation across images.
        adv_stage_y = np.array(adv_mean_activation_list).mean(axis=0)

        # -------------- clean images -----------------
        print('Processing adversarial examples ...')
        clean_mean_activation_list = []
        clean_image_list = os.listdir(clean_image_path)
        
        for image_name in tqdm(clean_image_list):
            img_path = os.path.join(clean_image_path, image_name)

            outputs = self._forward(feature_type=feature_type, img=img_path) # (stages, 1, C, H, W)
            stage_channel_mean = []
            for i in range(len(outputs)):
                feature_map = outputs[i]
                N, C, H, W = feature_map.shape
                assert N == 1, \
                    f'Batch_size must be 1.'
                
                sample_features = feature_map[0]
                channel_wise_mean = torch.mean(sample_features.view(C, -1), dim=-1) # (C, )
                channel_wise_mean = channel_wise_mean.cpu().detach().numpy()
                stage_channel_mean.append(channel_wise_mean)
            
            clean_mean_activation_list.append(stage_channel_mean)

        # get average mean activation across images.
        clean_stage_y = np.array(clean_mean_activation_list).mean(axis=0)

        row, col = (1, len(clean_stage_y))
        plt.figure(frameon=False, figsize=(10*col, 4*row), dpi=300)
        plt.subplots_adjust(wspace=0.3)

        for i in range(len(clean_stage_y)):
            y1 = clean_stage_y[i] # (256, )
            y2 = adv_stage_y[i]
            x = np.linspace(0, len(y1), len(y1), endpoint=False)

            if i == 0 and figure_name:
                plt.ylabel(figure_name)
            plt.subplot(row, col, i + 1) # subplot index start from 1
            plt.ylim(-2, 2)
            plt.title(f'{feature_type} stages {i}')
            plt.bar(x, y1, label='clean', color='#1565C0')
            plt.bar(x, y2, label='adv.', color='#7B1FA2', alpha=0.5)
            
        save_path = os.path.join(self.save_dir, 'adv_and_clean_mean_activation')
        if attack_name:
            save_path = os.path.join(save_path, attack_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f'{save_path}/{exp_name}.pdf')
        plt.clf()
        
    def save_activate_map_channel_wise(
            self,
            img=None,
            data_sample=None,
            data_idx=None,
            feature_type='backbone',
            attack_name=None,
            exp_name=None,
            figure_name=None,):
        """Save channel wise mean activate value.
        Args:
            img (str): imaga path.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            data_idx (int): index of data_sample.
            attack_name (str): attack method, default `None`.
            exp_name (str): exp name, default `None`.
            feature_type (str): 'backbone' or 'neck'.
        """
        assert img or data_sample, \
            f'`img` or `data_sample` cannot be `None` either.'
        
        if data_sample is not None:
            assert data_idx is not None, \
                f'if `data_sample` is not `None`, then `data_idx` must be given.'
        
        if img is None:
            img_path = data_sample.img_path
        else:
            img_path = img

        outputs = self._forward(feature_type=feature_type, img=img_path) # (stages, 1, C, H, W)

        row, col = (1, len(outputs))
        plt.figure(frameon=False, figsize=(4*col, 4*row), dpi=300)
        plt.subplots_adjust(wspace=0.3)

        for i in range(len(outputs)):
            feature_map = outputs[i]
            N, C, H, W = feature_map.shape
            assert N == 1, \
                f'Batch_size must be 1.'
            
            sample_features = feature_map[0]
            channel_wise_mean = torch.mean(sample_features.view(C, -1), dim=-1) # (C, )
            y = channel_wise_mean.cpu().detach().numpy()
            x = np.linspace(0, C, C, endpoint=False)

            if i == 0 and figure_name:
                plt.ylabel(figure_name)
            plt.subplot(row, col, i + 1) # subplot index start from 1
            plt.ylim(-2, 2)
            plt.title(f'{feature_type} stages {i}')
            plt.scatter(x, y, s=10)

        save_path = os.path.join(self.save_dir, 'activate_mean_by_channel')

        if attack_name:
            save_path = os.path.join(save_path, attack_name)
        if exp_name:
            save_path = os.path.join(save_path, exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_img_name = os.path.basename(img_path).split('.')[0]
        if data_sample is not None:
            save_img_name = str(data_idx) + '-' + save_img_name
        plt.savefig(f'{save_path}/{save_img_name}.jpg')
        plt.clf()

    def visualize_bboxes(self, bboxes, image_path, exp_name=None, customize_str=None, labels=None, scores=None, save=True, distinguished_color=False):
        """Drawing bboxes into the image.
        Args:
            bboxes (torch.Tensor | np.ndarray): format is xyxy, and shape is (N, 4).
            image_path (str): path of image.
            exp_name (str): experience name. Default `None`.
            customize_str (str): as a part of save path, e.g. `{save_dir}/{customize_str}-{image_name}.jpg`. Default `None`.
            labels (torch.Tensor | np.ndarray): bboxes' labels, shape is (N, 1). Default `None`.
            scores (torch.Tensor | np.ndarray): bboxes' scores, shape is (N, 1). Default `None`.
            save (bool): whether save the drawing result. Default `True`.
            distinguished_color (bool): wheter assign different color to different labels. Defalut `False`.
        """
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if distinguished_color:
            assert labels is not None, \
                f'`labels` must be given if `distinguished_color` is `True`.'        
        classes = self.dataset_meta['classes']
        # load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # TODO: Define font for labels and scroes

        # Draw each bbox
        for i, bbox in enumerate(bboxes):

            if distinguished_color:
                bbox_color = self.color_panel[labels[i]]
                text_color = bbox_color
            else:
                bbox_color = self.color_panel[0]
                text_color = "white"

            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=1, outline=bbox_color)

            # Prepare label and scroe text
            label_text = ''
            if labels is not None:
                label_text = str(classes[labels[i]])
            if scores is not None:
                label_text += f'{scores[i]: .2f}'

            if label_text:
                draw.text((bbox[0], bbox[1]), label_text, fill=text_color)
        
        if save:
            plt.clf()
            plt.figure(figsize=(6, 4), dpi=300)

            save_path = os.path.join(self.save_dir, exp_name) if exp_name else self.save_dir
            save_img_name = os.path.basename(image_path).split('.')[0]
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.imshow(np.array(image))
            plt.axis('off')
            if customize_str:
                plt.savefig(f'{save_path}/{customize_str}-{save_img_name}-{self.get_timestamp()}.jpg')
            else:
                plt.savefig(f'{save_path}/{save_img_name}-{self.get_timestamp()}.jpg')
            plt.clf()

        return np.array(image)


    def visualize_category_amount(self, proposal_bboxes, gt_bboxes, proposal2gt_idx, image_path, exp_name, draw_proposals=True):
        """Visualize each gt bboxes and its active proposal amount.
        Args:
            proposal_bboxes (torch.Tensor | np.ndarray): active proposal bboxes set, shape is (N, 4).
            gt_bboxes (torch.Tensor | np.ndarray): gt bboxes set, shape is (M, 4).
            proposal2gt_idx (torch.Tesnor | np.ndarray): the index of corresponding gt of each proposals, shape is (N, 1).
            image_path (str): path of image.
            exp_name (str): experience name.
            draw_proposals (bool): whether show the proposal bboxes. Default `True`.
        """
        if proposal_bboxes is not None and isinstance(proposal_bboxes, torch.Tensor):
            proposal_bboxes = proposal_bboxes.detach().cpu().numpy()
        if gt_bboxes is not None and isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = gt_bboxes.detach().cpu().numpy()
        if proposal2gt_idx is not None and isinstance(proposal2gt_idx, torch.Tensor):
            proposal2gt_idx = proposal2gt_idx.detach().cpu().numpy()

        gt_propsal_nums = Counter(proposal2gt_idx)
        gt_colors = []

        # load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Draw each bbox
        if draw_proposals:
            for bbox in proposal_bboxes:
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=1, outline=self.color_panel[0])

        for i, gts in enumerate(gt_bboxes):
            draw.rectangle([(gts[0], gts[1]), (gts[2], gts[3])], width=2, outline=self.color_panel[i + 1]) # proposal_bboxes 占了第一个颜色
            draw.text((gts[0], gts[1]), f'p_num: {str(gt_propsal_nums[i])}', fill=self.color_panel[i + 1])
            gt_colors.append(self.color_panel[i + 1])
 
        save_path = os.path.join(self.save_dir, exp_name)
        save_img_name = os.path.basename(image_path).split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.imshow(np.array(image))
        plt.axis('off')
        plt.savefig(f'{save_path}/{save_img_name}-{self.get_timestamp()}.jpg')

        # 柱形图，按gtbbox的area排序绘制，每个gt具有不同颜色，柱形高度为gt拥有的proposal数量。
        plt.clf()
        gt_bboxes_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_bboxes_areas = gt_bboxes_areas.astype(int)
        sorted_indices = np.argsort(gt_bboxes_areas)
        sorted_gt_bboxes_areas = [gt_bboxes_areas[ind] for ind in sorted_indices]
        sorted_gt_nums = [gt_propsal_nums[ind] for ind in sorted_indices]
        sorted_colors = [gt_colors[ind] for ind in sorted_indices]

        plt.figure(figsize=(10, 8), dpi=300)
        bars = plt.bar(np.arange(len(gt_bboxes)), sorted_gt_nums, color=sorted_colors, tick_label=sorted_gt_bboxes_areas)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        plt.xlabel('Bboxes')
        plt.ylabel('proposal_nums')
        plt.xticks(rotation=45)
        plt.title('Proposals amount of gt bboxes')
        plt.savefig(f'{save_path}/{save_img_name}-bar_figure-{self.get_timestamp()}.jpg')
        plt.clf()

    def visualize_intermediate_results(self, r, r_total, pertubed_image, attack_proposals, customize_str, image_path, exp_name):
        """Visualize the intermediate results.
        Args:
            r (np.ndarray): noise or gradient of pertubed image of this round, color channel: RGB.
            r_total (np.ndarray): the whole noise added to clean image, color channel: RGB.
            pertubed_image (np.ndarray): pertubed image for now, color channel: RGB.
            attack_proposals (torch.Tensor | np.ndarray): 
            customize_str (Any): as a part of save path, e.g. `{save_dir}/{customize_str}-{image_name}.jpg`.
            image_path (str): path of image.
            exp_name (str): name of experience.
        """
        assert isinstance(r, np.ndarray) and isinstance(pertubed_image, np.ndarray), \
            f'`r` and `pertubed_image` must be type of `np.ndarray`.'
        
        row, col = (1, 5)
        plt.figure(frameon=False, figsize=(3*col, 3*row), dpi=300)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        # convert RGB to BGR, cuz preprocess has processing of BGR2RGB,
        # that makes consistent result to pass a saved image to model.
        r = r.astype(np.uint8)
        r_total = r_total.astype(np.uint8)
        pertubed_image = pertubed_image.astype(np.uint8)
        pertubed_image_bgr = pertubed_image[..., [2, 1, 0]]
        attack_proposals = self.visualize_bboxes(attack_proposals, image_path, save=False)

        adv_results = self.get_pred(pertubed_image_bgr)
        adv_image = self.draw_dt_gt(
            name='attack',
            image=pertubed_image,
            draw_gt=False,
            data_sample=adv_results,
            pred_score_thr=0.1)

        vis_list = [attack_proposals, r, r_total, pertubed_image, adv_image]
        title_list = ['attack proposals', 'r', 'pertub', 'pertubed image', 'attack results']

        for i in range(col):
            plt.subplot(row, col, i + 1)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Step {customize_str} results")
            plt.xlabel(f"{title_list[i]}")
            plt.imshow(vis_list[i])
        
        save_path = os.path.join(self.save_dir, 'intermediate_results', exp_name)
        save_img_name = os.path.basename(image_path).split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/{save_img_name}-step{customize_str}-mediate_result.jpg')
        plt.clf()
