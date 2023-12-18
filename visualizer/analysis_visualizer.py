import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

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


    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
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
        plt.figure(frameon=False, figsize=(3*col, 3*row), dpi=300)
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

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
            plt.title(f'{feature_type} stages {i}', fontsize=10)
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

    def visualize_bboxes(self, bboxes, image_path, exp_name, customize_str, labels=None, scores=None):
        """Drawing bboxes into the image.
        Args:
            bboxes (torch.Tensor | np.ndarray): format is xyxy, and shape is (N, 4).
            image_path (str): path of image.
            exp_name (str): experience name.
            customize_str (str): as a part of save path, e.g. `{save_dir}/{customize_str}-{image_name}.jpg`
            labels (torch.Tensor | np.ndarray): bboxes' labels, shape is (N, 1). Default `None`.
            scores (torch.Tensor | np.ndarray): bboxes' scores, shape is (N, 1). Default `None`.
        """
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()        
        classes = self.dataset_meta['classes']

        # load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # TODO: Define font for labels and scroes

        # Draw each bbox
        for i, bbox in enumerate(bboxes):
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=1, outline=self.color_panel[0])

            # Prepare label and scroe text
            label_text = ''
            if labels is not None:
                label_text = str(classes[labels[i]])
            if scores is not None:
                label_text += f'{scores[i]: .2f}'

            if label_text:
                draw.text((bbox[0], bbox[1]), label_text, fill="white")
        
        save_path = os.path.join(self.save_dir, exp_name)
        save_img_name = os.path.basename(image_path).split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.imshow(np.array(image))
        plt.axis('off')
        plt.savefig(f'{save_path}/{customize_str}-{save_img_name}-{self.get_timestamp()}.jpg')

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

        # TODO: 可以加一个柱形图，按gtbbox的area排序绘制，每个gt具有不同颜色，柱形高度为gt拥有的proposal数量。
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

    def visualize_intermediate_results(self, r, pertubed_image, round, image_path):
        """Visualize the intermediate results.
        Args:
            r (np.ndarray): noise or gradient of pertubed image of this round, color channel: RGB.
            pertubed_image (np.ndarray): pertubed image for now, color channel: RGB.
            round (int): round of adversarial generating for now.
            image_path (str): path of image.
        """
        assert isinstance(r, np.ndarray) and isinstance(pertubed_image, np.ndarray), \
            f'`r` and `pertubed_image` must be type of `np.ndarray`.'
        
        row, col = (1, 3)
        plt.figure(frameon=False, figsize=(3*col, 3*row), dpi=300)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        # convert RGB to BGR, cuz preprocess has processing of BGR2RGB,
        # that makes consistent result to pass a saved image to model.
        noise_image = r.astype(np.uint8)
        pertubed_image = pertubed_image.astype(np.uint8)
        pertubed_image_bgr = pertubed_image[..., [2, 1, 0]]

        adv_results = self.get_pred(pertubed_image_bgr)
        adv_image = self.draw_dt_gt(
            name='attack',
            image=pertubed_image,
            draw_gt=False,
            data_sample=adv_results,
            pred_score_thr=0.1)

        vis_list = [noise_image, pertubed_image, adv_image]
        title_list = ['r', 'pertubed image', 'attack results']

        for i in range(col):
            plt.subplot(row, col, i + 1)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Step {round} results")
            plt.title(f"{title_list[i]}", fontsize=10)
            plt.imshow(vis_list[i])
        
        save_path = os.path.join(self.save_dir, 'intermediate_results')
        save_img_name = os.path.basename(image_path).split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/{save_img_name}-step{round}-mediate_result.jpg')
