import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

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
        color = ['#FFB6C1', '#D8BFD8', '#9400D3', '#483D8B', '#4169E1', '#00FFFF','#B1FFF0','#ADFF2F','#EEE8AA','#FFA500','#FF6347']

        # load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # TODO: Define font for labels and scroes

        # Draw each bbox
        for i, bbox in enumerate(bboxes):
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], width=1, outline=color[0])

            # Prepare label and scroe text
            label_text = ''
            if labels is not None:
                label_text = str(labels[i])
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

