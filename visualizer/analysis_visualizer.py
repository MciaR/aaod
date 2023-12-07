import os
import torch
import matplotlib.pyplot as plt
import numpy as np

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

    def save_activate_map_channel_wise(
            self,
            img=None,
            data_sample=None,
            data_idx=None,
            feature_type='backbone',
            figure_name=None,):
        """Save channel wise mean activate value.
        Args:
            img (str): imaga path.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            data_idx (int): index of data_sample.
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f'{save_path}/{data_idx}.jpg')


