import os

from visualizer import AnalysisVisualizer

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


if __name__ == '__main__':
    config_file = 'configs/faster_rcnn_r101_fpn_coco.py'
    checkpoint_file = 'pretrained/fr_r101_coco_0394.pth'

    analysiser = AnalysisVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file)
    dataset = analysiser.get_dataset()

    for i in range(9):
        analysiser.save_activate_map_channel_wise(data_sample=dataset[i]['data_samples'], data_idx=i, feature_type='neck', exp_name='clean')
