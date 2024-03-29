import os

from visualizer import ExpVisualizer
from setting import get_model

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    model_name = "FR_R101"
    dataset_name = "COCO"
    
    config_file, ckpt_file = get_model(model_name, dataset_name)
    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=ckpt_file, use_attack=False)
    dataset = vis.dataset
    data_idx = 0

    vis.show_stage_results(img='data/coco2017/images/val2017/000000090284.jpg', data_sample=None, save=True, show_mlvl_pred=False, exp_name=dataset_name + '_' +  model_name, dataset_idx=data_idx)
    # vis.show_single_pic_feats(data_sample=dataset[data_idx]['data_samples'], feature_type='neck', arrangement=(3, 4), show_stage=3, pic_overlay=True, save=True, dataset_idx=data_idx, exp_name=dataset_name, feat_normalize=True)


    