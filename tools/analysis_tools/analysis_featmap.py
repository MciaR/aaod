import os

from visualizer import ExpVisualizer
from setting import get_model

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    model_name = "FR_R101"
    dataset_name = "COCO"
    
    config_file, ckpt_file = get_model(model_name, dataset_name)
    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=ckpt_file, fig_fontsize=18, use_attack=False)
    dataset = vis.dataset

    vis.show_stage_results(data_sample=dataset[0]['data_samples'], save=True)


    