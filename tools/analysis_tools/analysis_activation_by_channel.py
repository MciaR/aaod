import os

from visualizer import AnalysisVisualizer
from setting import get_model

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


if __name__ == "__main__":
    model_name = "FR_R101"
    dataset_name = "COCO"

    config_file, ckpt_file = get_model(model_name, dataset_name)
    analysiser = AnalysisVisualizer(cfg_file=config_file, ckpt_file=ckpt_file)
    dataset = analysiser.get_dataset()

    # for i in range(1):
    #     analysiser.save_activate_map_channel_wise(data_sample=dataset[i]['data_samples'], data_idx=i, feature_type='neck', exp_name='clean')

    analysiser.analysis_adv_and_clean_mean_activation(feature_type='neck', adv_images_path='data/coco2017/FRMR/adv/FR_R101', clean_image_path='data/coco2017/images/val2017', attack_name='FAMR', exp_name='adv')