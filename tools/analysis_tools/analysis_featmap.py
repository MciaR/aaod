from visualizer import ExpVisualizer

from setting import get_model

if __name__ == "__main__":
    model_name = "FR_R101"
    dataset_name = "COCO"
    
    config_file, ckpt_file = get_model(model_name, dataset_name)
    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=ckpt_file, use_attack=False)
    dataset = vis.dataset

    


    