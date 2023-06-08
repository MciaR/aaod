from visualizer import ExpVisualizer


def parse_args():
    pass


if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/fr_vgg16_coco.py'
    checkpoint_file = 'pretrained/fr_vgg16_0607_ep1.pth'

    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file, use_attack=False)
    show_layer = 3 # (256, 512, 1024, 2048) channels
    top_k = 100
    pic_overlay = False

    # vis.show_single_pic_feats(img=img, show_layer=3, top_k=top_k, pic_overlay=pic_overlay)
    dataset = vis.dataset
    vis.show_stage_results(data_sample=dataset[0]['data_samples'], save=True, grey=False)
    # vis.show_attack_results(model_name="FR_R101_COCO", data_sample=dataset[0]['data_samples'], save=True)
