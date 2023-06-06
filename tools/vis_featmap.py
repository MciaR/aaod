from visualizer import ExpVisualizer


def parse_args():
    pass


if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/faster_rcnn_r101_dcn_c3_c5_fpn_coco.py'
    checkpoint_file = 'pretrained/faster_rcnn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth'

    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file, use_attack=True)
    show_layer = 3 # (256, 512, 1024, 2048) channels
    top_k = 100
    pic_overlay = False

    # vis.show_single_pic_feats(img=img, show_layer=3, top_k=top_k, pic_overlay=pic_overlay)
    dataset = vis.dataset
    vis.show_stage_results(data_samples=dataset[0]['data_samples'], save=True, grey=True)
    vis.show_attack_results(model_name="FR_R101_COCO", data_samples=dataset[0]['data_samples'], save=True)
