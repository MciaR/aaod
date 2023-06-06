from visualizer import ExpVisualizer


def parse_args():
    pass


if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/faster_rcnn_r101_fpn_coco.py'
    checkpoint_file = 'pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'

    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file, use_attack=True)
    show_layer = 3 # (256, 512, 1024, 2048) channels
    top_k = 100
    pic_overlay = False

    # vis.show_single_pic_feats(img=img, show_layer=3, top_k=top_k, pic_overlay=pic_overlay)
    dataset = vis.dataset
    vis.show_cmp_results(data_samples=dataset[0]['data_samples'], stage='backbone', save=False, grey=True)
    # vis.show_cmp_results(data_samples=dataset[0]['data_samples'], stage='backbone', save=True, grey=True)
    # vis.show_cmp_results(img=img)
    # dataset = vis.dataset
    # for data in dataset:
    #     print(data)
    #     print(data)