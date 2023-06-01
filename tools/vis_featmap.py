from visualizer import ExpVisualizer


def parse_args():
    pass


if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/faster_rcnn_r101_fpn.py'
    checkpoint_file = 'pretrained/resnet/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
    img = 'data/coco2014/images/val2014/COCO_val2014_000000000042.jpg'

    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file)
    show_layer = 3 # (256, 512, 1024, 2048) channels
    top_k = 100
    pic_overlay = False

    # vis.show_single_pic_feats(img=img, show_layer=3, top_k=top_k, pic_overlay=pic_overlay)
    dataset = vis.dataset
    vis.show_cmp_results(data_samples=dataset[0]['data_samples'], stage='backbone')
    # vis.show_cmp_results(img=img)
    # dataset = vis.dataset
    # for data in dataset:
    #     print(data)
    #     print(data)