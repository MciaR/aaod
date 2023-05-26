from utils import AAVisualizer


def parse_args():
    pass

if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/faster_rcnn_r101_fpn.py'
    checkpoint_file = 'pretrained/resnet/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
    imgs = ['data/coco2014/images/val2014/COCO_val2014_000000000042.jpg']
    vis = AAVisualizer(cfg_file=config_file, ckpt_path=checkpoint_file)


    feat = vis._forward(vis.model.backbone, imgs=imgs)[0]
    heatmap = vis.draw_featmap(feat[0].squeeze(), channel_reduction=None, topk=10)
    vis.show(img=heatmap)

