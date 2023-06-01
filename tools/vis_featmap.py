import numpy as np
import matplotlib.pyplot as plt
import time

from utils import AAVisualizer
from PIL import Image


def parse_args():
    pass


class ExpVisualizer():
    """Visualizer for get exp results."""
    def __init__(self, cfg_file, ckpt_file):
        self.visualizer = AAVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
        self.dataset = self.visualizer.get_dataset()
    
    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%dT%H%M%S", time.gmtime())

    def show_single_pic_feats(self, img, show_layer=0, top_k = 100, pic_overlay=False):
        """Show `top_k` channels of featuremap of a pic."""

        feat = self.visualizer._forward(self.visualizer.model.backbone, img=img)

        image = Image.open(img)
        _image = np.array(image)
        _feature = feat[show_layer].squeeze()
        _resize_shape = (_image.shape[0], _image.shape[1])

        # just show original feature map
        if not pic_overlay:
            _image = None
            _resize_shape = None

        heatmap = self.visualizer.draw_featmap(_feature, _image, channel_reduction=None, arrangement=(10, 10), topk=top_k, with_text=False, alpha=0.5, resize_shape=_resize_shape)
        self.visualizer.show(img=heatmap)

    def show_cmp_results(
            self, 
            img=None,
            data_samples=None,
            save=False,
            stage='backbone'):
        """Show `ori_img`, `squeeze_mean_channel(all output stages)`, `upsample and overlay heatmap img(all output stages)`.
        Args:
            img (str): path of img.
            data_samples (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            model (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'fpn'` - `model.fpn`.

        """
        assert img is not None or data_samples is not None, \
            f'`img` and `data_samples` cannot be None both.'
        if data_samples is None:
            img_path = img
        else:
            img_path = data_samples.img_path
        

        feat = self.visualizer._forward(self.visualizer.model.backbone, img=img_path)
        if stage == 'neck':
            feat = self.visualizer.model.neck(feat)
        elif stage == 'head':
            feat = self.visualizer.model.neck(feat)
            feat = self.visualizer.model.head(feat)
        output_stages = len(feat)  # channels : (256, 512, 1024, 2048) and indices: [0, 1, 2, 3] for fpn

        image = Image.open(img_path)
        _image = np.array(image)

        row, col = (3, output_stages)

        _resize_shape = (_image.shape[0], _image.shape[1])

        plt.figure(frameon=False, figsize=(10, 8))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        ind = 1
        gt_image = self.visualizer.draw_dt_gt(
            name='gt',
            image=_image,
            data_sample=data_samples,
            draw_gt=True,
            draw_pred=False
        )
        # ====== First row: original pic ======
        for i in range(col):
            plt.subplot(row, col, ind)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Origin Pic")
            plt.imshow(gt_image)
            plt.title(f"({_image.shape[0]} x {_image.shape[1]})", fontsize=10)
            ind += 1
        
        # ====== Second row: squeeze_mean_channels ======
        for i in range(col):
            plt.subplot(row, col, ind)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Feature map (mean)")
            _feature = feat[i].squeeze()
            feature_map = self.visualizer.draw_featmap(_feature, channel_reduction='squeeze_mean')
            plt.title(f"{tuple(_feature.shape)}", fontsize=10)
            plt.imshow(feature_map)
            ind += 1

        # ====== Third row: upsample and overlay heatmap img ======
        for i in range(col):
            plt.subplot(row, col, ind)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Heatmap")
            _feature = feat[i].squeeze()
            heatmap = self.visualizer.draw_featmap(_feature, _image, channel_reduction='squeeze_mean', alpha=0.5, resize_shape=_resize_shape)
            result = self.visualizer.get_pred(img_path)
            result_heatmap = self.visualizer.draw_dt_gt(
                name='result',
                image=heatmap,
                draw_gt=False,
                data_sample=result
            )
            plt.title(f"({_image.shape[0]} x {_image.shape[1]})", fontsize=10)
            plt.imshow(result_heatmap)
            ind += 1
        
        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            plt.savefig('records/pics/featmap/{}_{}_{}.png'.format(self.get_timestamp(), img_name, stage))
        else:
            plt.show()


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
    vis.show_cmp_results(data_samples=dataset[0]['data_samples'], stage='neck', save=True)
    # vis.show_cmp_results(img=img)
    # dataset = vis.dataset
    # for data in dataset:
    #     print(data)
    #     print(data)



