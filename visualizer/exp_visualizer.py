import numpy as np
import matplotlib.pyplot as plt
import time

from attack import ELAttack
from visualizer import AAVisualizer
from PIL import Image


class ExpVisualizer():
    """Visualizer for get exp results.

    init:
        use_attack (bool): if it is `True`, then will be initialize `ELAttack`.
    """
    def __init__(self, cfg_file, ckpt_file, use_attack=False):
        self.use_attack = use_attack
        if self.use_attack:
            setattr(self, 'attack', ELAttack(cfg_file=cfg_file, ckpt_file=ckpt_file))   
            self.visualizer = self.attack.visualizer
            self.runner = self.attack
            self.model = self.attack.model
        else:
            self.visualizer = AAVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
            self.runner = self.visualizer
            self.model = self.visualizer.model

        self.dataset = self.visualizer.get_dataset()
    
    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%dT%H%M%S", time.gmtime())

    def show_single_pic_feats(self, img, show_layer=0, top_k = 100, pic_overlay=False):
        """Show `top_k` channels of featuremap of a pic."""

        feat = self.runner._forward(self.model.backbone, img=img)

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
            stage='backbone',
            grey=False):
        """Show `ori_img`, `squeeze_mean_channel(all output stages)`, `upsample and overlay heatmap img(all output stages)`.
        Args:
            img (str): path of img.
            data_samples (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            model (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'fpn'` - `model.fpn`.
            grey (bool): `True` means return greymap, else return heatmap.
            attack (bool): `True` means using attack method.

        """
        assert img is not None or data_samples is not None, \
            f'`img` and `data_samples` cannot be None both.'
        if data_samples is None:
            img_path = img
        else:
            img_path = data_samples.img_path
        
        feat = self.runner._forward(self.model.backbone, img=img_path)
        if stage == 'neck':
            feat = self.model.neck(feat)
        # TODO: add head forward
        # elif stage == 'head':
        #     feat = self.visualizer.model.neck(feat)
        #     feat = self.visualizer.model.head(feat)

        output_stages = len(feat)  # channels : (256, 512, 1024, 2048) and indices: [0, 1, 2, 3] for fpn


        image = Image.open(img_path)
        _image = np.array(image)

        row, col = (4, output_stages) if self.use_attack else (3, output_stages)

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
            feature_map = self.visualizer.draw_featmap(_feature, channel_reduction='squeeze_mean', grey=grey)
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
            heatmap = self.visualizer.draw_featmap(_feature, _image, channel_reduction='squeeze_mean', alpha=0.5, resize_shape=_resize_shape, grey=grey)
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
        
        # ====== Fourth row: attack results ======
        if self.use_attack:
            ad_image_path = self.runner.attack(img_path)
            ad_result = self.visualizer.get_pred(ad_image_path)

            ad_image = Image.open(ad_image_path)
            ad_image = np.array(ad_image)

            ad_pred = self.visualizer.draw_dt_gt(
                name='attack',
                image=ad_image,
                draw_gt=False,
                data_sample=ad_result,
                pred_score_thr=0.3)
        
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Attack results")
                plt.title(f"({ad_image.shape[0]} x {ad_image.shape[1]})", fontsize=10)
                plt.imshow(ad_pred)
                ind += 1        

        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            plt.savefig('records/pics/featmap/{}_{}_{}.png'.format(self.get_timestamp(), img_name, stage))
        else:
            plt.show()
