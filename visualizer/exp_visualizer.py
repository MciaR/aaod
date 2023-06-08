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
        self.visualizer = AAVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
        self.runner = self.visualizer
        self.model = self.visualizer.model
        self.dataset = self.visualizer.get_dataset()
        if self.use_attack:
            setattr(self, 'attacker', ELAttack())  
    
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

    def show_stage_results(
            self,
            img=None,
            data_sample=None,
            save=False,
            grey=False,
            overlaid=False,
    ):
        """Show `ori_img`, `squeeze_mean_channel(backbone)`, `squeeze_mean_channel(neck)`, `final results of each level of extract_feature`.
        Args:
            img (str): path of img.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            stage (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
            grey (bool): `True` means return greymap, else return heatmap.
            attack (bool): `True` means using attack method.

        """
        assert img is not None or data_sample is not None, \
            f'`img` and `data_sample` cannot be None both.'
        if data_sample is None:
            img_path = img
        else:
            img_path = data_sample.img_path

        backbone_feat = self.runner._forward(stage='backbone', img=img_path)

        if self.model.with_neck:
            neck_feat = self.runner._forward(stage='neck', img=img_path)
        else:
            neck_feat = []

        output_stages = max(len(neck_feat), len(backbone_feat))  # channels : (256, 512, 1024, 2048) and indices: [0, 1, 2, 3] for fpn

        image = Image.open(img_path)
        _image = np.array(image)
        overlaid_image = _image if overlaid else None

        row, col = (5, output_stages) if self.model.with_neck else (4, output_stages)

        plt.figure(frameon=False, figsize=(10, 8))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        ind = 1
        gt_image = self.visualizer.draw_dt_gt(
            name='gt',
            image=_image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False
        )
        # ====== First row: original pic ======
        for i in range(col):
            if i == 0:
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                plt.ylabel(f"Origin Pic")
                plt.imshow(gt_image)
                plt.title(f"({_image.shape[0]} x {_image.shape[1]})", fontsize=10)
            ind += 1
        
        # ====== Second row: backbone ======
        for i in range(col):
            if i < len(backbone_feat):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Backbone featmap")
                _feature = backbone_feat[i].squeeze()
                feature_map = self.visualizer.draw_featmap(_feature, overlaid_image, channel_reduction='squeeze_mean', grey=grey)
                plt.title(f"{tuple(_feature.shape)}", fontsize=10)
                plt.imshow(feature_map)
            ind += 1

        # ====== Third row: neck ======
        if self.model.with_neck:
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Fpn featmap")
                _feature = neck_feat[i].squeeze()
                feature_map = self.visualizer.draw_featmap(_feature, overlaid_image, channel_reduction='squeeze_mean', grey=grey)
                plt.title(f"{tuple(_feature.shape)}", fontsize=10)
                plt.imshow(feature_map)
                ind += 1

        # ====== Fourth row: each level pred results of neck ======            
        for i in range(col):
            plt.subplot(row, col, ind)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(f"Pred result")
            pred_res = self.visualizer.get_multi_level_pred(index=i, img=img_path)
            neck_pred = self.visualizer.draw_dt_gt(
                name='pred',
                image=_image,
                draw_gt=False,
                data_sample=pred_res,
                pred_score_thr=0.3)
            plt.title(f"Fpn {i} pred", fontsize=10)
            plt.imshow(neck_pred)
            ind += 1

        # ====== Fiveth row: pred ======
        plt.subplot(row, col, ind)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.ylabel(f"Pred result")
        pred_res = self.visualizer.get_pred(img=img_path)
        final_pred = self.visualizer.draw_dt_gt(
            name='pred',
            image=_image,
            draw_gt=False,
            data_sample=pred_res,
            pred_score_thr=0.3)
        plt.title(f"Fpn {i} pred", fontsize=10)
        plt.imshow(final_pred) 

        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            plt.savefig('records/pics/featmap/{}_{}.png'.format(self.get_timestamp(), img_name))
        plt.show()

    def show_attack_results(
            self, 
            model_name,
            img=None,
            data_sample=None,
            save=False):
        """Show `ori_img`, `noise`, `adv_samples`, `attack_results`.
        Args:
            img (str): path of img.
            model_name (str): name of infer model.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
        """
        assert self.use_attack, \
            f'`use_attack` must be `True` when calling function `show_attack_results.`'
        
        assert img is not None or data_sample is not None, \
            f'`img` and `data_sample` cannot be None both.'
        if data_sample is None:
            img_path = img
        else:
            img_path = data_sample.img_path
    
        row, col = (1, 4)
        plt.figure(frameon=False, figsize=(10, 8))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # ====== ori_image & noise & adv_image & pred results =======
        image = Image.open(img_path)
        _image = np.array(image)

        pertub_img_path, ad_image_path = self.attacker.attack(img_path)
        ad_result = self.visualizer.get_pred(ad_image_path)

        ad_image = Image.open(ad_image_path)
        _ad_image = np.array(ad_image)
        pertub_img = Image.open(pertub_img_path)
        _pertub_img = np.array(pertub_img)

        ad_pred = self.visualizer.draw_dt_gt(
            name='attack',
            image=_ad_image,
            draw_gt=False,
            data_sample=ad_result,
            pred_score_thr=0.3)
        
        image_list = [_image, _pertub_img, _ad_image, ad_pred]
        image_name = ['Ori image', 'Pertub noise ', 'Adversarial sample', 'Attack result']
        for i in range(col):
            plt.subplot(row, col, i + 1)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(model_name)
            plt.title(f'{image_name[i]}', fontsize=10)
            plt.imshow(image_list[i])  

        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            plt.savefig('records/pics/attack/{}_{}_{}.png'.format('attack', self.get_timestamp(), img_name))
        plt.show()
