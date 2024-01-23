import numpy as np
import matplotlib.pyplot as plt
import time
import math
import torch
import os

from attack import BaseAttack
from visualizer import AnalysisVisualizer
from PIL import Image


class ExpVisualizer():
    """Visualizer for get exp results.
    Args:
        use_attack (bool): if it is `True`, then will initialize attacker.
        attack_method (str): `['dcf',]`.
    """
    def __init__(self, cfg_file, ckpt_file, use_attack=False, attacker=None):
        self.use_attack = use_attack 
        self.analysiser = AnalysisVisualizer(cfg_file=cfg_file, ckpt_file=ckpt_file)
        self.visualizer = self.analysiser
        self.runner = self.visualizer
        self.model = self.visualizer.model
        self.dataset = self.visualizer.get_dataset()
        if self.use_attack:
            assert attacker is not None and isinstance(attacker, BaseAttack), \
                f'when `user_attack` is True, `attacker` must be set.'
            setattr(self, 'attacker', attacker)  

        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = self.visualizer.fig_fontsize
    
    @staticmethod
    def get_timestamp():
        return time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    
    @staticmethod
    def cvt_params2savename(params: dict, remain_list: list):
        file_name = ''
        for key in remain_list:
            value = params[key]
            if type(value) is float:
                value = str(value)
                file_name += key + value
            elif type(value) is str:
                file_name += value
            elif type(value) is int:
                file_name += key + str(value)
            elif type(value) is list:
                _v = ''
                for item in value:
                    _v += str(item)
                file_name += key + _v
            elif type(value) is bool:
                file_name += key + str(int(value))

            file_name += '-'

        return file_name

    def show_single_pic_feats(
            self,
            img=None,
            dataset_idx=None, 
            data_sample=None, 
            feature_type='backbone', 
            show_stage=0, 
            arrangement=(4, 6), 
            pic_overlay=False,
            save=True,
            exp_name=None,
            feat_normalize=False):
        """Save `top_k` channels of featuremap of a pic on stage `show_stage`.
        Args:
            img (str): path of img.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            show_stage (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
            arrangement (tuple): row and col of result.
            pic_overlay (bool): whether make ori image overlay with heatmap.
            exp_name (str): experience name.
            feature_type (str): `backbone` or `neck`.
            dataset_idx (int): index of dataset.
            feat_normalize (bool): wheter use global MINMAX to universally normalize featmap.
        """

        assert img is not None or data_sample is not None, \
            f'`img` and `data_sample` cannot be None both.'
        if data_sample is None:
            img_path = img
        else:
            img_path = data_sample.img_path

        feat = self.runner._forward(feature_type=feature_type, img=img_path)

        image = Image.open(img_path)
        _image = np.array(image)
        _feature = feat[show_stage].squeeze(0)

        # just show original feature map
        if not pic_overlay:
            _image = None

        channel, height, width = _feature.shape
        topk = arrangement[0] * arrangement[1]
        row, col = arrangement

        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(_feature, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = _feature[indices]
        
        plt.figure(frameon=False, figsize=(3*col, 2.2*row), dpi=300)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        if feat_normalize:
            normalize_min = torch.min(topk_featmap).detach().cpu().numpy()
            normalize_max = torch.max(topk_featmap).detach().cpu().numpy()
            normalize_minmax = (normalize_min, normalize_max)
        else:
            normalize_minmax = None

        ind = 1
        for i in range(row):
            for j in range(col):
                feat_ind = i * col + j
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                single_feature = topk_featmap[feat_ind].unsqueeze(0)
                feature_map = self.visualizer.draw_featmap(single_feature, _image, channel_reduction='squeeze_mean', normalize_minmax=normalize_minmax)
                plt.xlabel(f"Channel {indices[feat_ind]}")
                plt.imshow(feature_map)
                ind += 1

        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            save_dir = 'records/analysis/topk_featmap'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = f'{img_name}_{show_stage}_{self.get_timestamp()}'
            if dataset_idx:
                save_name = f'{dataset_idx}-{save_name}'
            if exp_name:
                save_name = f'{save_name}_{exp_name}'
            plt.savefig('{}/{}.png'.format(save_dir, save_name))
        else:
            plt.show()


    def show_stage_results(
            self,
            dataset_idx=None,
            img=None,
            data_sample=None,
            save=False,
            grey=False,
            overlaid=False,
            show_thr=0.3,
            show_mlvl_pred=False,
            exp_name=None,
    ):
        """Show `ori_img`, `squeeze_mean_channel(backbone)`, `squeeze_mean_channel(neck)`, `final results of each level of extract_feature`.
        Args:
            img (str): path of img.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            stage (str): string and model map. e.g. `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
            grey (bool): `True` means return greymap, else return heatmap.
            attack (bool): `True` means using attack method.
            show_thr (float): pred result threshold to show.
            show_mlvl_pred (bool): wheter show results repesct to multi-level feature.
            exp_name (str): experience name for save dir.
        """
        assert img is not None or data_sample is not None, \
            f'`img` and `data_sample` cannot be None both.'
        if data_sample is None:
            img_path = img
        else:
            img_path = data_sample.img_path

        backbone_feat = self.runner._forward(feature_type='backbone', img=img_path)

        if self.model.with_neck:
            neck_feat = self.runner._forward(feature_type='neck', img=img_path)
        else:
            neck_feat = []

        output_stages = max(len(neck_feat), len(backbone_feat))  # channels : (256, 512, 1024, 2048) and indices: [0, 1, 2, 3] for fpn

        image = Image.open(img_path)
        _image = np.array(image)
        overlaid_image = _image if overlaid else None

        row, col = [3, output_stages] if self.model.with_neck else [2, output_stages]
        if show_mlvl_pred:
            row += 1

        plt.figure(frameon=False, figsize=(3*col, 2.2*row), dpi=300)
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
                plt.ylabel(f"Image")
                plt.imshow(gt_image)
                plt.xlabel("GT")
            if i == 1:
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                plt.xlabel(f"Pred")
                pred_res = self.visualizer.get_pred(img=img_path)
                final_pred = self.visualizer.draw_dt_gt(
                    name='pred',
                    image=_image,
                    draw_gt=False,
                    data_sample=pred_res,
                    pred_score_thr=show_thr)
                plt.imshow(final_pred) 
            ind += 1
        
        # ====== Second row: backbone ======
        for i in range(col):
            if i < len(backbone_feat):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Backbone")
                _feature = backbone_feat[i].squeeze(0)
                feature_map = self.visualizer.draw_featmap(_feature, overlaid_image, channel_reduction='squeeze_mean', grey=grey)
                plt.xlabel(f"{tuple(_feature.shape)}")
                plt.imshow(feature_map)
            ind += 1

        # ====== Third row: neck ======
        if self.model.with_neck:
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Fpn")
                _feature = neck_feat[i].squeeze(0)
                feature_map = self.visualizer.draw_featmap(_feature, overlaid_image, channel_reduction='squeeze_mean', grey=grey)
                plt.xlabel(f"{tuple(_feature.shape)}")
                plt.imshow(feature_map)
                ind += 1

        # for fr
        # ====== Fourth row: each level pred results of neck ======    
        if show_mlvl_pred:        
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Pred")
                pred_res = self.visualizer.get_multi_level_pred(index=i, img=img_path)
                neck_pred = self.visualizer.draw_dt_gt(
                    name='pred',
                    image=_image,
                    draw_gt=False,
                    data_sample=pred_res,
                    pred_score_thr=show_thr)
                plt.xlabel(f"Fpn {i} pred")
                plt.imshow(neck_pred)
                ind += 1
                
        plt.tight_layout()
        if save:
            img_name = img_path.split('/')[-1].split('.')[0]
            save_dir = 'records/analysis/featmap'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = f'{img_name}_{self.get_timestamp()}'
            if dataset_idx:
                save_name = f'{dataset_idx}-{save_name}'
            if exp_name:
                save_name = f'{save_name}_{exp_name}'
            plt.savefig('{}/{}.png'.format(save_dir, save_name))
        else:
            plt.show()

    def save_heatmap_channel(
        self,
        features,
        preds,
        topk,
        save_dir,
        save_name,
        normalize_features=None,
        grey=False,
        alpha=0.5,
        ):
        """Save topk heatmap channels of each image.
        Args:
            features (torch.Tensor): features of this image.
            preds (np.ndarray): predict results respect to each stage of features (it's a image which can be show or save directly).
            topk (int): amount of channel you want to show.
            save_dir (str): save directory.
            save_name (str): file basename you save.
            normalize_features (torch.Tensor): map features to target tensor value space.
            grey (bool): save grey heatmap or colorful.
            aphla (float): features' capacity.
        """
        edge_len = math.sqrt(topk)
        if edge_len - int(edge_len) != 0.0:
            edge_len += 1
        edge_len = int(edge_len)

        for i in range(len(features)):
            stage_feat = features[i].squeeze(0)
            stage_pred = preds[i]
            normalize_target = normalize_features[i].squeeze(0) if normalize_features is not None else None
            # TODO: 目前normalize_features是没用的
            topk_heat_channel = self.visualizer.draw_featmap(stage_feat, stage_pred, channel_reduction=None, topk=topk, arrangement=(edge_len, edge_len), grey=grey, alpha=alpha, normalize_minmax=None)
            heatmaps = Image.fromarray(topk_heat_channel)
            heatmaps.save(f'{save_dir}/{save_name}-stage{i}.png')

    def show_attack_results(
            self, 
            model_name,
            dataset_idx=None,
            img=None,
            data_sample=None,
            show_features=True,
            show_lvl_preds=True,
            save_analysis=True,
            save=False,
            save_topk_heatmap=False,
            feature_grey=True,
            remain_list=['lr', 'M'],
            show_thr=0.3,
            feat_normalize=True):
        """Show `ori_img`, `noise`, `adv_samples`, `attack_results`.
        Args:
            img (str): path of img.
            model_name (str): name of infer model.
            dataset_idx (int): index of dataset.
            data_sample (DetDataSample): e.g. dataset[0]['data_sample'].
            show_features (bool): whether show the features, includes ori, gt, adv.
            show_lvl_preds (bool): whether show the pred results of every stages.
            save_analysis (bool): whether save analysis result.
            save (bool): whether save pic. if it is True, pic will not be shown when running.
            save_topk_heatmap (bool): whether save topk heatmap.
            feature_grey (bool): whether show grey feature map or heatmap.
            attack_params (dict): attacker parameters.
            remain_list (list): decide which field will be saved in result file name.
            show_thr (float): pred result threshold to show.
            feat_normalize (bool): wheter use global MINMAX to universally normalize featmap.
        """
        assert self.use_attack, \
            f'`use_attack` must be `True` when calling function `show_attack_results.`'
        
        assert img is not None or data_sample is not None, \
            f'`img` and `data_sample` cannot be None both.'
        if data_sample is None:
            img_path = img
        else:
            img_path = data_sample.img_path
            
        exp_name = self.attacker.exp_name
    
        row, col = (1, 5)
        if show_features:
            row += 3
        if show_lvl_preds:
            row += 2
        plt.figure(frameon=False, figsize=(3*col, 2.2*row), dpi=300)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # ====== ori_image & noise & adv_image & pred results =======
        image = Image.open(img_path)
        _image = np.array(image)

        _gt_image = self.visualizer.draw_dt_gt(
            name='attack',
            image=_image,
            draw_gt=True,
            draw_pred=False,
            data_sample=data_sample,
            pred_score_thr=show_thr)
        
        clean_pred = self.visualizer.get_pred(img_path) # 是经过了NMS处理的
        _clean_image = self.visualizer.draw_dt_gt(
            name='attack',
            image=_image,
            draw_gt=False,
            data_sample=clean_pred,
            pred_score_thr=show_thr)

        # base_attack 里存了png
        pertub_img_path, ad_image_path = self.attacker.attack(img_path, data_sample=data_sample)
        ad_result = self.visualizer.get_pred(ad_image_path)

        ad_image = Image.open(ad_image_path)
        _ad_image = np.array(ad_image)
        pertub_img = Image.open(pertub_img_path)
        _pertub_img = np.array(pertub_img)

        # 这里最好采用这种方式，因为PIL如果存jpg格式的话会对图片进行压缩，导致像素会有一定的不一致。（存PNG可以解决这个问题）
        # ad_result, _pertub_img, _ad_image = self.attacker.attack(img_path, save=False)

        ad_pred = self.visualizer.draw_dt_gt(
            name='attack',
            image=_ad_image,
            draw_gt=False,
            data_sample=ad_result,
            pred_score_thr=show_thr)
        
        ind = 1
        # ===== First row: result =======
        image_list = [_gt_image, _clean_image, _pertub_img, _ad_image, ad_pred]
        image_name = ['GT', 'Ori image', 'Pertub noise ', 'Adversarial sample', 'Attack result']
        for i in range(col):
            plt.subplot(row, col, ind)
            plt.xticks([],[])
            plt.yticks([],[])
            if i == 0:
                plt.ylabel(model_name)
            plt.xlabel(f'{image_name[i]}')
            plt.imshow(image_list[i])  
            ind += 1

        if show_features:
            feature_type = self.attacker.feature_type
            # clean backbone featmap
            ori_backbone_feat = self.runner._forward(feature_type=feature_type, img=img_path)
            # target featmap
            gt_backbone_feat = self.attacker.get_target_feature(img=img_path)
            # adv backbone featmap
            adv_backbone_feat = self.runner._forward(feature_type=feature_type, img=ad_image_path)

            normalize_target_minmax = [None for _ in range(col)]
            if feat_normalize:
                for i in range(col):
                    if i < len(ori_backbone_feat):
                        stage_ori_feat = torch.mean(ori_backbone_feat[i].squeeze(0), dim=0).detach().cpu().numpy()
                        stage_gt_feat = torch.mean(gt_backbone_feat[i].squeeze(0), dim=0).detach().cpu().numpy()
                        stage_adv_feat = torch.mean(adv_backbone_feat[i].squeeze(0), dim=0).detach().cpu().numpy()

                        global_min = np.min([stage_ori_feat, stage_gt_feat, stage_adv_feat])
                        global_max = np.max([stage_ori_feat, stage_gt_feat, stage_adv_feat])

                        normalize_target_minmax[i] = (global_min, global_max)

            # ====== Second row: ori backbone ======
            for i in range(col):
                if i < len(ori_backbone_feat):
                    plt.subplot(row, col, ind)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    if i == 0:
                        plt.ylabel(f"ori {feature_type}")
                    _feature = ori_backbone_feat[i].squeeze(0)
                    feature_map = self.visualizer.draw_featmap(_feature, None, channel_reduction='squeeze_mean', grey=feature_grey, normalize_minmax=normalize_target_minmax[i])
                    plt.xlabel(f"stage {i}")
                    plt.imshow(feature_map)
                ind += 1

            # ====== Third row: gt backbone ======
            for i in range(col):
                if i < len(gt_backbone_feat):
                    plt.subplot(row, col, ind)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    if i == 0:
                        plt.ylabel(f"adv gt {feature_type}")
                    _feature = gt_backbone_feat[i].squeeze(0)
                    feature_map = self.visualizer.draw_featmap(_feature, None, channel_reduction='squeeze_mean', grey=feature_grey, normalize_minmax=normalize_target_minmax[i])
                    plt.xlabel(f"stage {i}")
                    plt.imshow(feature_map)
                ind += 1

            # ===== Fourth row: adv backbone =====
            for i in range(col):
                if i < len(adv_backbone_feat):
                    plt.subplot(row, col, ind)
                    plt.xticks([],[])
                    plt.yticks([],[])
                    if i == 0:
                        plt.ylabel(f"adv {feature_type}")
                    _feature = adv_backbone_feat[i].squeeze(0)
                    feature_map = self.visualizer.draw_featmap(_feature, None, channel_reduction='squeeze_mean', grey=feature_grey, normalize_minmax=normalize_target_minmax[i])
                    plt.xlabel(f"stage {i}")
                    plt.imshow(feature_map)
                ind += 1  

        if show_lvl_preds:
            # for fr
            # ====== row 5: each level pred results of clean ======      
            clean_stage_preds = []      
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Pred result")
                pred_res = self.visualizer.get_multi_level_pred(index=i, img=img_path)
                clean_neck_pred = self.visualizer.draw_dt_gt(
                    name='pred',
                    image=_image,
                    draw_gt=False,
                    data_sample=pred_res,
                    pred_score_thr=0)
                clean_stage_preds.append(clean_neck_pred)
                _feature = ori_backbone_feat[i].squeeze(0)
                clean_heatmap_pred = self.visualizer.draw_featmap(_feature, clean_neck_pred, channel_reduction='squeeze_mean', grey=feature_grey, alpha=0.5)
                plt.xlabel(f"clean Fpn {i} pred")
                plt.imshow(clean_heatmap_pred)
                ind += 1

            # ====== row 6: each level pred results adv ======      
            adv_stage_preds = []      
            for i in range(col):
                plt.subplot(row, col, ind)
                plt.xticks([],[])
                plt.yticks([],[])
                if i == 0:
                    plt.ylabel(f"Pred result")
                pred_res = self.visualizer.get_multi_level_pred(index=i, img=ad_image_path)
                adv_neck_pred = self.visualizer.draw_dt_gt(
                    name='pred',
                    image=_ad_image,
                    draw_gt=False,
                    data_sample=pred_res,
                    pred_score_thr=0)
                adv_stage_preds.append(adv_neck_pred)
                _feature = adv_backbone_feat[i].squeeze(0)
                adv_heatmap_pred = self.visualizer.draw_featmap(_feature, adv_neck_pred, channel_reduction='squeeze_mean', grey=feature_grey, alpha=0.5, normalize_minmax=normalize_target_minmax[i])
                plt.xlabel(f"adv Fpn {i} pred")
                plt.imshow(adv_heatmap_pred)
                ind += 1

        plt.tight_layout()

        attack_params = self.attacker.attack_params
        params_str = self.cvt_params2savename(attack_params, remain_list)
        
        if save:
            img_name = os.path.basename(img_path).split('.')[0]

            save_dir = f'records/attack_result/{self.attacker.get_attack_name()}/{exp_name}/{params_str}'
            if save_topk_heatmap:
                save_dir += f'/{img_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img_idx = img_name if dataset_idx is None else dataset_idx

            plt.savefig(f'{save_dir}/{img_idx}-{self.get_timestamp()}.png')
            if save_topk_heatmap:
                self.save_heatmap_channel(ori_backbone_feat, clean_stage_preds, topk=100, save_dir=save_dir, save_name='clean', normalize_features=None, grey=feature_grey, alpha=0.5)
                self.save_heatmap_channel(adv_backbone_feat, adv_stage_preds, topk=100, save_dir=save_dir, save_name='adv', normalize_features=ori_backbone_feat, grey=feature_grey, alpha=0.5)
        else:
            plt.show()

        if save_analysis:
            self.analysiser.save_activate_map_channel_wise(img=ad_image_path, feature_type=feature_type, attack_name=self.attacker.get_attack_name(), exp_name=exp_name + f'/{params_str}')
        
