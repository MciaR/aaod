import torch
import torch.nn.functional as F
import numpy as np
import os
import mmcv
import copy

from typing import List
from attack import BaseAttack, FMRAttack, EFMRAttack
from visualizer import AnalysisVisualizer
from PIL import Image
from mmengine.structures import InstanceData
from torch.optim.lr_scheduler import StepLR


class FusionAttack(BaseAttack):
    """Fusion Attack (i.e. Enhanced Feature Representation Mean Regression Attack). 
    Args:
        fmr_params (dict): params dict of FMR.
        edag_params (dict): params dict of EDAG.
        exp_name (str): experience name. Default `None`.
    """
    def __init__(self, 
                 cfg_file, 
                 ckpt_file,
                 fmr_params,
                 edag_params,
                 M,
                 fmr_weight,
                 exp_name=None,                
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, exp_name=exp_name, cfg_options=edag_params['cfg_options'],
                         attack_params=dict(fmr=fmr_params, edag=edag_params, M=M, fmr_weight=fmr_weight))
        self.fmr = FMRAttack(cfg_file=cfg_file, ckpt_file=ckpt_file, device=device, **fmr_params)
        self.edag = EFMRAttack(cfg_file=cfg_file, ckpt_file=ckpt_file, device=device, **edag_params)
        self.feature_type = self.fmr.feature_type # just for exp_visualizer
    
    def get_target_feature(self, img):
        return self.fmr.get_target_feature(img)

    def generate_adv_samples(self, x, data_sample, log_info=True):
        """Attack method to generate adversarial image.
        Funcs:
            Fusion of FMR and EDAG Attack.
        Args:
            x (str): clean image path.
            log_info (bool): if print the train information.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        # initialize r
        data = self.get_data_from_img(img=x)
        clean_image = data['inputs']
        data['data_samples'][0].gt_instances = data_sample.gt_instances
        batch_data_samples = data['data_samples']

        # get targets from predict
        target_bboxes, target_scores, target_labels, positive_indices, num_classes = self.edag.get_targets(clean_image, data)
        # get adv labels
        adv_labels = self.edag.get_adv_targets(target_labels, num_classes=num_classes)

        # get feature map of clean img.
        bb_outs = self._forward(img=x, feature_type=self.fmr.feature_type)
        # target featmap
        target_fm = [bb_outs[i] for i in self.fmr.stages]
        # featmap that the attack should be generated
        attack_gt_featmap = [self.fmr.modify_featmap(fm) for fm in target_fm]

        # pertubed image, `X_m` in paper
        pertubed_image = clean_image.clone()
        pertubed_image.requires_grad = True

        step = 0
        total_targets = len(target_labels)
        edag_loss_metric = torch.nn.CrossEntropyLoss(reduction='sum')
        dis_metric = torch.nn.MSELoss()
        sim_metric = torch.nn.BCELoss()
        edag_continue = True # if edag has attack 100% bboxes, then just continue fmr attack.

        if log_info:
            print(f'Start generating adv, total rpn proposal: {total_targets}.')

        try:
            while step < self.M:
                
                # fmr loss ==============
                # get features
                pertub_bb_output = self.model.backbone(pertubed_image)
                if self.fmr.feature_type == 'neck':
                    pertub_bb_output = self.model.neck(pertub_bb_output)
                pertub_featmap = [pertub_bb_output[i] for i in self.fmr.stages]
                
                l1 = 0

                for p_fm, gt_fm in zip(pertub_featmap, attack_gt_featmap):
                    if self.fmr.channel_mean:
                        p_fm = p_fm.mean(dim=1) # compress (B, C, H, W) to (B, H, W)

                    # calculate consine_similarity
                    p_fm_vector = p_fm.view(gt_fm.shape[0], -1) # (B, H*W) if self.channel_mean else (B, C*H*W)
                    gt_fm_vector = gt_fm.view(gt_fm.shape[0], -1)
                    labels = torch.ones(gt_fm.shape[0], 1, device=self.device)
                    cosine_similarity = ((F.cosine_similarity(p_fm_vector, gt_fm_vector) + 1.) / 2.).unsqueeze(-1) # map consie_similarity to [0, 1]
                    sim_loss = sim_metric(cosine_similarity, labels)

                    # calculate distance
                    dis_loss = dis_metric(p_fm, gt_fm)

                    # decide loss format
                    if self.fmr.constrain == 'consine_sim':
                        pertub_loss = sim_loss
                    elif self.fmr.constrain == 'distance':
                        pertub_loss = dis_loss
                    elif self.fmr.constrain == 'combine':
                        pertub_loss = sim_loss + dis_loss

                    l1 += (1 / len(self.fmr.stages)) * pertub_loss

                l2 = dis_metric(pertubed_image, clean_image)
                fmr_loss = l1 + self.fmr.alpha * l2

                # edag loss ==============
                edag_loss = 0
                if edag_continue:
                    results = self.model.predict(pertubed_image, batch_data_samples)
                    
                    logits = results[0].pred_instances.scores
                    positive_logtis = logits[positive_indices] # logits corresponding with targets and advs.

                    # remain the correct targets, drop the incorrect i.e. successful attack targets.
                    # active_target_idx &= (logits.argmax(dim=1) != adv_labels)
                    active_target_mask = (positive_logtis.argmax(dim=1) != adv_labels)
                    active_logits = positive_logtis[active_target_mask]
                    target_labels = target_labels[active_target_mask]
                    adv_labels = adv_labels[active_target_mask]
                    positive_indices = self.edag.update_positive_indices(positive_indices, active_target_mask)
                    # if still has unsuccessfual target.
                    if len(active_logits) > 0:
                        # comput loss
                        correct_loss = edag_loss_metric(active_logits, target_labels)
                        adv_loss = edag_loss_metric(active_logits, adv_labels)
                        # decreasing adv_loss to make pertubed image predicted wrong, and increasing correct_loss to let result far from original correct labels.
                        edag_loss = adv_loss - correct_loss
                    else:
                        # if all targets has been attacked successfully, attack ends.
                        edag_continue = False

                total_loss = self.fmr_weight * fmr_loss + edag_loss

                # backward and comput pertubed image gradient 
                total_loss.backward()
                pertubed_image_grad = pertubed_image.grad.detach()

                with torch.no_grad():
                    # Normalize grad, from paper Eq.(3)
                    r = (self.edag.gamma / pertubed_image_grad.norm(float("inf"))) * pertubed_image_grad 
                    pertubed_image -= r # gradient reverse direction is the direction of decreasing total_loss

                # Zero gradients
                pertubed_image_grad.zero_()
                self.model.zero_grad()

                if step % 10 == 0 and log_info:
                    print("Generation step [{}/{}], fmr_loss: {}, edag_loss: {}, attack percent: {}%.".format(step, self.M, fmr_loss, edag_loss, (total_targets - len(active_logits)) / total_targets * 100))
                    # _exp_name = f'{self.get_attack_name()}/{self.exp_name}'
                    # self.vis.visualize_intermediate_results(r=self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0]),
                    #                                         r_total = self.reverse_augment(x=pertubed_image.squeeze()-clean_image.squeeze(), datasample=data['data_samples'][0]),
                    #                                         pertubed_image=self.reverse_augment(x=pertubed_image.squeeze(), datasample=data['data_samples'][0]),
                    #                                         customize_str=step,
                    #                                         attack_proposals=torch.cat(accum_proposals, dim=0),
                    #                                         image_path=data['data_samples'][0].img_path,
                    #                                         exp_name=_exp_name)
                step += 1
        except Exception as e:
            img_name = os.path.basename(img_name)
            print(f'Attacking error occur, skip image {img_name}. Error info: {e}')
        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = pertubed_image.squeeze() - clean_image.squeeze()
        adv_tensor = pertubed_image.squeeze()

        pertub = self.fmr.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.fmr.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print(f"Generate adv compeleted! Cost iterations {step}.")

        return pertub, adv_image
