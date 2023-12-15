import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mmcv

from attack import BaseAttack
from PIL import Image
from torch.optim.lr_scheduler import StepLR


class THAAttack(BaseAttack):
    """Top-k High Activation Attack.
    Args:         
        eplison (float): niose strength.    
        p (int): default `2`, p-norm to calculate distance between clean and adv image.
        adv_type (str): 
            - default `residual`, that means only optimize the noise added to image. 
            - `direct`, that means optimize the whole adversarial sample.
        constrain (str):
            - default `consine_sim`, that means use consine similarity to comput loss.
            - `distance`, that means use distance function to comput loss.
        channel_mean (bool):
            - default `False`, means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
            - `True`, calculate each point mean by channel-wise, the featmap shape is (B, H, W).
    """
    def __init__(self, 
                 cfg_file, 
                 ckpt_file,
                 modify_percent=0.7,
                 scale_factor=0.01,
                 exp_name=None,
                 feature_type = 'neck', # `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
                 channel_mean=False, # means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
                 stages: list = [4], # attack stage of backbone. `(0, 1, 2, 3)` for resnet. 看起来0,3时效果最好。ssd和fr_vgg16就取0
                 p: int = 2, # attack param
                 alpha: float = 5,  # attack param, factor of distance loss. 0.125 for ssd300, 0.25 for fr
                 lr: float = 0.005, # default 0.05
                 M: int = 1000, # attack param, max step of generating perbutaion. 300 for fr, 1000 for ssd.
                 adv_type='direct', # `direct` or `residual`, `direct` means cal pertub noise as whole image directly, `residual` means only cal pertub noise.
                 constrain='consine_sim', #  - default `consine_sim`, that means use consine similarity to comput loss. `distance`, that means use distance function to comput loss.
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, exp_name=exp_name,
                         attack_params=dict(modify_percent=modify_percent, scale_factor=scale_factor, p=p, alpha=alpha, stages=stages, M=M, lr=lr, feature_type=feature_type, adv_type=adv_type, constrain=constrain, channel_mean=channel_mean))

    def get_topk_info(
            self,
            input: torch.Tensor,
            k: int = 10,
            largest: bool = True,
            ):
        flatten_tensor = input.flatten()
        values, topk_indices = torch.topk(input=flatten_tensor, k=k, dim=0, largest=largest)
        assert len(input.shape) == 2, \
            f' featmap tensor must be shape (H, W)'
        H, W = input.shape

        h_indices = topk_indices // W
        w_indices = topk_indices % W

        indices = torch.stack((h_indices, w_indices), dim=1)

        return values, indices

    def get_target_feature(
        self,
        img,
        ):
        """Get target features for visualizer."""
        ori_features = self._forward(img=img, feature_type=self.feature_type)
        target_features = []
        for i in range(len(ori_features)):
            if i in self.stages:
                _gt_feat = self.modify_featmap(ori_features[i]) # _gt_feat : (B, H, W) if `channel_mean` is True else (B, C, H, W)
                if self.channel_mean:
                    _gt_feat = _gt_feat.unsqueeze(1) # _gt_feat : (B, 1, H, W)
            else:
                _gt_feat = ori_features[i]
            target_features.append(_gt_feat)

        return target_features

    def modify_featmap(
            self,
            featmap: torch.Tensor
            ):
        """Modify topk value in each featmap (H, W).
        Args:
            featmap (torch.Tensor): shape `(N, C, H, W)`
            mean_featmap
            scale_factor (float): miniumize factor
        """
        N, C, H, W = featmap.shape
        modified_feat = None if self.channel_mean else featmap.clone()

        for sample_ind in range(N):
            sample_featmap = featmap[sample_ind]
            k = int(H * W * self.modify_percent)
            mean_featmap = torch.mean(sample_featmap, dim=0)
            _, topk_indices = self.get_topk_info(input=mean_featmap, k=k, largest=True)

            # scale indices value in each featmap
            if self.channel_mean:
                mean_featmap[topk_indices[:, 0], topk_indices[:, 1]] = mean_featmap[topk_indices[:, 0], topk_indices[:, 1]] * self.scale_factor
                mean_featmap = mean_featmap.unsqueeze(0)
                if modified_feat is None:
                    modified_feat = mean_featmap
                else:
                    torch.stack((modified_feat, mean_featmap), dim=0)
            else:
                modified_feat[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] = modified_feat[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] * self.scale_factor

        return modified_feat

    def reverse_augment(self, x, datasample):
        """Reverse tensor to input image."""
        ori_shape = datasample.ori_shape
        pad_shape = datasample.pad_shape

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375] # for fr
        # std = [1, 1, 1] # for ssd
        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)

        # revert normorlize
        ori_pic = x * std_t + mean_t
        # revert bgr_to_rgb
        # dont need to revert bgr_to_rgb, beacuse saving format is RGB if using PIL.Image
        # ori_pic = ori_pic[[2, 1, 0], ...]
        # revert pad
        ori_pic = ori_pic[:, :datasample.img_shape[0], :datasample.img_shape[1]]

        # (c, h, w) to (h, w, c)
        ori_pic = ori_pic.permute(1, 2, 0)
        # cut overflow values
        ori_pic = torch.clamp(ori_pic, 0, 255)

        ori_pic = ori_pic.detach().cpu().numpy()

        # for fr
        ori_pic, _ = mmcv.imrescale(
                            ori_pic,
                            ori_shape,
                            interpolation='bilinear',
                            return_scale=True,
                            backend='cv2')
        # ori_pic, _, _ = mmcv.imresize(
        #                     ori_pic,
        #                     (ori_shape[1], ori_shape[0]),
        #                     interpolation='bilinear',
        #                     return_scale=True,
        #                     backend='cv2')
        
        return ori_pic

    def test_adv_result(self, clean_img, adv_img, metric='std'):
        """Output advesrairal result compared to clean."""

        clean_featmap = self.model.backbone(clean_img)
        adv_featmap = self.model.backbone(adv_img)
        if self.feature_type == 'neck':
            clean_featmap = self.model.neck(clean_featmap)
            adv_featmap = self.model.neck(adv_featmap)

        # cal std
        if metric == 'std':
            stage_std_clean = []
            stage_std_adv = []
            for i in range(len(clean_featmap)):
                stage_clean_featmap = clean_featmap[i] # (B, C, H, W)
                stage_adv_featmap = adv_featmap[i]

                B, C, H, W = stage_adv_featmap.shape

                batch_std_clean = 0
                batch_std_adv = 0

                for sample_ind in range(B):
                    sample_clean_featmap = stage_clean_featmap[sample_ind] # (C, H, W)
                    sample_adv_featmap = stage_adv_featmap[sample_ind]

                    std_clean = torch.std(sample_clean_featmap.view(C, -1), dim=-1) # (C, )
                    std_adv = torch.std(sample_adv_featmap.view(C, -1), dim=-1)

                    batch_std_clean += std_clean.mean() # (1, )
                    batch_std_adv += std_adv.mean() # (1, )

                batch_std_adv /= B
                batch_std_clean /= B

                stage_std_clean.append(batch_std_clean.cpu().detach().numpy())
                stage_std_adv.append(batch_std_adv.cpu().detach().numpy())

            print(self.stages, stage_std_clean, stage_std_adv, sep='\n')

    def generate_adv_samples(self, x, log_info=True):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image path.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        # get feature map of clean img.
        bb_outs = self._forward(img=x, feature_type=self.feature_type)
        # target featmap
        target_fm = [bb_outs[i] for i in self.stages]
        # featmap that the attack should be generated
        attack_gt_featmap = [self.modify_featmap(fm) for fm in target_fm]

        # initialize r
        data = self.get_data_from_img(img=x)
        clean_image = data['inputs']

        if self.adv_type == 'residual':
            r = clean_image.clone() + torch.randn(clean_image.shape, requires_grad=True, device=self.device) # for ssd
            r.retain_grad()
        else:
            r = torch.randn(clean_image.shape, requires_grad=True, device=self.device) # for fr
        
        # params
        step = 0
        optimizer = torch.optim.Adam(params=[r], lr=self.lr)
        scheduler = StepLR(optimizer,
                               gamma = 0.3, # The number we multiply learning rate until the milestone. 
                               step_size = self.M * 0.3)
        
        sim_metric = torch.nn.BCELoss() # combined with consine_similarity, suitable for direction and value.
        dis_metric = torch.nn.MSELoss() # cal loss directly, suitable for value.

        while step < self.M:
            # calculate output featmap
            pertub_bb_output = self.model.backbone(r)
            if self.feature_type == 'neck':
                pertub_bb_output = self.model.neck(pertub_bb_output)
            pertub_featmap = [pertub_bb_output[i] for i in self.stages]

            l1 = 0
            
            for p_fm, gt_fm in zip(pertub_featmap, attack_gt_featmap):
                if self.channel_mean:
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
                if self.constrain == 'consine_sim':
                    pertub_loss = sim_loss
                elif self.constrain == 'distance':
                    pertub_loss = dis_loss
                elif self.constrain == 'combine':
                    pertub_loss = sim_loss + dis_loss

                l1 += (1 / len(self.stages)) * pertub_loss

            l2 = dis_metric(r, clean_image)
            loss = l1 + self.alpha * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            if step % 10 == 0 and log_info:
                print("Train step [{}/{}], lr: {:3f}, loss: {}, pertub_loss: {}, distance_loss: {}.".format(step, self.M, optimizer.param_groups[0]["lr"] , loss, l1, l2))

        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = r.squeeze() - clean_image.squeeze()
        adv_tensor = r.squeeze()

        pertub = self.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print("Generate adv compeleted!")

        return pertub, adv_image