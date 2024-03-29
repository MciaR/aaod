import torch
import torch.nn.functional as F

from attack import BaseAttack
from torch.optim.lr_scheduler import StepLR


class FRMRAttack(BaseAttack):
    """Feature Representation Mean Regression Attack.
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
                 global_scale=1.1,
                 use_channel_scale=True,
                 exp_name=None,
                 feature_type = 'backbone', # `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
                 channel_mean=False, # means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
                 stages: list = [4], # attack stage of backbone. `(0, 1, 2, 3)` for resnet. 看起来0,3时效果最好。ssd和fr_vgg16就取0
                 alpha: float = 5,  # attack param, factor of distance loss. 0.125 for ssd300, 0.25 for fr
                 lr: float = 0.005, # default 0.05
                 M: int = 1000, # attack param, max step of generating perbutaion. 300 for fr, 1000 for ssd.
                 early_stage: bool = False,
                 cfg_options=None,
                 adv_type='direct', # `direct` or `residual`, `direct` means cal pertub noise as whole image directly, `residual` means only cal pertub noise.
                 constrain='consine_sim', #  - default `consine_sim`, that means use consine similarity to comput loss. `distance`, that means use distance function to comput loss.
                 device='cuda:0') -> None:
        # if not attack backbone early stage
        if not early_stage:
            cfg_options=None
        else:
            assert cfg_options is not None, \
                f'if `early_stage` is True, cfg_options must be set.'
        super().__init__(cfg_file, ckpt_file, device=device, exp_name=exp_name, cfg_options=cfg_options,
                         attack_params=dict(early_stage=early_stage, global_scale=global_scale, use_channel_scale=use_channel_scale, alpha=alpha, stages=stages, M=M, lr=lr, feature_type=feature_type, adv_type=adv_type, constrain=constrain, channel_mean=channel_mean))

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

    @staticmethod
    def scale_map_function(x: torch.Tensor):
        """Get scale factor by each element in x.
        Args:
            x (torch.Tensor): must be (C, )
        Returns:
            scales (torch.Tensor): scale factor respect to each element of x, it's in range [0.0, 2.0].
        """
        # mean_val = torch.mean(x)
        # return (1 - torch.sin(torch.pi * (x - 0.5))) * mean_val / x
        return -1. * torch.ones_like(x)
    
    # def modify_featmap(
    #     self,
    #     featmap: torch.Tensor,
    # ):
    #     """Modify activation of featmap to mean.
    #     Args:
    #         featmap (torch.Tensor): feature map you want to modifiy.
    #         global_scale (float): scale factor for each channel.
    #         channel_scale (bool): if `True`, channels `x` will be multiplied by `scale_map_function(x)`
    #     """
    
    #     N, C, H, W = featmap.shape
    #     modify_feat = torch.ones(N, C, H, W, device=self.device)

    #     for sample_ind in range(N):
    #         sample_featmap = featmap[sample_ind]
    #         # (C, H*W)
    #         sample_featmap = sample_featmap.reshape(C, -1)
    #         # (C,)
    #         channel_mean = torch.mean(sample_featmap, dim=-1)
    #         channel_scale = self.scale_map_function(channel_mean) if self.use_channel_scale else torch.ones_like(channel_mean)
    #         for c in range(C):
    #             modify_feat[sample_ind][c, :, :] = modify_feat[sample_ind][c, :, :] * channel_mean[c] * self.global_scale * channel_scale[c]

    #     return modify_feat

    def modify_featmap(
            self,
            featmap: torch.Tensor,
    ):
        """Modify activation of feamap point-wise.
        Args:
            featmap (torch.Tensor): feature map you want to modify.
        """
        reverse_feat = featmap.clone() * -1.

        return reverse_feat

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

    def generate_adv_samples(self, x, data_sample=None, log_info=True):
        """Attack method to generate adversarial image.
        Args:
            x (str): clean image path.
            log_info (bool): if print the train information.
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
        optimizer = torch.optim.Adam(params=[r], lr=self.lr) # only update r's gradient.
        scheduler = StepLR(optimizer,
                               gamma = 0.3, # The number we multiply learning rate until the milestone. 
                               step_size = self.M * 0.3)
        
        sim_metric = torch.nn.BCELoss() # combined with consine_similarity, suitable for direction and value.
        dis_metric = torch.nn.MSELoss() # cal loss directly, suitable for value.

        while step < self.M:
            # calculate output featmap
            pertub_bb_output = self.model.backbone(r)
            if self.feature_type == 'neck' and self.model.with_neck:
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
                cosine_similarity = torch.clamp(cosine_similarity, 0, 1) # make value to [0, 1] surely
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
            self.test_adv_result(clean_img=clean_image, adv_img=adv_tensor.unsqueeze(0))

        return pertub, adv_image
