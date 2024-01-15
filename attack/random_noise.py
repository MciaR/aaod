import torch
import torch.nn.functional as F

from attack import BaseAttack
from torch.optim.lr_scheduler import StepLR


class RandomNoise(BaseAttack):
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
                 exp_name=None,
                 feature_type = 'backbone', # `'backbone'` - `model.backbone`, `'neck'` - `model.neck`.
                 channel_mean=False, # means use `C` (channel) to comput loss, the featmap shape is (B, C, H, W).
                 lr: float = 0.005, # default 0.05
                 M: int = 1000, # attack param, max step of generating perbutaion. 300 for fr, 1000 for ssd.
                 adv_type='direct', # `direct` or `residual`, `direct` means cal pertub noise as whole image directly, `residual` means only cal pertub noise.
                 constrain='consine_sim', #  - default `consine_sim`, that means use consine similarity to comput loss. `distance`, that means use distance function to comput loss.
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, exp_name=exp_name,
                         attack_params=dict(M=M, lr=lr, feature_type=feature_type, adv_type=adv_type, constrain=constrain, channel_mean=channel_mean))

    def get_target_feature(
        self,
        img,
        ): 
        """Get target features for visualizer."""
        ori_features = self._forward(img=img, feature_type=self.feature_type)

        return ori_features

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

            loss = dis_metric(r, clean_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            if step % 10 == 0 and log_info:
                print("Train step [{}/{}], lr: {:3f}, loss: {}".format(step, self.M, optimizer.param_groups[0]["lr"] , loss))

        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = r.squeeze() - clean_image.squeeze()
        adv_tensor = r.squeeze()

        pertub = self.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print("Generate adv compeleted!")

        return pertub, adv_image
