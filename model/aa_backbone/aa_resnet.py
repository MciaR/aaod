import torch

from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet


@MODELS.register_module()
class AAResNet(ResNet):
    """ResNet backbone for adversarial attack ."""
    def __init__(self,
                depth,
                in_channels=3,
                stem_channels=None,
                base_channels=64,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                style='pytorch',
                deep_stem=False,
                avg_down=False,
                frozen_stages=-1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                dcn=None,
                stage_with_dcn=(False, False, False, False),
                plugins=None,
                with_cp=False,
                zero_init_residual=True,
                pretrained=None,
                init_cfg=None):
        super(AAResNet, self).__init__(depth, in_channels, stem_channels, base_channels, num_stages, strides, dilations, out_indices, style, deep_stem, avg_down, frozen_stages, conv_cfg, norm_cfg, norm_eval, dcn, stage_with_dcn, plugins, with_cp, zero_init_residual, pretrained, init_cfg)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        outs = self.attack_method(outs)
        return tuple(outs)
    
    def attack_method(self, bb_outputs, topk = 1):
        """ Find mean featmap max and min activate value pixel, and switch them."""
        attack_result = []
        out_len = len(bb_outputs)
        for i in range(out_len):
            featmap = bb_outputs[i]
            # feat_maps: (N, C, H, W)
            attack_result.append(self.modify_featmap(featmap=featmap))
        
        return attack_result

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
    
    def modify_featmap(
            self,
            featmap: torch.Tensor,
            modify_percent: float = 0.3,
            scale_factor: float = 0.3):
        """Modify topk value in each featmap (H, W).
        Args:
            featmap (torch.Tensor): shape `(N, C, H, W)`
            mean_featmap
            scale_factor (float): miniumize factor
        """
        N, C, H, W = featmap.shape
        for sample_ind in range(N):
            sample_featmap = featmap[sample_ind]
            k = int(H * W * modify_percent)
            mean_featmap = torch.mean(sample_featmap, dim=0)
            _, topk_indices = self.get_topk_info(input=mean_featmap, k=k, largest=True)

            # scale indices value in each featmap
            featmap[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] = featmap[sample_ind, :, topk_indices[:, 0], topk_indices[:, 1]] * scale_factor

        return featmap


        