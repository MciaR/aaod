MODEL_CFG_PREFIX = {
    'FR_R101': 'configs/faster_rcnn_r101_fpn',
    'FR_VGG16': 'configs/fr_vgg16',
    'SSD300': 'configs/ssd300'
}

DATASET_SUFFIX = {
    'COCO': '_coco',
    'VOC': '_voc' 
}

CKPT_FILE_PREFIX = {
    'FR_R101': 'pretrained/fr_r101',
    'FR_VGG16': 'pretrained/fr_vgg16',
    'SSD300': 'pretrained/ssd300'
}

CFG = {
    'FMR': {
        'attack_params': {
            # NOTE: best for now, 2023.12.11
            'global_scale': 1,
            'use_channel_scale': False,
            'feature_type': 'neck',
            'channel_mean': False,
            'stages': [0],
            'p': 2,
            'alpha': 5,
            'lr': 0.05,
            'M': 500,
            # 'adv_type': 'direct', 
            'adv_type': 'residual',
            # 'constrain': 'distance'
            # 'constrain': 'consine_sim'
            'constrain': 'distance'
        },
        # field which will be saved in result name.
        'remain_list': ['feature_type', 'channel_mean', 'stages', 'alpha', 'lr', 'M', 'adv_type', 'constrain', 'global_scale', 'use_channel_scale']
    },
    'THA': {
        'attack_params': {
            # NOTE: best for now, 2023.12.11
            'modify_percent': 0.7,
            'scale_factor': 0.01,
            'feature_type' :  'neck',
            'channel_mean': False,
            'stages': [0], # attack stage of backbone. `(0, 1, 2, 3)` for resnet. 看起来0,3时效果最好。ssd和fr_vgg16就取0
            'p': 2,
            'alpha': 1,
            'lr': 0.05,
            'M': 300, 
            'adv_type': 'residual',
            'constrain': 'distance', # distance 似乎也不错，但consine_sim的噪声更小
        },
        'remain_list': ['feature_type', 'channel_mean', 'stages', 'alpha', 'lr', 'M', 'adv_type', 'constrain', 'modify_percent', 'scale_factor']
    },
    'EXPDAG': {
        'attack_params': {
            'gamma': 0.5,
            'M': 500,
            'cfg_options': dict(
                model = dict(
                    test_cfg = dict(
                        rpn=dict( # makes attack dense region.
                        nms_pre=5000,
                        max_per_img=5000,
                        nms=dict(type='nms', iou_threshold=0.9),
                        min_bbox_size=0),
                        rcnn=None, # makes pred result no nms.
                    ),
                )
            )
        },
        'remain_list': ['gamma', 'M']
    },
    'DAG': {
        'attack_params': {
            'gamma': 0.5,
            'M': 500,
            'cfg_options': dict(
                model = dict(
                    test_cfg = dict(
                        rpn=dict( # makes attack dense region.
                        nms_pre=5000,
                        max_per_img=5000,
                        nms=dict(type='nms', iou_threshold=0.9),
                        min_bbox_size=0),
                        rcnn=None, # makes pred result no nms.
                    ),
                )
            )
        },
        'remain_list': ['gamma', 'M']
    },
    'EFMR': {
        'attack_params': { # NOTE: Best for fr now: 2023.12.19, 现在攻击的结果有一点点没攻击干净（只有一两个物体，但score下降了）。
            'gamma': 0.7, # controls noise strength
            'M': 150, # controls iterations (time consuming)
            'cfg_options': dict(
                model = dict(
                    test_cfg = dict(
                        rpn=dict( # makes attack dense region.
                        nms_pre=1000, # nms pre should > max_per_img, otherwise after nms, there will be less than max_per_img. i.e. there are less that max_per_img for rcnn.
                        max_per_img=500,
                        nms=dict(type='nms', iou_threshold=0.99),
                        min_bbox_size=0),
                        rcnn=None, # makes pred result no nms.
                    ),
                )
            ),
            # 'attack_target': {0: 4}, # 0 represents `people`, 4 represents `airplane`.
        },
        'remain_list': ['gamma', 'M']
    },
    'Fusion': {
        'attack_params': { # NOTE: best for now 2023.12.20.
            'M': 300,
            'frmr_weight': 0.2, # it seems like 0.2 better that 0.5.
            'frmr_params': None,
            'edag_params': None,
        },
        'remain_list': ['M', 'frmr_weight']
    }
}
