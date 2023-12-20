from visualizer import ExpVisualizer
from attack import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
            'constrain': 'consine_sim', # distance 似乎也不错，但consine_sim的噪声更小
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
            'fmr_weight': 0.2, # it seems like 0.2 better that 0.5.
            'fmr_params': None,
            'edag_params': None,
        },
        'remain_list': ['M', 'fmr_weight']
    }
}

def execute_attack(attacker_name, model_name, dataset_name, exp_name, start, end):
    """Execute specific attack.
    Args:
        attacker_name (str): attacker name.
        model_name (str): model config name.
        dataset_name (str): dataset name. Options: `COCO` or `VOC`. 
        exp_name (str): experience name, decide folder which result will be saved.
        start (int): start index of dataset image.
        end (int): end index of dataset image.
    """

    assert attacker_name in ['FMR', 'THA', 'EXPDAG', 'DAG', 'EFMR', 'Fusion'], \
        f'`attacker_name` must be `FMR`, `THA`, `DAG`, `EFMR`, `Fusion` or `EXPDAG`.'
    
    attack_cfg = CFG[attacker_name]
    attacker_params = attack_cfg['attack_params']
    model_config_path = MODEL_CFG_PREFIX[model_name] + DATASET_SUFFIX[dataset_name] + '.py'
    checkpoint_file_path = CKPT_FILE_PREFIX[model_name] + DATASET_SUFFIX[dataset_name] + '.pth'
    attacker_params.update({'cfg_file': model_config_path, 'ckpt_file': checkpoint_file_path})
    attacker_params.update({'exp_name': model_name + '/' + exp_name}) # update exp_name to attacker_papams
    remain_list = attack_cfg['remain_list']

    # expVisualizer params
    show_features = True
    show_lvl_preds = True
    save_analysis = True

    if attacker_name == 'FMR':
        attacker = FMRAttack(**attacker_params)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif attacker_name == 'EXPDAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = EXPDAGAttack(**attacker_params)
    elif attacker_name == 'DAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = DAGAttack(**attacker_params)
    elif attacker_name == 'EFMR':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = EFMRAttack(**attacker_params)
    elif attacker_name == 'Fusion':
        save_analysis = False
        attacker_params.update({'fmr_params': CFG['FMR']['attack_params'], 'edag_params': CFG['EFMR']['attack_params']})
        attacker = FusionAttack(**attacker_params)

    vis = ExpVisualizer(cfg_file=attacker_params['cfg_file'], ckpt_file=attacker_params['ckpt_file'], use_attack=True, attacker=attacker)
    dataset = vis.dataset

    for i in range(start, end):
        vis.show_attack_results(model_name=model_name + '_' + dataset_name, data_sample=dataset[i]['data_samples'], dataset_idx=i, save=True, feature_grey=False, remain_list=remain_list, 
                                show_features=show_features, show_lvl_preds=show_lvl_preds, save_analysis=save_analysis, show_thr=0.1)

if __name__ == '__main__':
    execute_attack(attacker_name='Fusion', model_name='FR_R101', dataset_name='COCO', exp_name='test1220', start=0, end=1)
