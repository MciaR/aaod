from visualizer import ExpVisualizer
from attack import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def execute_attack(name, exp_name, start, end):
    """Execute specific attack.
    Args:
        name (str): attacker name.
        exp_name (str): experience name, decide folder which result will be saved.
        start (int): start index of dataset image.
        end (int): end index of dataset image.
    """
    cfg = {
        'FMR': {
            'attack_params': {
                # NOTE: best for now, 2023.12.11
                'cfg_file': 'configs/faster_rcnn_r101_fpn_coco.py',
                'ckpt_file': 'pretrained/fr_r101_coco.pth',
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
                'cfg_file': "configs/faster_rcnn_r101_fpn_coco.py", 
                'ckpt_file': "pretrained/fr_r101_coco.pth",
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
        'HEFMA': {

        },
        'DAG': {
            'attack_params': {
                'cfg_file': "configs/faster_rcnn_r101_fpn_coco.py", 
                'ckpt_file': "pretrained/fr_r101_coco.pth",
                'gamma': 0.5,
                'M': 300,
            }
        }
    }
    assert name in ['FMR', 'THA', 'HEFMA', 'DAG'], \
        f'`name` must be `FMR`, `THA`, `DAG` or `HEFMA`.'
    
    attack_cfg = cfg[name]
    attacker_params = attack_cfg['attack_params']
    remain_list = attack_cfg['remain_list']
    # expVisualizer params
    show_features = True
    show_lvl_preds = True
    save_analysis = True

    if name == 'FMR':
        attacker = FMRAttack(**attacker_params)
    elif name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif name == 'HEFMA':
        attacker = HEFMAAttack(**attacker_params)
    elif name == 'DAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = DAGAttack(**attacker_params)

    vis = ExpVisualizer(cfg_file=attacker_params['cfg_file'], ckpt_file=attacker_params['ckpt_file'], use_attack=True, attacker=attacker)
    dataset = vis.dataset

    for i in range(start, end):
        vis.show_attack_results(model_name="FR_R101_COCO", data_sample=dataset[i]['data_samples'], dataset_idx=i, save=True, feature_grey=False, remain_list=remain_list, exp_name=exp_name, 
                                show_features=show_features, show_lvl_preds=show_lvl_preds, save_analysis=save_analysis)


if __name__ == '__main__':
    # execute_attack(name='FMR', exp_name='negative_one_point_wise_no_optimizer_1213', start=0, end=1)
    # execute_attack(name='THA', exp_name='test', start=1, end=2)
    execute_attack(name='DAG', exp_name='test', start=0, end=1)
