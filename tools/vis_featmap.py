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
                'ckpt_file': 'pretrained/fr_r101_coco_0394.pth',
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

        },
        'HEFMA': {

        }
    }
    assert name in ['FMR', 'THA', 'HEFMA'], \
        f'`name` must be `FMR`, `THA` or `HEFMA`.'
    
    attack_cfg = cfg[name]
    attacker_params = attack_cfg['attack_params']
    remain_list = attack_cfg['remain_list']
    if name == 'FMR':
        attacker = FMRAttack(**attacker_params)
    elif name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif name == 'HEFMA':
        attacker = HEFMAAttack(**attacker_params)

    vis = ExpVisualizer(cfg_file=attacker_params['cfg_file'], ckpt_file=attacker_params['ckpt_file'], use_attack=True, attacker=attacker)
    dataset = vis.dataset

    for i in range(start, end):
        vis.show_attack_results(model_name="FR_R101_COCO", data_sample=dataset[i]['data_samples'], dataset_idx=i, save=True, feature_grey=False, remain_list=remain_list, exp_name=exp_name)


if __name__ == '__main__':
    execute_attack(name='FMR', exp_name='negative_one_point_wise', start=0, end=1)
