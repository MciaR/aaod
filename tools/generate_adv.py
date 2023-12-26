import os
from attack import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
import torch

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

ATTACK_PARAMS = {
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
            'constrain': 'distance',
        },
    },
    'EXPDAG': {
        'attack_params': {
            # NOTE: best for now, 2023.12.18
            # 'cfg_file': "configs/fr_vgg16_coco.py", 
            # 'ckpt_file': "pretrained/fr_vgg16_coco.pth",
            'cfg_file': "configs/faster_rcnn_r101_fpn_coco.py", 
            'ckpt_file': "pretrained/fr_r101_coco.pth",
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
        }
    },
    'DAG': {
        'attack_params': {
            'cfg_file': "configs/faster_rcnn_r101_fpn_coco.py", 
            'ckpt_file': "pretrained/fr_r101_coco.pth",
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
        }
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
            )
        }
    },
    'Fusion': {
        'attack_params': {
            'M': 300,
            'fmr_weight': 0.2,
            'fmr_params': None,
            'edag_params': None,
        },
    },
    'Fusion1': {
        'attack_params': {
            'M': 300,
            'fmr_weight': 1,
            'fmr_params': None,
            'edag_params': None,
        },
    },
    'TSA': { # Two stage attack, used FMR attacked result to attack, so the config same to EFMR(EDAG).
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
            )
        }
    },

}

IMAGE_PATH_PREFIX = {
    'COCO': 'data/coco2017',
    'VOC': None,
}

def generate_and_save(start, end, model, dataset_name, attacker_name, device):
    assert model in ['FR_R101', 'FR_VGG16', 'SSD300'] and dataset_name in ['COCO', 'VOC'] and attacker_name in ['FMR', 'THA', 'EXPDAG', 'DAG', 'EFMR', 'Fusion', 'Fusion1', 'TSA']

    model_config_path = MODEL_CFG_PREFIX[model] + DATASET_SUFFIX[dataset_name] + '.py'
    checkpoint_file_path = CKPT_FILE_PREFIX[model] + DATASET_SUFFIX[dataset_name] + '.pth'
    attack_params = ATTACK_PARAMS[attacker_name]['attack_params']

    attack_params.update({'cfg_file': model_config_path, 'ckpt_file': checkpoint_file_path})

    if attacker_name == 'FMR':
        attacker = FMRAttack(**attack_params, device=device)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attack_params, device=device)
    elif attacker_name == 'EXPDAG':
        attacker = EXPDAGAttack(**attack_params, device=device)
    elif attacker_name == 'DAG':
        attacker = DAGAttack(**attack_params, device=device)
    elif attacker_name == 'EFMR':
        attacker = EFMRAttack(**attack_params, device=device)
    elif attacker_name == 'Fusion' or attacker_name == 'Fusion1':
        attack_params.update({'fmr_params': ATTACK_PARAMS['FMR']['attack_params'], 'edag_params': ATTACK_PARAMS['EFMR']['attack_params']})
        attacker = FusionAttack(**attack_params, device=device)
    elif attacker_name == 'TSA':
        attacker = EFMRAttack(**attack_params, device=device)

    adv_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'adv', model + '_tiny')
    pertub_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'pertub', model + '_tiny')

    if not os.path.exists(adv_save_dir):
        os.makedirs(adv_save_dir)
    
    if not os.path.exists(pertub_save_dir):
        os.makedirs(pertub_save_dir)

    dataset = attacker.dataset
    start_idx = int(start * len(dataset))
    end_idx = int(end * len(dataset))

    iterations = tqdm(range(start_idx, end_idx))
    for i in iterations:
        iterations.set_description(f'{device} generating {i}th image')

        data = dataset[i]
        data_sample = data['data_samples']
        img_path = data_sample.img_path
        img_name = os.path.basename(img_path).split('.')[0]

        pertub, adv = attacker.generate_adv_samples(x=img_path, data_sample=data_sample, log_info=False)

        adv_image = Image.fromarray(adv.astype(np.uint8))
        pertub_image = Image.fromarray(pertub.astype(np.uint8))

        # jpg will compress noise to make attack weaker.
        adv_image.save(os.path.join(adv_save_dir, img_name + '.png'))
        pertub_image.save(os.path.join(pertub_save_dir, img_name + '.png'))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    # params
    model = 'FR_R101'
    dataset_name = 'COCO'
    attacker_name = 'TSA'

    # must know dataset length
    p1 = Process(target=generate_and_save, args=(0, 0.5, model, dataset_name, attacker_name, 'cuda:0'))
    p2 = Process(target=generate_and_save, args=(0.5, 1, model, dataset_name, attacker_name, 'cuda:1'))
    # start
    p1.start()
    p2.start()

    # end
    p1.join()
    p2.join()


