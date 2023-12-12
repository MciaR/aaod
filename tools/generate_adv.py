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
            'constrain': 'consine_sim', # distance 似乎也不错，但consine_sim的噪声更小
        },
    },
    'HEFMA': {

    }
}

IMAGE_ROOT = {
    # 'COCO': 'data/coco2017/images/val2017', 
    'COCO': 'data/tiny_coco2017/images/val2017', # tiny for debug now.
    'VOC': None,
}

IMAGE_PATH_PREFIX = {
    'COCO': 'data/coco2017',
    'VOC': None,
}

def generate_and_save(img_list, model, dataset, attacker_name, device):
    assert model in ['FR_R101', 'FR_VGG16', 'SSD300'] and dataset in ['COCO', 'VOC'] and attacker_name in ['FMR', 'THA', 'HEFMA']

    model_config_path = MODEL_CFG_PREFIX[model] + DATASET_SUFFIX[dataset] + '.py'
    checkpoint_file_path = CKPT_FILE_PREFIX[model] + DATASET_SUFFIX[dataset] + '.pth'
    attack_params = ATTACK_PARAMS[attacker_name]['attack_params']

    attack_params.update({'cfg_file': model_config_path, 'ckpt_file': checkpoint_file_path})

    if attacker_name == 'FMR':
        attacker = FMRAttack(**attack_params, device=device)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attack_params, device=device)
    elif attacker_name == 'HEFMA':
        attacker = HEFMAAttack(**attack_params, device=device)

    image_root = IMAGE_ROOT[dataset]
    adv_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset], attacker_name, 'adv', model + '_tiny')
    pertub_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset], attacker_name, 'pertub', model + '_tiny')

    if not os.path.exists(adv_save_dir):
        os.makedirs(adv_save_dir)
    
    if not os.path.exists(pertub_save_dir):
        os.makedirs(pertub_save_dir)

    for img in tqdm(img_list):
        img_path = os.path.join(image_root, img)

        pertub, adv = attacker.generate_adv_samples(x=img_path, log_info=False)

        adv_image = Image.fromarray(adv.astype(np.uint8))
        pertub_image = Image.fromarray(pertub.astype(np.uint8))

        adv_image.save(os.path.join(adv_save_dir, img))
        pertub_image.save(os.path.join(pertub_save_dir, img))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    # params
    model = 'FR_R101'
    dataset = 'COCO'
    attacker_name = 'THA'

    image_list = os.listdir(IMAGE_ROOT[dataset])
    img_l1 = image_list[:250]
    img_l2 = image_list[250:]

    p1 = Process(target=generate_and_save, args=(img_l1, model, dataset, attacker_name, 'cuda:0'))
    p2 = Process(target=generate_and_save, args=(img_l2, model, dataset, attacker_name, 'cuda:1'))
    # start
    p1.start()
    p2.start()

    # end
    p1.join()
    p2.join()


