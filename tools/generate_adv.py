import os
from attack import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
import torch


def generate_and_save(img_list, device):
    config_file = 'configs/faster_rcnn_r101_fpn_coco.py'
    checkpoint_file = 'pretrained/fr_r101_coco_0394.pth'
    attack_params = {
        'cfg_file': config_file,
        'ckpt_file': checkpoint_file,
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
    }

    attacker = FMRAttack(**attack_params, device=device)

    image_root ='data/coco2017/images/val2017'
    adv_save_dir = 'data/coco2017/FMR/adv/fr_r101_tiny'
    pertub_save_dir = 'data/coco2017/FMR/pertub/fr_r101_tiny'
    # image_root = 'data/VOCdevkit/tiny_voc/JPEGImages'
    # adv_save_dir = 'data/VOCdevkit/adv/fr_vgg_tiny/JPEGImages'
    # pertub_save_dir = 'data/VOCdevkit/pertub/fr_vgg_tiny/JPEGImages'

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
    
    # image_list = os.listdir('data/VOCdevkit/tiny_voc/JPEGImages')
    image_list = os.listdir('data/tiny_coco2017/images/val2017')
    img_l1 = image_list[:250]
    img_l2 = image_list[250:]

    p1 = Process(target=generate_and_save, args=(img_l1, 'cuda:0'))
    p2 = Process(target=generate_and_save, args=(img_l2, 'cuda:1'))

    # start
    p1.start()
    p2.start()

    # end
    p1.join()
    p2.join()


