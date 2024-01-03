import os
import torch
import numpy as np

from attack import *
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
from setting import get_attacker_params


IMAGE_PATH_PREFIX = {
    'COCO': 'data/coco2017',
    'VOC': None,
}

def generate_and_save(start, end, model, dataset_name, attacker_name, device):
    attacker_params = get_attacker_params(model_name=model, dataset_name=dataset_name, attacker_name=attacker_name)

    if attacker_name == 'FRMR':
        attacker = FRMRAttack(**attacker_params, device=device)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attacker_params, device=device)
    elif attacker_name == 'EXPDAG':
        attacker = EXPDAGAttack(**attacker_params, device=device)
    elif attacker_name == 'DAG':
        attacker = DAGAttack(**attacker_params, device=device)
    elif attacker_name == 'EDAG':
        attacker = EDAGAttack(**attacker_params, device=device)
    elif attacker_name == 'Fusion':
        attacker = FusionAttack(**attacker_params, device=device)
    elif attacker_name == 'TSA':
        attacker = EDAGAttack(**attacker_params, device=device)

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


