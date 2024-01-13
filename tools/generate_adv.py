import os
import torch
import shutil
import time
import numpy as np

from attack import *
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process
from setting import get_attacker_params


IMAGE_PATH_PREFIX = {
    'COCO': 'data/coco2017',
    'VOC': 'data/VOCdevkit',
}

def format_eta(seconds):
    # 将秒转换为天，时，分
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(days)}D: {int(hours):02}H: {int(minutes):02}M"

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

    adv_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'adv', model)
    pertub_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'pertub', model)

    if dataset_name == "VOC":
        # copy png annotations to adversarial samples dir
        # annotations_dir = os.path.join(adv_save_dir, 'Annotations')
        # if not os.path.exists(annotations_dir):
        #     os.makedirs(annotations_dir)
        # source_anno_root = 'data/VOCdevkit/VOC2007_test/Annotations'
        # for file_name in os.listdir(source_anno_root):
        #     anno_source_path = os.path.join(source_anno_root, file_name)
        #     anno_target_path = os.path.join(annotations_dir, file_name)
        #     shutil.copyfile(anno_source_path, anno_target_path)

        adv_save_dir = os.path.join(adv_save_dir, 'PNGImages')
        pertub_save_dir = os.path.join(pertub_save_dir, 'PNGImages')

    if not os.path.exists(adv_save_dir):
        os.makedirs(adv_save_dir)
    
    if not os.path.exists(pertub_save_dir):
        os.makedirs(pertub_save_dir)

    dataset = attacker.dataset
    start_idx = int(start * len(dataset))
    end_idx = int(end * len(dataset))

    iterations = tqdm(range(start_idx, end_idx))
    start_time = time.time()
    for i in iterations:
        cur_time = time.time()
        passed_time = cur_time - start_time
        average_sample_time = passed_time / (i - start_idx) if i > start_idx else 1
        eta = format_eta((end_idx - i) * average_sample_time)
        iterations.set_description(f'{device} generating {i}th image, ETA: {eta}.')

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
    num_gpus = torch.cuda.device_count()
    print(f'Total GPU nums: {num_gpus}, image will be divided to {num_gpus} parts to run.')
    processes = []
    
    # params
    model = 'FR_R101'
    dataset_name = 'VOC'
    attacker_name = 'FRMR'

    for i in range(num_gpus):
        start = i / num_gpus
        end = (i + 1) / num_gpus
        device = f'cuda:{i}'
        p = Process(target=generate_and_save, args=(start, end, model, dataset_name, attacker_name, device))
        p.start()
        processes.append(p)

    # asnyc
    for p in processes:
        p.join()

    if dataset_name == 'VOC':
        print('Image in VOC Datasets need transfer png to jpg...')
        adv_png_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'adv', model, 'PNGImages')
        adv_jpg_save_dir = os.path.join(IMAGE_PATH_PREFIX[dataset_name], attacker_name, 'adv', model, 'JPEGImages')
        if not os.path.exists(adv_jpg_save_dir):
            os.makedirs(adv_jpg_save_dir)
        for img in os.listdir(adv_png_save_dir):
            img_name = img.split('.')[0]
            jpg_img = img_name + '.jpg'
            source_file = os.path.join(adv_png_save_dir, img)
            target_file = os.path.join(adv_jpg_save_dir, jpg_img)
            shutil.copyfile(source_file, target_file)
    print('Generate Adversarial Samples has done.')

