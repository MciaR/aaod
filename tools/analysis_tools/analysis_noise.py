import os

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np


COCO_IMAGE_PATH = 'data/coco2017/images/val2017'
VOC_IMAGE_PATH = 'data/VOCdevkit/VOC2007_test/JPEGImages'
COCO_ADV_PATH = 'data/coco2017/'
VOC_ADV_PATH = 'data/VOCdevkit/'

def cal_metrics(attack_name, model_name, dataset_name):
    assert attack_name in ['EDAG', 'FRMR', 'TSA', 'DAG', 'Fusion'], \
        f'`attack_name` must be EDAG, FRMR, TSA, DAG or Fusion.'
    if dataset_name == 'COCO':
        assert model_name in ['FR_R101', 'DINO', 'CenterNet'], \
            f'the model on COCO dataset expected `FR_R101`, `DINO` or `CenterNet`, but got {model_name}'
        adv_samples_path = os.path.join(COCO_ADV_PATH, attack_name, 'adv', model_name)
        clean_samples_path = COCO_IMAGE_PATH
    elif dataset_name == 'VOC':
        assert model_name in ['FR_R101', 'FR_VGG16', 'SSD300'], \
            f'the model on PascalVOC dataset expected `FR_R101`, `FR_VGG16` or `SSD300`, but got {model_name}'
        adv_samples_path = os.path.join(VOC_ADV_PATH, attack_name, 'adv', model_name, 'JPEGImages')
        clean_samples_path = VOC_IMAGE_PATH

    adv_img_list = os.listdir(adv_samples_path)
    clean_img_list = os.listdir(clean_samples_path)

    # to make sure the order of two list is same.
    adv_img_list = sorted(adv_img_list)
    clean_img_list = sorted(clean_img_list)

    assert len(adv_img_list) == len(clean_img_list)
    valid_num = 0
    psnr_list = []
    ssim_list = []

    for adv_name, clean_name in tqdm(zip(adv_img_list, clean_img_list)):
        adv_name_prefix = adv_name.split('.')[0]
        clean_name_prefix = clean_name.split('.')[0]
        assert adv_name_prefix == clean_name_prefix, \
            f'adv image and clean image must be the same. got {adv_name_prefix} and {clean_name_prefix}'
        adv_img_path = os.path.join(adv_samples_path, adv_name)
        clean_img_path = os.path.join(clean_samples_path, clean_name)

        adv_image = imread(adv_img_path)
        clean_image = imread(clean_img_path)

        if adv_image.ndim == 3:
            adv_grey = rgb2gray(adv_image)
        else:
            adv_grey = adv_image
        
        if clean_image.ndim == 3:
            clean_grey = rgb2gray(clean_image)
        else:
            clean_grey = clean_image

        try:
            psnr_value = psnr(adv_image, clean_image)
            ssim_value = ssim(adv_grey, clean_grey)
            valid_num += 1
        except:
            print(adv_name)

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

    print(f'Total pics: {len(adv_img_list)}, valid pics: {valid_num}, drop nums: {len(adv_img_list) - valid_num}.')
    return sum(psnr_list) / valid_num, sum(ssim_list) / valid_num

if __name__  == "__main__":
    attack_name = 'FRMR'
    # model_name = 'SSD300'
    # dataset_name = 'VOC'

    # print(f'{attack_name} attacking {model_name} on {dataset_name}, psnr ans ssim: {cal_metrics(attack_name, model_name, dataset_name)}')

    output = ["dataset model psnr ssim"]

    dataset2model = {'COCO': ['FR_R101', 'DINO', 'CenterNet'], 'VOC': ['FR_R101', 'FR_VGG16', 'SSD300']}
    for dataset, model_list in dataset2model.items():
        for model in model_list:
            print(f'Calculating {model} on {dataset} ...')
            psnr_result, ssim_result = cal_metrics(attack_name, model, dataset)
            output.append(f"{dataset} {model} {psnr_result} {ssim_result}")

    with open('records/analysis/noise_magnitude.txt', 'a') as f:
        f.writelines(output)

    print('PSNR and SSIM has been saved in records/analysis/noise_magnitude.txt')
