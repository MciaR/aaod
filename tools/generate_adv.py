import os
from attack import HAFAttack
import cv2
from tqdm import tqdm
from multiprocessing import Process
import torch


def generate_and_save(img_list, device):

    haf_attack = HAFAttack(cfg_file='configs/fr_vgg16_coco.py', ckpt_file='test_model/fr_vgg16_coco_0184.pth', device=device)

    image_root = 'data/coco2017/images/val2017'
    adv_save_dir = 'data/coco2017/adv/fr_vgg16_tiny'
    pertub_save_dir = 'data/coco2017/pertub/fr_vgg16_tiny'

    if not os.path.exists(adv_save_dir):
        os.makedirs(adv_save_dir)
    
    if not os.path.exists(pertub_save_dir):
        os.makedirs(pertub_save_dir)

    for img in tqdm(img_list):
        img_path = os.path.join(image_root, img)

        pertub, adv = haf_attack.generate_adv_samples(x=img_path, stage=[0], p=2, M=300, eplison=0.1)

        cv2.imwrite(os.path.join(pertub_save_dir, img), pertub)
        cv2.imwrite(os.path.join(adv_save_dir, img), adv)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    image_list = os.listdir('data/coco2017/images/val2017')
    img_l1 = image_list[:2500]
    img_l2 = image_list[2500:]

    p1 = Process(target=generate_and_save, args=(img_l1, 'cuda:0'))
    p2 = Process(target=generate_and_save, args=(img_l2, 'cuda:1'))

    # start
    p1.start()
    p2.start()

    # end
    p1.join()
    p2.join()


