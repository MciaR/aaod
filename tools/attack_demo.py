import os
import numpy as np

from visualizer import AAVisualizer
from PIL import Image
import matplotlib.pyplot as plt
from attack import (FRMRAttack, THAAttack, EDAGAttack, FusionAttack)
from setting import get_attacker_params, get_model

def run_attack(img_path, model, dataset_name, attacker_name):
    attacker_params = get_attacker_params(model_name=model, dataset_name=dataset_name, attacker_name=attacker_name)

    # NOTE: DO NOT SUPPORT DAG, because DAG relys on GT, so can't infer a natural image without annotations.
    if attacker_name == 'FRMR':
        attacker = FRMRAttack(**attacker_params)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif attacker_name == 'EDAG':
        attacker = EDAGAttack(**attacker_params)
    elif attacker_name == 'Fusion':
        attacker = FusionAttack(**attacker_params)

    _, adv =  attacker.generate_adv_samples(x=img_path)
    clean = Image.open(img_path)

    adv_image = adv.astype(np.uint8)
    clean_image = np.array(clean)

    cfg_path, ckpt_path = get_model(model_name=model, dataset_name=dataset_name)
    vis = AAVisualizer(cfg_path, ckpt_path)

    adv_pred = vis.get_pred(adv_image)
    clean_pred = vis.get_pred(clean_image)

    adv_result = vis.draw_dt_gt(
        name='adv',
        image=adv_image,
        draw_gt=False,
        data_sample=adv_pred,
        pred_score_thr=0.3)

    clean_result = vis.draw_dt_gt(
        name='clean',
        image=clean_image,
        draw_gt=False,
        data_sample=clean_pred,
        pred_score_thr=0.3)

    assert adv_result.shape[0] == clean_result.shape[0], \
        f'The height of two images must be same, but got {adv_result.shape[0]} and {clean_result.shape[0]}.'

    demo_np = np.concatenate([clean_result, adv_result], axis=1)
    demo = Image.fromarray(demo_np)

    save_path = 'demo'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, f'{attacker_name}_{model}_{dataset_name}.jpg')
    demo.save(save_file)

    # seperately save
    Image.fromarray(adv_result).save(save_path + '/' + 'adv_result.pdf')
    Image.fromarray(clean_result).save(save_path + '/' + 'clean_result.pdf')


if __name__ == '__main__':
    run_attack('data/VOCdevkit/VOC2007_test/JPEGImages/000252.jpg', 'SSD300', 'VOC', 'FRMR')
    # run_attack('data/coco2017/images/val2017/000000000785.jpg', 'CenterNet', 'COCO', 'FRMR')

    