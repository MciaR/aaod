import mmcv

from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from attack import *
from setting import get_attacker_params

def run_attack(img_path, model, dataset_name, attacker_name):
    attacker_params = get_attacker_params(model_name=model, dataset_name=dataset_name, attacker_name=attacker_name)

    if attacker_name == 'FRMR':
        attacker = FRMRAttack(**attacker_params)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif attacker_name == 'EDAG':
        attacker = EDAGAttack(**attacker_params)
    elif attacker_name == 'Fusion':
        attacker = FusionAttack(**attacker_params)

    attacker.attack(img=img_path, save_root='demos/')

    img = mmcv.imread(img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

if __name__ == '__main__':
    run_attack('data/test_img/roadview.png', 'FR_R101', 'COCO', 'EDAG')

    