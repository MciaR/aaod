from visualizer import ExpVisualizer
from attack import *
from setting import get_attacker_params, get_remain_list
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def execute_attack(attacker_name, model_name, dataset_name, exp_name, start, end):
    """Execute specific attack.
    Args:
        attacker_name (str): attacker name.
        model_name (str): model config name.
        dataset_name (str): dataset name. Options: `COCO` or `VOC`. 
        exp_name (str): experience name, decide folder which result will be saved.
        start (int): start index of dataset image.
        end (int): end index of dataset image.
    """
    attacker_params = get_attacker_params(model_name, dataset_name, attacker_name)
    remain_list = get_remain_list(attacker_name)
    attacker_params.update({'exp_name': f'{model_name}/{exp_name}'})

    # expVisualizer params
    show_features = True
    show_lvl_preds = True
    save_analysis = True

    if model_name == "FR_VGG16":
        show_lvl_preds = False

    if attacker_name == 'FRMR':
        if model_name == 'DINO' or model_name == 'CenterNet' or model_name == 'SSD300':
            show_lvl_preds = False
        attacker = FRMRAttack(**attacker_params)
    elif attacker_name == 'THA':
        attacker = THAAttack(**attacker_params)
    elif attacker_name == 'EXPDAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = EXPDAGAttack(**attacker_params)
    elif attacker_name == 'DAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = DAGAttack(**attacker_params)
    elif attacker_name == 'EDAG':
        show_features = False
        show_lvl_preds = False
        save_analysis = False
        attacker = EDAGAttack(**attacker_params)
    elif attacker_name == 'Fusion':
        if model_name == 'DINO' or model_name == 'CenterNet' or model_name == 'SSD300':
            show_lvl_preds = False
        save_analysis = False
        attacker = FusionAttack(**attacker_params)
    elif attacker_name == 'RN':
        attacker = RandomNoise(**attacker_params)

    vis = ExpVisualizer(cfg_file=attacker_params['cfg_file'], ckpt_file=attacker_params['ckpt_file'], use_attack=True, attacker=attacker)
    dataset = vis.dataset

    for i in range(start, end):
        vis.show_attack_results(model_name=model_name + '_' + dataset_name, data_sample=dataset[i]['data_samples'], dataset_idx=i, save=True, feature_grey=False, remain_list=remain_list, 
                                show_features=show_features, show_lvl_preds=show_lvl_preds, save_analysis=save_analysis, show_thr=0.3)

if __name__ == '__main__':
    execute_attack(attacker_name='EDAG', model_name='DINO', dataset_name='COCO', exp_name='edag_coco_bbox_filter_exp', start=0, end=1)
