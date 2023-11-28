from visualizer import ExpVisualizer
from attack import HAFAttack
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def parse_args():
    pass


if __name__ == '__main__':
    # 指定模型的配置文件和 checkpoint 文件路径
    # config_file = 'configs/fr_vgg16_voc.py'
    # checkpoint_file = 'test_model/fr_vgg_16_voc_0638.pth'
    config_file = 'configs/faster_rcnn_r101_fpn_coco.py'
    checkpoint_file = 'pretrained/fr_r101_coco_0394.pth'
    # config_file = 'configs/ssd300_coco.py'
    # checkpoint_file = 'pretrained/ssd/ssd300_coco_0255.pth'
    # config_file = 'configs/faster_rcnn_r101_dcn_c3_c5_fpn_coco.py'
    # checkpoint_file = 'pretrained/faster_rcnn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth'

    # initialize attacker
    attack_params = {
        'cfg_file': config_file,
        'ckpt_file': checkpoint_file,
        'feature_type': 'neck',
        'channel_mean': False,
        'stage': [4],
        'p': 2,
        'alpha': 2,
        'lr': 0.05,
        'M': 10,
        'adv_type': 'direct', 
        # 'adv_type': 'residual',
        # 'constrain': 'distance'
        'constrain': 'consine_sim'
    }

    # field which will be saved in result name.
    remain_list = ['feature_type', 'channel_mean', 'stage', 'alpha', 'lr', 'M', 'adv_type', 'constrain']
    # decide folder which result will be saved.
    exp_name = 'reduce_std_by_each_channel/dataset_top200'

    attacker = HAFAttack(**attack_params) 
    vis = ExpVisualizer(cfg_file=config_file, ckpt_file=checkpoint_file, use_attack=True, attacker=attacker)
    dataset = vis.dataset

    # show_layer = 3 # (256, 512, 1024, 2048) channels
    # top_k = 100
    # pic_overlay = False

    # vis.show_single_pic_feats(img=img, show_layer=3, top_k=top_k, pic_overlay=pic_overlay)
    # vis.show_stage_results(data_sample=dataset[60]['data_samples'], save=True, grey=True, show_thr=0.3)
    for i in range(200):
        vis.show_attack_results(model_name="FR_R101_COCO", data_sample=dataset[i]['data_samples'], save=True, feature_grey=False, attack_params=attack_params, remain_list=remain_list, exp_name=exp_name)
    # vis.show_stage_results(img='ad_result/base_attack/000000397133.jpg', save=False, grey=True, show_thr=0.1)
