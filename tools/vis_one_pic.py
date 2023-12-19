import cv2
import mmcv
import os
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/faster_rcnn_r101_fpn_coco.py'
checkpoint_file = 'pretrained/fr_r101_coco.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# 测试单张图片并展示结果
img = 'records/attack_pics/EFMRAttack/FR_R101/500_anchors_gamma05_iou99_png/adv/000000397133.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)

# 显示结果
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True,
    pred_score_thr=0.3)

visualizer.show()