# from PIL import Image
# import numpy as np
# import torch
# from mmengine.model.utils import stack_batch
# import mmcv
# from mmdet.datasets.transforms import Resize
# import cv2

# a = Image.open('data/coco2017/images/train2017/000000000009.jpg')
# a = np.array(a)
# a_resize, _ = mmcv.imrescale(
#                     a,
#                     (800, 1333),
#                     interpolation='bilinear',
#                     return_scale=True,
#                     backend='cv2')

# a_t = torch.tensor(a_resize).permute(2, 1, 0)

# mean=[123.675, 116.28, 103.53]
# std=[58.395, 57.12, 57.375]
# mean_t = torch.tensor(mean).view(-1, 1, 1)
# std_t = torch.tensor(std).view(-1, 1, 1)

# pic = (a_t - mean_t) / std_t
# pic = pic[[2, 1, 0], ...]

# # ================== Revert ======================

# # revert normorlize
# ori_pic = pic * std_t + mean_t
# # revert bgr_to_rgb
# ori_pic = ori_pic[[2, 1, 0], ...]
# # clip padding

# # (c, h, w) to (h, w, c)
# ori_pic = ori_pic.permute(2, 1, 0)

# ori_pic = ori_pic.detach().cpu().numpy()

# ori_pic, _ = mmcv.imrescale(
#                     ori_pic,
#                     a.shape[:2],
#                     interpolation='bilinear',
#                     return_scale=True,
#                     backend='cv2')

# print(a-ori_pic)
# cv2.imwrite('ori_pic.png', ori_pic)
import torch

# 创建一个示例张量
tensor = torch.rand((2, 3, 4))

# 计算张量的2范数
norm = torch.norm(tensor, p=2)

print(norm.item())  # 输出: 5.4772257804870605
