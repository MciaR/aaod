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
# import torch

# # 创建一个示例张量
# tensor = torch.rand((2, 3, 4))

# # 计算张量的2范数
# norm = torch.norm(tensor, p=2)

# print(norm.item())  # 输出: 5.4772257804870605
# from PIL import Image
# import numpy as np
# import cv2

# img_list = np.random.rand(640, 640, 3) * 255
# img_np = np.array(img_list).astype(np.uint8)
# img = Image.fromarray(img_np)
# img.save('test.png')

# _img = Image.open('test.png')
# _img_np = np.array(_img)

# cv2.imwrite('test_cv2.png', img_np)
# cv_img_np = cv2.imread('test_cv2.png')

# print(img_np)
# print(_img_np)
# print(img_np == _img_np) # if png, is TRUE, if jpg or jpeg, is FALSE, because of the compress
# print(img_np == cv_img_np)

# ========================== heatmap test ==============================
# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# def get_topk_info(
#         input: torch.Tensor,
#         k: int = 3,
#         largest: bool = True,
#         ):
#     flatten_tensor = input.flatten()
#     values, topk_indices = torch.topk(input=flatten_tensor, k=k, dim=0, largest=largest)
#     assert len(input.shape) == 2, \
#         f' featmap tensor must be shape (H, W)'
#     H, W = input.shape

#     h_indices = topk_indices // W
#     w_indices = topk_indices % W

#     indices = torch.stack((h_indices, w_indices), dim=1)

#     return values, indices

# def generate_featmap():

#     a = torch.tensor([[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]])
#     scale_factor = 0.01
#     print(a)
#     # topk_values, topk_indices = get_topk_info(a)
#     # print(topk_values, topk_indices)
#     topk_indices = torch.tensor([[0, 0], [0, 1], [0, 2]])
#     b = a.clone()
#     a[topk_indices[:, 0], topk_indices[:, 1]] = a[topk_indices[:, 0], topk_indices[:, 1]] * scale_factor
#     print(a)

#     return b, a, topk_indices

# def cvt_feat2heat(
#         feat_map: torch.Tensor,
#         alpha = 0.5,
#         img = None,
#         grey = False,
#         ):
    
#     if isinstance(feat_map, torch.Tensor):
#         feat_map = feat_map.cpu().detach().numpy()

#     norm_img = np.zeros(feat_map.shape)
#     norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
#     norm_img = np.asarray(norm_img, dtype=np.uint8)
#     if grey:
#         heat_img = np.stack((norm_img,) * 3, -1)
#     else:
#         heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
#         heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
#     if img is not None:
#         heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
#     return heat_img

# row, col = (1, 2)
# plt.figure(frameon=False, figsize=(12, 3), dpi=300)
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

# ori_feat, modified_feat, indices = generate_featmap()

# ori_heatmap = cvt_feat2heat(ori_feat)
# plt.subplot(row, col, 1)
# plt.imshow(ori_heatmap)
# plt.title(f"ori heatmap", fontsize=10)

# modified_heatmap = cvt_feat2heat(modified_feat)
# plt.subplot(row, col, 2)
# plt.imshow(modified_heatmap)
# plt.title(f"modified heatmap", fontsize=10)

# plt.tight_layout()
# plt.show()

# ================= heatmap convert uni-space ================
# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# a = torch.randint(0, 256, (64, 64)).cpu().detach().numpy()
# b = torch.randint(50, 101, (64, 64)).cpu().detach().numpy()

# def map2uni_space(target: np.ndarray, src: np.ndarray):
#     H, W = src.shape
#     target_min = np.min(target)
#     target_max = np.max(target)

#     src_min_idx = np.argmin(src.flatten())
#     src_min_x, src_min_y = src_min_idx // W, src_min_idx % W
#     src_max_idx = np.argmax(src.flatten())
#     src_max_x, src_max_y = src_max_idx // W, src_max_idx % W

#     output = src.copy()
#     # print(output[src_min_x, src_min_y] == np.min(src))
#     output[src_min_x, src_min_y] = target_min
#     output[src_max_x, src_max_y] = target_max

#     return output

# def cvt_feat2heat(
#         feat_map: torch.Tensor,
#         alpha = 0.5,
#         img = None,
#         grey = False,
#         ):
    
#     if isinstance(feat_map, torch.Tensor):
#         feat_map = feat_map.cpu().detach().numpy()

#     norm_img = np.zeros(feat_map.shape)
#     norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
#     norm_img = np.asarray(norm_img, dtype=np.uint8)
#     if grey:
#         heat_img = np.stack((norm_img,) * 3, -1)
#     else:
#         heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
#         heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
#     if img is not None:
#         heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
#     return heat_img

# row, col = (2, 2)
# plt.figure(frameon=False, figsize=(12, 3), dpi=300)
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

# idx = 1
# ori_feat_list = [a, b]
# for i in range(col):
#     _featmap = ori_feat_list[i]
#     plt.subplot(row, col, idx)
#     plt.imshow(cvt_feat2heat(_featmap))
#     plt.title(f"ori heatmap {i}", fontsize=10)
#     idx += 1

# uni_feat_list = [a, map2uni_space(a, b)]
# for i in range(col):
#     _featmap = uni_feat_list[i]
#     plt.subplot(row, col, idx)
#     plt.imshow(cvt_feat2heat(_featmap))
#     plt.title(f"uni heatmap {i}", fontsize=10)
#     idx += 1

# plt.tight_layout()
# plt.show()

# import torch


# def F(x):
#     mean_val = torch.mean(x)
#     print(mean_val)
#     return (1 - torch.sin(torch.pi * (x - 0.5))) * mean_val / x

# featmap = torch.rand(8)
# print(featmap)
# scale = F(featmap)
# print(scale)

# print(featmap * scale)
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt


# os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# C = 2048
# channel_value = torch.rand(1, C)
# np_channel_value = channel_value.cpu().detach().numpy()
# x = np.linspace(1, C, C)

# plt.scatter(x, np_channel_value)
# plt.show()

# import torch
# from torchvision.models import resnet50

# model = resnet50()
# print(model)

# import torch
# bboxes1 = torch.randint(0, 224, (5, 4))
# bboxes2 = torch.randint(0, 224, (6, 4))

# def pairwise_iou(bboxes1, bboxes2):
#     """Calculate each pair iou in bboxes1 and bboxes2.
#     Args:
#         bboxes1 (torch.Tensor): shape is (N, 4)
#         bboxes2 (torch.Tensor): shape is (M, 4)
#     Returns:
#         paired_ious (torh.Tensor): shape is (N, M)
#     """
#     N, M = bboxes1.shape[0], bboxes2.shape[0]
#     area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
#     area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

#     ious = []

#     for i in range(N):
#         bbox1 = bboxes1[i]
#         xmin = torch.max(bbox1[0], bboxes2[:, 0])
#         ymin = torch.max(bbox1[1], bboxes2[:, 1])
#         xmax = torch.min(bbox1[2], bboxes2[:, 2])
#         ymax = torch.min(bbox1[3], bboxes2[:, 3])

#         w = xmax - xmin
#         h = ymax - ymin
#         inter = w * h 
#         iou = inter / (area1[i] + area2 - inter)
#         ious.append(iou)

#     ious = torch.stack(ious, dim=0)
#     return ious

# print(pairwise_iou(bboxes1, bboxes2).shape)

import numpy as np
import torch
from collections import Counter
# M = 3, N = 5
# pred_scores = torch.tensor([[0.1, 0.9], [0.4, 0.6], [0.3, 0.7]])
# paired_idx = torch.tensor([2, 1, 4])
# gt_labels = torch.tensor([1, 0, 1, 0, 1])

# #cls_idx = gt_labels[paired_idx]
# cls_idx_repeat = gt_labels.repeat(3, 1)
# cls_idx = cls_idx_repeat[torch.arange(3), paired_idx]
# print(cls_idx)
# paired_scores = pred_scores[torch.arange(pred_scores.shape[0]), cls_idx]
# print(paired_scores)
# score_cond = paired_scores > 0.1
# print(score_cond)
# idx = torch.tensor([1, 0, 0])
# nums = torch.tensor([[0.1, 0.9], [0.8, 0.4], [0.3, 0.7]])
# output = nums[torch.arange(nums.shape[0]), idx]
# print(output)

def update_positive_indices(positive_indices, active_mask):
    """
    Update positive_indices based on active_mask.
    Args:
        positive_indices (torch.Tensor): The original mask with shape (5000,).
        active_mask (torch.Tensor): The active mask with shape less than or equal to positive_indices.
    Returns:
        torch.Tensor: Updated positive_indices.
    """
    # 将 active_mask 映射回 positive_indices 的长度
    expanded_active_mask = torch.ones_like(positive_indices, dtype=torch.bool)
    expanded_active_mask[positive_indices] = active_mask

    # 更新 positive_indices：两个 mask 的逻辑与
    updated_positive_indices = positive_indices & expanded_active_mask

    return updated_positive_indices

positive_indices = torch.tensor([0, 0, 1, 0, 1]).bool()
active_mask = torch.tensor([0, 1]).bool()
print(update_positive_indices(positive_indices, active_mask))