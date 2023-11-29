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

# img_list = np.random.rand(640, 640, 3) * 255
# img_np = np.array(img_list).astype(np.uint8)
# img = Image.fromarray(img_np)
# img.save('test.png')

# _img = Image.open('test.png')
# _img_np = np.array(_img)

# print(img_np)
# print(_img_np)
# print(img_np == _img_np) # if png, is TRUE, if jpg or jpeg, is FALSE, because of the compress

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
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


a = torch.randint(0, 256, (64, 64)).cpu().detach().numpy()
b = torch.randint(50, 101, (64, 64)).cpu().detach().numpy()

def map2uni_space(target: np.ndarray, src: np.ndarray):
    H, W = src.shape
    target_min = np.min(target)
    target_max = np.max(target)

    src_min_idx = np.argmin(src.flatten())
    src_min_x, src_min_y = src_min_idx // W, src_min_idx % W
    src_max_idx = np.argmax(src.flatten())
    src_max_x, src_max_y = src_max_idx // W, src_max_idx % W

    output = src.copy()
    # print(output[src_min_x, src_min_y] == np.min(src))
    output[src_min_x, src_min_y] = target_min
    output[src_max_x, src_max_y] = target_max

    return output

def cvt_feat2heat(
        feat_map: torch.Tensor,
        alpha = 0.5,
        img = None,
        grey = False,
        ):
    
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.cpu().detach().numpy()

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    if grey:
        heat_img = np.stack((norm_img,) * 3, -1)
    else:
        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img

row, col = (2, 2)
plt.figure(frameon=False, figsize=(12, 3), dpi=300)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

idx = 1
ori_feat_list = [a, b]
for i in range(col):
    _featmap = ori_feat_list[i]
    plt.subplot(row, col, idx)
    plt.imshow(cvt_feat2heat(_featmap))
    plt.title(f"ori heatmap {i}", fontsize=10)
    idx += 1

uni_feat_list = [a, map2uni_space(a, b)]
for i in range(col):
    _featmap = uni_feat_list[i]
    plt.subplot(row, col, idx)
    plt.imshow(cvt_feat2heat(_featmap))
    plt.title(f"uni heatmap {i}", fontsize=10)
    idx += 1

plt.tight_layout()
plt.show()