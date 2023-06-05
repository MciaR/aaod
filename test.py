import torch
import cv2
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# pics = torch.rand((1, 256, 188, 334))
# topk = 10
# featmap = pics.squeeze()
# mean_fm = torch.mean(featmap, dim=0)


# def get_extremum_activates(featmap, topk=10, largest=True):
#     # inputs: (H, W)
#     shape = featmap.shape
#     assert len(shape) == 2, \
#         f'inputs must be a Tensor with shape (h, w).'
    


# img_path = 'data/coco2014/images/train2014/COCO_train2014_000000000009.jpg'
# img = cv2.imread(img_path)

# a = torch.Tensor(img)
# a = a.permute(2, 0, 1)
# results = F.interpolate(
#     a[None],
#     (a.shape[1] * 2, a.shape[2] * 2),
#     mode='bilinear',
#     align_corners=False)[0]
# plt.imshow(results[0].numpy())
# plt.show()



