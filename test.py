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

# Create a random tensor
tensor = torch.tensor([[
    [
        [1, 1, 1],
        [2, 2, 2]
    ],
    [
        [3, 3, 3],
        [4, 4, 4]
    ],
    [
        [5, 5, 5],
        [6, 6, 6]
    ]
]], dtype=torch.float32)
bb_outputs = [tensor]

# reshaped_tensor = tensor.flatten()  # -1 infers the size based on other dimensions
# max_values, max_values_index = torch.topk(reshaped_tensor, k=10, dim=0)
# print(max_values, max_values_index)

def get_topk_info(
        input: torch.Tensor,
        k: int = 10,
        largest: bool = True,
        ):
    flatten_tensor = input.flatten()
    values, topk_indices = torch.topk(input=flatten_tensor, k=k, dim=0, largest=largest)
    assert len(input.shape) == 2, \
        f' featmap tensor must be shape (H, W)'
    H, W = input.shape

    h_indices = topk_indices // W
    w_indices = topk_indices % W

    indices = torch.stack((h_indices, w_indices), dim=1)

    return values, indices

def modify_featmap(
        featmap: torch.Tensor,
        modify_percent: float = 0.4,
        scale_factor: float = 0.2):
    """Modify topk value in each featmap (H, W).
    Args:
        featmap (torch.Tensor): shape `(C, H, W)`
        mean_featmap
        scale_factor (float): miniumize factor
    """
    C, H, W = featmap.shape
    k = int(H * W * modify_percent)
    mean_featmap = torch.mean(featmap, dim=0)
    _, topk_indices = get_topk_info(input=mean_featmap, k=k, largest=True)

    # scale indices value in each featmap
    featmap[:, topk_indices[:, 0], topk_indices[:, 1]] = featmap[:, topk_indices[:, 0], topk_indices[:, 1]] * scale_factor

    return featmap

def attack_method(bb_outputs):
    """ Find mean featmap max and min activate value pixel, and switch them."""
    attack_result = []
    out_len = len(bb_outputs)
    for i in range(out_len):
        feat_maps = bb_outputs[i]
        # feat_maps: (1, C, H, W)
        featmap = feat_maps.squeeze()
        attack_result.append(modify_featmap(featmap=featmap).unsqueeze(0))
    
    return attack_result

print(bb_outputs)
print(attack_method(bb_outputs=bb_outputs))
