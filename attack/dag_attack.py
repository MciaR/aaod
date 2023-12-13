import torch
import torch.nn.functional as F
import numpy as np
import mmcv
import random

from attack import BaseAttack
from PIL import Image
from torch.optim.lr_scheduler import StepLR


class DAGAttack(BaseAttack):
    """Dense Adversary Generation. 
    Impelemented from paper: (Xie C, Wang J, Zhang Z, et al. Adversarial examples for semantic segmentation and object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 1369-1378.)
    Args:         
        gamma (float): scale factor of normalizing noise `r`.
        M (float): SGD total step, if iter reach the limit or every RP has been attack, the loop ends (for DAG).
    """
    def __init__(self, 
                 cfg_options,
                 cfg_file="configs/faster_rcnn_r101_fpn_coco.py", 
                 ckpt_file="pretrained/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth",
                 gamma=0.5,
                 M=500,
                 device='cuda:0') -> None:
        assert cfg_options is not None, \
            f'`cfg_options` cannot be `None` for DAG Attack.'
        super().__init__(cfg_file, ckpt_file, device=device, cfg_options=cfg_options,
                         attack_params=dict(gamma=gamma, M=M))
        
    def reverse_augment(self, x, datasample):
        """Reverse tensor to input image."""
        ori_shape = datasample.ori_shape
        pad_shape = datasample.pad_shape

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375] # for fr
        # std = [1, 1, 1] # for ssd
        mean_t = torch.tensor(mean, device=self.device).view(-1, 1, 1)
        std_t = torch.tensor(std, device=self.device).view(-1, 1, 1)

        # revert normorlize
        ori_pic = x * std_t + mean_t
        # revert bgr_to_rgb
        # NOTE: dont need to revert bgr_to_rgb, beacuse saving format is RGB if using PIL.Image
        # ori_pic = ori_pic[[2, 1, 0], ...]
        # revert pad
        ori_pic = ori_pic[:, :datasample.img_shape[0], :datasample.img_shape[1]]

        # (c, h, w) to (h, w, c)
        ori_pic = ori_pic.permute(1, 2, 0)
        # cut overflow values
        ori_pic = torch.clamp(ori_pic, 0, 255)

        ori_pic = ori_pic.detach().cpu().numpy()

        # for fr
        ori_pic, _ = mmcv.imrescale(
                            ori_pic,
                            ori_shape,
                            interpolation='bilinear',
                            return_scale=True,
                            backend='cv2')
        # ori_pic, _, _ = mmcv.imresize(
        #                     ori_pic,
        #                     (ori_shape[1], ori_shape[0]),
        #                     interpolation='bilinear',
        #                     return_scale=True,
        #                     backend='cv2')
        
        return ori_pic    
    
    def get_pred_results(
            self,
            x,
            data):
        """Get pred result of x.
        Args:
            x (torch.Tensor): image tensor after preprocess and transform.
            data (dict): a dict variable which send to `model.test_step()`.
        Returns:
            result (DetDataSample): result of pred.
        """
        data['inputs'] = x
        # forward the model
        with torch.no_grad():
            results = self.model.predict(x, batch_data_samples=data['data_samples'])[0]

        return results.pred_instances.scores
    
    def get_adv_targets(self, clean_labels: torch.Tensor, num_classes: int):
        """Assign a set of correct labels to adversarial labels randomly.
        Args:
            clean_labels (torch.Tensor): shape is (NMS_NUM, 1).
            num_classes (int): number of classes (including background cls).
        """
        # add a random num which range from [1, num_classes] to clean_labels and mod 81, then we get the adv_labels.
        # that makes adv_labels[i] != clean_labels[i], and every element in adv_labels range from [0, num_classes]
        adv_labels = (clean_labels + torch.randint(1, num_classes, (clean_labels.shape[0], ), device=self.device)) % num_classes

        return adv_labels

    def generate_adv_samples(self, x, log_info=True):
        """Attack method to generate adversarial image.
        Funcs:
            `Loss_total={{\sum}_{n=1}^N}[f_{l_n}(X + r,t_n) - f_{l'_n}(X + r,t_n)]` (you can find it in the DAG paper),
            that means for a noised image (X+r), suppressing the confidence of original correct class(l_n) while increasing the incorrect class(l'_n).
            It can implemented with substract of two loss: `correct_loss=loss_metric(logits, clean_labels)`, `adv_loss=loss_metric(logits, adv_labels)`,
            `Loss_total = adv_loss - correct_loss`, that makes adv_loss smaller and correct_loss bigger.
        Args:
            x (str): clean image path.
            log_info (bool): if print the train information.
        Return:
            noise (np.ndarray | torch.Tensor): niose which add to clean image.
            adv (np.ndarray | torch.Tensor): adversarial image.
        """
        # initialize r
        data = self.get_data_from_img(img=x)
        clean_image = data['inputs']

        # get clean labels
        clean_scores = self.get_pred_results(clean_image, data)
        clean_labels = clean_scores.argmax(dim=-1)
        # get adv labels
        adv_labels = self.get_adv_targets(clean_labels, num_classes=clean_scores.shape[1])

        # pertubed image, `X_m` in paper
        pertubed_image = clean_image.clone()
        pertubed_image.requires_grad = True

        step = 0
        loss_metric = torch.nn.CrossEntropyLoss(reduction='sum')

        while step < self.M:

            # get preds result, hack the predict, the batch_data_samples is fake.
            logits = self.model.predict(pertubed_image, batch_data_samples=data['data_samples'])[0].pred_instances.scores

            # remain the correct targets, drop the incorrect i.e. successful attack targets.
            active_target_idx = logits.argmax(dim=1) != adv_labels

            active_logits = logits[active_target_idx]
            active_clean_labels = clean_labels[active_target_idx]
            active_adv_labels = adv_labels[active_target_idx]

            # if all targets has been attacked successfully, attack ends.
            if len(active_clean_labels) == 0:
                break

            # comput loss
            correct_loss = loss_metric(active_logits, active_clean_labels)
            adv_loss = loss_metric(active_logits, active_adv_labels)
            # decreasing adv_loss to make pertubed image predicted wrong, and increasing correct_loss to let result far from original correct labels.
            total_loss = adv_loss - correct_loss

            # backward and comput pertubed image gradient 
            total_loss.backward()
            pertubed_image_grad = pertubed_image.grad.detach()

            with torch.no_grad():
                # Normalize grad, from paper Eq.(3)
                r = (self.gamma / pertubed_image_grad.norm(float("inf"))) * pertubed_image_grad 
                pertubed_image -= r # gradient reverse direction is the direction of decreasing total_loss

            # Zero gradients
            pertubed_image_grad.zero_()
            self.model.zero_grad()

            if step % 10 == 0 and log_info:
                print("Generation step [{}/{}], loss: {}, attack percent: {}.".format(step, self.M, total_loss, (len(clean_labels) - len(active_clean_labels)) / len(clean_labels) * 100))

            step += 1

        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = pertubed_image.squeeze() - clean_image.squeeze()
        adv_tensor = pertubed_image.squeeze()

        pertub = self.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print(f"Generate adv compeleted! Cost iterations {step}.")

        return pertub, adv_image
