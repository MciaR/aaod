import torch
import torch.nn.functional as F
import numpy as np
import mmcv
import copy

from attack import BaseAttack
from visualizer import AnalysisVisualizer
from PIL import Image
from mmengine.structures import InstanceData
from torch.optim.lr_scheduler import StepLR


class EFMRAttack(BaseAttack):
    """Enhanced Feature Map Regression Attack. 
        gamma (float): scale factor of normalizing noise `r`.
        M (float): SGD total step.
    """
    def __init__(self, 
                 cfg_options,
                 cfg_file, 
                 ckpt_file,
                 exp_name=None,
                 gamma=0.5,
                 M=500,
                 device='cuda:0') -> None:
        assert cfg_options is not None, \
            f'`cfg_options` cannot be `None` for DAG Attack.'
        super().__init__(cfg_file, ckpt_file, device=device, cfg_options=cfg_options, exp_name=exp_name,
                         attack_params=dict(gamma=gamma, M=M))
        self.vis = AnalysisVisualizer(cfg_file=self.cfg_file, ckpt_file=self.ckpt_file)
        
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

    def get_targets(
            self,
            clean_image,
            data):
        """Get active RPN proposals. 
        Args:
            clean_image (torch.Tensor): clean image tensor after preprocess and transform.
            data (dict): a dict variable which send to `model.test_step()`.
        Returns:
            result (DetDataSample): result of pred.
        """
        data['inputs'] = clean_image
        # forward the model
        with torch.no_grad():
            # results = self.model.predict(x, batch_data_samples=data['data_samples'])[0]
            batch_data_samples = data['data_samples']
            batch_inputs = data['inputs']

            x = self.model.extract_feat(batch_inputs)
            # If there are no pre-defined proposals, use RPN to get proposals
            rpn_results_list_rescale = self.model.rpn_head.predict(
                x, batch_data_samples, rescale=True) # rescale to origin image size.
            
            rpn_results_list = self.model.rpn_head.predict(
                x, batch_data_samples, rescale=False)
            
            results_list = self.model.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=True)

            batch_data_samples = self.model.add_pred_to_datasample(
                batch_data_samples, results_list)
            
        proposal_bboxes = rpn_results_list_rescale[0].bboxes
        pred_scores = batch_data_samples[0].pred_instances.scores
        gt_bboxes = batch_data_samples[0].gt_instances.bboxes.to(self.device) # gt_bboxes is also original image size.
        gt_labels = batch_data_samples[0].gt_instances.labels.to(self.device)
        num_classes = pred_scores.shape[1]

        # use rescaled bbox to select positive proposals.
        # _base_exp_name = f'{self.get_attack_name()}/{self.exp_name}'
        # self.vis.visualize_bboxes(proposal_bboxes, batch_data_samples[0].img_path, exp_name=f'proposal_show/{_base_exp_name}', customize_str='original')
        _, positive_labels, remains, positive_proposal2gt_idx = self.select_positive_proposals(proposal_bboxes, pred_scores, gt_bboxes, gt_labels)
        # self.vis.visualize_bboxes(proposal_bboxes[remains], batch_data_samples[0].img_path, exp_name=f'proposal_show/{_base_exp_name}', customize_str='filtered')
        # self.vis.visualize_category_amount(proposal_bboxes[remains], gt_bboxes, positive_proposal2gt_idx, batch_data_samples[0].img_path, exp_name=f'gt2proposal_amount/{_base_exp_name}')

        # get un-rescaled bbox and corresponding scores
        active_rpn_instance = InstanceData()
        active_rpn_instance.bboxes = rpn_results_list[0].bboxes[remains] # rescaled, not original image size.
        active_rpn_instance.labels = rpn_results_list[0].labels[remains]

        rpn_results_list[0] = active_rpn_instance

        return rpn_results_list, positive_labels, num_classes, rpn_results_list_rescale[0].bboxes[remains] # for analysis process of proposal attacking.
    
    @staticmethod
    def pairwise_iou(bboxes1, bboxes2):
        """Calculate each pair iou in bboxes1 and bboxes2.
        Args:
            bboxes1 (torch.Tensor): shape is (N, 4)
            bboxes2 (torch.Tensor): shape is (M, 4)
        Returns:
            paired_ious (torh.Tensor): shape is (N, M)
        """
        N, M = bboxes1.shape[0], bboxes2.shape[0]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        ious = []

        for i in range(N):
            bbox1 = bboxes1[i]
            xmin = torch.max(bbox1[0], bboxes2[:, 0])
            ymin = torch.max(bbox1[1], bboxes2[:, 1])
            xmax = torch.min(bbox1[2], bboxes2[:, 2])
            ymax = torch.min(bbox1[3], bboxes2[:, 3])

            w = torch.clamp(xmax - xmin, min=0)
            h = torch.clamp(ymax - ymin, min=0)
            inter = w * h 
            iou = inter / (area1[i] + area2 - inter)
            ious.append(iou)

        ious = torch.stack(ious, dim=0)
        return ious

    
    def select_positive_proposals(
            self,
            proposal_bboxes: torch.Tensor, 
            pred_scores: torch.Tensor, 
            gt_bboxes: torch.Tensor, 
            gt_labels: torch.Tensor):
        """Select high-quality targets for Attack.
        Args:
            proposal_bboxes (torch.Tensor): pred bboxes, shape is (proposal_num, 80 * 4).
            pred_scores (torch.Tensor): pred scores, shape is (proposal_num, 80 + 1).
            gt_bboxes (torch.Tensor): gt bboxes, shape is (gt_num, 4).
            gt_labels (torch.Tensor): gt bboxes, shape is (gt_num, 1).
        Returns:
            positive_bboxes (torch.Tensor): remaining high quality targets bboxes.
            positive_scores (torch.Tensor): remaining high quality targets scores.
            remains (torch.Tensor[bool]): filter flag for proposal_bboxes and its label.
        """
        # cal iou
        ious = self.pairwise_iou(proposal_bboxes, gt_bboxes)

        # find the max iou gt_bboxes for each proposal_bboxes, 
        # that means every proposal_bboxes just has one gt_bboxes to pair.
        paired_ious, paired_gt_idx = ious.max(dim=1)

        # Filter for ious > 0.1
        iou_remains = paired_ious > 0.1

        # Filter for score of proposal > 0.1
        # NOTE: Below 2 sentence is equals to `label_idx = gt_labels[paired_gt_idx].`
        # cls_idx_repeat = gt_labels.repeat(proposal_num, 1)
        # label_idx = cls_idx_repeat[torch.arange(proposal_num), paired_gt_idx]
        label_idx = gt_labels[paired_gt_idx] # (proposal_num, 1) get the label indices of gt bboxes which paired to proposal_bboxes.
        paired_scores = pred_scores[torch.arange(pred_scores.shape[0]), label_idx] # (proposal_num, 1) get the scores corresponding to paried bboxes.
        score_remains = paired_scores > 0.1

        # Filter for positive proposals and their correspoinding gt labels
        remains = iou_remains & score_remains

        return proposal_bboxes[remains], label_idx[remains], remains, paired_gt_idx[remains]

    
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
    
    def update_target_rpn_results(
            self,
            target_rpn_results,
            remains,
    ):
        """Filter for active target_rpn_results (rpn proposals and labels).
        Args:
            target_rpn_results (List[DetDatasample]): Default length is 1.
            remains (torch.tensor[bool]): filter flag for proposal_bboxes and its label.
        """
        active_rpn_instance = InstanceData()
        active_rpn_instance.bboxes = target_rpn_results[0].bboxes[remains]
        active_rpn_instance.labels = target_rpn_results[0].labels[remains]

        target_rpn_results[0] = active_rpn_instance

        return target_rpn_results

    def generate_adv_samples(self, x, data_sample, log_info=True):
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
        data['data_samples'][0].gt_instances = data_sample.gt_instances
        batch_data_samples = data['data_samples']

        # get target labels and proposal bboxes which from RPN
        target_rpn_results, target_labels, num_classes, attack_proposals_img_scale = self.get_targets(clean_image, data)
        # get adv labels
        adv_labels = self.get_adv_targets(target_labels, num_classes=num_classes)

        # pertubed image, `X_m` in paper
        pertubed_image = clean_image.clone()
        pertubed_image.requires_grad = True

        step = 0
        total_targets = len(target_labels)
        loss_metric = torch.nn.CrossEntropyLoss(reduction='sum')

        if log_info:
            print(f'Start generating adv, total rpn proposal: {total_targets}.')

        # record attack process of proposals
        accum_proposals = []

        while step < self.M:

            # get features
            features = self.model.extract_feat(pertubed_image)

            predict_list = self.model.roi_head.predict(
                features, target_rpn_results, batch_data_samples, rescale=True)
            
            logits = predict_list[0].scores

            # remain the correct targets, drop the incorrect i.e. successful attack targets.
            # active_target_idx &= (logits.argmax(dim=1) != adv_labels)
            active_target_idx = logits.argmax(dim=1) != adv_labels

            active_logits = logits[active_target_idx]
            target_labels = target_labels[active_target_idx]
            adv_labels = adv_labels[active_target_idx]
            target_rpn_results = self.update_target_rpn_results(target_rpn_results, active_target_idx)

            # if all targets has been attacked successfully, attack ends.
            if len(target_labels) == 0:
                break

            # comput loss
            correct_loss = loss_metric(active_logits, target_labels)
            adv_loss = loss_metric(active_logits, adv_labels)
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

            attacked_this_round_proposals = attack_proposals_img_scale[~active_target_idx]
            accum_proposals.append(attacked_this_round_proposals)

            if step % 10 == 0 and log_info:
                print("Generation step [{}/{}], loss: {}, attack percent: {}%.".format(step, self.M, total_loss, (total_targets - len(target_labels)) / total_targets * 100))
                _exp_name = f'{self.get_attack_name()}/{self.exp_name}'
                self.vis.visualize_intermediate_results(r=self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0]),
                                                        r_total = self.reverse_augment(x=pertubed_image.squeeze()-clean_image.squeeze(), datasample=data['data_samples'][0]),
                                                        pertubed_image=self.reverse_augment(x=pertubed_image.squeeze(), datasample=data['data_samples'][0]),
                                                        customize_str=step,
                                                        attack_proposals=torch.cat(accum_proposals, dim=0),
                                                        image_path=data['data_samples'][0].img_path,
                                                        exp_name=_exp_name)
                accum_proposals = []
            attack_proposals_img_scale = attack_proposals_img_scale[active_target_idx] # for analysis process of proposal attacking    
            step += 1

        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = pertubed_image.squeeze() - clean_image.squeeze()
        adv_tensor = pertubed_image.squeeze()

        pertub = self.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print(f"Generate adv compeleted! Cost iterations {step}.")

        return pertub, adv_image
