import torch

from attack import BaseAttack
from visualizer import AnalysisVisualizer
from mmdet.models.utils import select_single_mlvl


class EDAGAttack(BaseAttack):
    """Enhanced DAG Attack. 
        gamma (float): scale factor of normalizing noise `r`.
        M (float): SGD total step.
    """
    def __init__(self, 
                 cfg_file, 
                 ckpt_file,
                 cfg_options,
                 exp_name=None,
                 gamma=0.5,
                 M=500,
                 active_score_thr=0.,
                 targeted=True,
                 model_name='fr',
                 attack_target: dict = None,
                 device='cuda:0') -> None:
        super().__init__(cfg_file, ckpt_file, device=device, exp_name=exp_name, cfg_options=cfg_options,
                         attack_params=dict(gamma=gamma, M=M, model_name=model_name, active_score_thr=active_score_thr, targeted=targeted))
        assert model_name in ['fr', 'ssd', 'dino', 'centernet'], \
            f'EDAG now just support `fr`, `ssd`, `dino` and `centernet` as model_name.'
        self.attack_target = attack_target
        self.vis = AnalysisVisualizer(cfg_file=self.cfg_file, ckpt_file=self.ckpt_file)


    def get_final_predicts_from_mlvl(self, scores_list, bbox_list):
        """Merge multi-level preds.
        Args:
            scores_list (List[torch.Tensor]): raw score prediction of single stage detector, each element is a 3-D Tensor, shape is (num_priors * num_classes, W, H).
            bbox_list (List[torch.Tensor]): raw bboxes prediction of single stage detector,  each element is a 3-D Tensor, shape is (num_priors * 4, W, H).
        Returns:
            pred_scores (torch.Tensor): total predict scores, shape is (N, num_classes).
            pred_bboxes (torch.Tensor): total predict bboxes, shape is (N, 4).
        """
        cls_out_channels = self.model.bbox_head.cls_out_channels
        bbox_dim = self.model.bbox_head.bbox_coder.encode_size
        mlvl_num_priors = [single_lvl_bbox.shape[0] // 4 for single_lvl_bbox in bbox_list]
        mlvl_scores = []
        mlvl_bboxes = []
        for level_idx, (cls_score, bbox_pred) in enumerate(zip(scores_list, bbox_list)):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, bbox_dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_out_channels)
            mlvl_scores.append(cls_score)
            mlvl_bboxes.append(bbox_pred)
        
        return torch.cat(mlvl_scores), torch.cat(mlvl_bboxes)

    def get_model_predicts(self, x, data_samples):
        """Get model raw prediction without postprocessing.
        Args:
            x (torch.Tensor): img tensor.
            data_samples (List[DetDataSample]): batch of data sample.
        Return:
            pred_scores (torch.Tensor): total predict scores, shape is (N, num_classes).
            pred_bboxes (torch.Tensor): total predict bboxes, shape is (N, 4).
        """
        if self.model_name == 'fr':
            # `mode='tensor'` means get raw output without postprocess.
            pred_scores, pred_bboxes = self.model(x, data_samples=data_samples, mode='tensor')[0]
        else:
            pred_scores, pred_bboxes = self.model(x, data_samples=data_samples, mode='tensor')

        if self.model_name == 'dino':
            # get last decoder output for this sample
            pred_scores = pred_scores[-1][0]
            pred_bboxes = pred_bboxes[-1][0]
        elif self.model_name == 'ssd' or self.model_name == 'centernet':
            # get output scores and bboxes
            sample_cls_scores_list = select_single_mlvl(pred_scores, 0, detach=False)
            sample_bbox_pred_list = select_single_mlvl(pred_bboxes, 0, detach=False)
            # ssd300 will get 8732 bboxes and centerNet will get 20267 bboxes
            pred_scores, pred_bboxes = self.get_final_predicts_from_mlvl(sample_cls_scores_list, sample_bbox_pred_list)

        return pred_scores, pred_bboxes
          
    def get_targets(
            self,
            clean_image,
            data_samples):
        """Get active RPN proposals. 
        Args:
            clean_image (torch.Tensor): clean image tensor after preprocess and transform.
            data_samples (List[DetDataSample]): batch of data sample.
        Returns:
            Return of `self.select_positive_targets().`
        """

        # forward the model
        with torch.no_grad():
            pred_scores, pred_bboxes = self.get_model_predicts(clean_image, data_samples)
            
        return self.select_positive_targets(pred_bboxes, pred_scores, data_samples[0])
    
    def nms(self, pred_bboxes, pred_scores, pred_labels, iou_thr=0.5, score_thr=0.3):
        """bboxes non-maxmium supression post processing.
        Args:
            pred_bboxes (torch.Tensor): pred bboxes, shape is (N, 4).
            pred_scores (torch.Tensor): pred scores, shape is (N, 1).
            pred_labels (torch.Tensor): pred labels, shape is (N, 1).
            iou_thr (float): remove bboxes whose iou with target > iou_thr.
            scores_thr (float): remove bboxes whose scores < scores_thr.
        Returns:
            result_bboxes (torch.Tensor): bboxes after nms.
            result_scores (torch.Tensor): scores after nms.
        """

        x1 = pred_bboxes[:, 0]
        y1 = pred_bboxes[:, 1]        
        x2 = pred_bboxes[:, 2]
        y2 = pred_bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        sorted_idx = torch.argsort(pred_scores, descending=True)
        round = 0
        keep = []

        while len(sorted_idx):
            # max score bboxes
            i = sorted_idx[0]
            keep.append(i)

            if len(sorted_idx) == 1:
                break

            inter_xmin = torch.maximum(x1[i], x1[sorted_idx[1:]])
            inter_ymin = torch.maximum(y1[i], y1[sorted_idx[1:]])
            inter_xmax = torch.minimum(x2[i], x2[sorted_idx[1:]])
            inter_ymax = torch.minimum(y2[i], y2[sorted_idx[1:]])

            w = torch.clamp(inter_xmax - inter_xmin, 0.)
            h = torch.clamp(inter_ymax - inter_ymin, 0.)

            inter_area = w * h
            ious = inter_area / (areas[i] + areas[sorted_idx[1:]] - inter_area)
            
            idx = torch.where(ious <= iou_thr)[0]
            sorted_idx = sorted_idx[idx + 1]

            round += 1

        keep = torch.tensor(keep, device=self.device)
        pred_bboxes = pred_bboxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        valid_indices = (pred_scores >= score_thr)
        pred_bboxes = pred_bboxes[valid_indices]
        pred_scores = pred_scores[valid_indices]
        pred_labels = pred_labels[valid_indices]

        return pred_bboxes, pred_scores, pred_labels


    def select_positive_targets(
            self,
            pred_bboxes,
            pred_scores,
            data_sample,
            iou_thr=0.7,
            score_thr=0.1,
            ):
        """Select high-quality targets for Attack.
        Args:
            pred_bboxes (torch.Tensor): pred bboxes, shape is (proposal_num, 80 * 4).
            pred_scores (torch.Tensor): pred scores, shape is (proposal_num, 80 + 1).
            iou_thr (float): iou filter condition.
            score_thr (float): score filter condition.
        Returns:
            active_bboxes (torch.Tensor): remaining high quality targets bboxes.
            active_scores (torch.Tensor): remaining high quality targets scores.
            active_labels (torch.Tensor): remaining high quality targets labels.
            valid_indices (torch.Tensor): valid indices.
            C (int): num_classes of prediction, may equals to `real number of classes` + 1.
            remains (torch.Tensor[bool]): filter flag for proposal_bboxes and its label.

        """

        N, C = pred_scores.shape
        pred_scores, paired_label_idx = pred_scores.softmax(dim=-1).max(dim=-1)
        # that means has background class.
        if C == self.num_classes + 1:
            valid_indices = (paired_label_idx < C - 1)
        else:
            valid_indices = torch.ones_like(pred_scores, device=self.device).bool()

        # if pred_bboxes shape is (N, 4*80)
        if self.num_classes * 4 == pred_bboxes.shape[1]:
            pred_bboxes = pred_bboxes.reshape(N, -1, 4)
            pred_bboxes = pred_bboxes[valid_indices]
            active_bboxes = pred_bboxes[torch.arange(len(pred_bboxes), device=self.device), paired_label_idx[valid_indices]]
        else:
            active_bboxes = pred_bboxes[valid_indices]
        active_scores = pred_scores[valid_indices]
        active_labels = paired_label_idx[valid_indices]

        # _exp_name = f'effective_bboxes/{self.get_attack_name()}/{self.exp_name}'
        # self.vis.visualize_bboxes(active_bboxes, data_sample.img_path, exp_name=_exp_name, labels=active_labels, scores=active_scores, distinguished_color=True)

        # need to filter high quality bboxes.
        filter_indices = active_scores > self.active_score_thr
        active_scores = active_scores[filter_indices]
        active_bboxes = active_bboxes[filter_indices]
        active_labels = active_labels[filter_indices]
        
        # get final valid_indices
        expaned_filter_indices = torch.ones_like(valid_indices, dtype=torch.bool, device=self.device)
        expaned_filter_indices[valid_indices] = filter_indices
        valid_indices = valid_indices & expaned_filter_indices
        # final_bboxes, final_scores, final_labels = self.nms(active_bboxes, active_scores, active_labels)

        # _exp_name = f'nmsed_bboxes/{self.get_attack_name()}/{self.exp_name}'
        # self.vis.visualize_bboxes(active_bboxes, data_sample.img_path, exp_name=_exp_name, labels=active_labels, scores=active_scores, distinguished_color=True)

        return active_bboxes, active_scores, active_labels, valid_indices, C
    
    def get_adv_targets(self, clean_labels: torch.Tensor, num_classes: int):
        """Assign a set of correct labels to adversarial labels randomly.
        Args:
            clean_labels (torch.Tensor): shape is (NMS_NUM, 1).
            num_classes (int): number of classes (including background cls).
        """
        # add a random num which range from [1, num_classes] to clean_labels and mod 81, then we get the adv_labels.
        # that makes adv_labels[i] != clean_labels[i], and every element in adv_labels range from [0, num_classes]
        target2adv_labels = (torch.arange(0, num_classes, device=self.device) + torch.randint(1, num_classes, (num_classes, ), device=self.device)) % num_classes
        # update attack target
        if self.attack_target is not None:
            for key, value in self.attack_target.items():
                target2adv_labels[key] = value
        adv_labels = target2adv_labels[clean_labels]
        return adv_labels
    
    def update_positive_indices(self, positive_indices, active_mask):
        """
        Update positive_indices based on active_mask.
        Args:
            positive_indices (torch.Tensor): The original mask with shape (5000,).
            active_mask (torch.Tensor): The active mask with shape less than or equal to positive_indices.
        Returns:
            torch.Tensor: Updated positive_indices.
        """
        # 将 active_mask 映射回 positive_indices 的长度
        expanded_active_mask = torch.ones_like(positive_indices, dtype=torch.bool, device=self.device)
        expanded_active_mask[positive_indices] = active_mask

        # 更新 positive_indices：两个 mask 的逻辑与
        updated_positive_indices = positive_indices & expanded_active_mask

        return updated_positive_indices


    def generate_adv_samples(self, x, data_sample=None, log_info=True):
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
        batch_data_samples = data['data_samples']

        # get targets from predict
        target_bboxes, target_scores, target_labels, positive_indices, num_classes = self.get_targets(clean_image, batch_data_samples)
        # get adv labels
        adv_labels = self.get_adv_targets(target_labels, num_classes=num_classes)

        # pertubed image, `X_m` in paper
        pertubed_image = clean_image.clone()
        pertubed_image.requires_grad = True

        step = 0
        total_targets = len(target_labels)
        loss_metric = torch.nn.CrossEntropyLoss(reduction='sum')

        if log_info:
            print(f'Start generating adv, total attack bboxes: {total_targets}.')

        while step < self.M:

            # get features
            logits, pred_bboxes = self.get_model_predicts(pertubed_image, batch_data_samples)
            logits = logits.softmax(dim=-1)
            # NOTE: there may occur an error: `IndexError: The shape of the mask [500] at index 0 does not match the shape of the indexed tensor [0, 21] at index 0.`
            # that means logits shape is (0, 21), but dont know why output logits has no bbox. maybe is environment problem?
            if len(logits) == 0:
                break
            positive_logtis = logits[positive_indices] # logits corresponding with targets and advs.

            if self.targeted:
                # remain the correct targets, drop the incorrect i.e. successful attack targets.
                # active_target_idx &= (logits.argmax(dim=1) != adv_labels)
                active_target_mask = (positive_logtis.argmax(dim=1) != adv_labels)
                active_logits = positive_logtis[active_target_mask]
                target_labels = target_labels[active_target_mask]
                adv_labels = adv_labels[active_target_mask]

                # comput adv targeted loss
                adv_loss = loss_metric(active_logits, adv_labels)

                positive_indices = self.update_positive_indices(positive_indices, active_target_mask)
            else:
                active_logits = positive_logtis
                adv_loss = 0
            
            # if all targets has been attacked successfully, attack ends.
            if len(active_logits) == 0:
                break

            # comput loss
            correct_loss = loss_metric(active_logits, target_labels)

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
                if self.targeted:
                    print("Generation step [{}/{}], loss: {}, attack percent: {}%.".format(step, self.M, total_loss, (total_targets - len(active_logits)) / total_targets * 100))
                else:
                    print("Generation step [{}/{}], untargeted loss: {}.".format(step, self.M, total_loss))
                # _exp_name = f'{self.get_attack_name()}/{self.exp_name}'
                # self.vis.visualize_intermediate_results(r=self.reverse_augment(x=r.squeeze(), datasample=data['data_samples'][0]),
                #                                         r_total = self.reverse_augment(x=pertubed_image.squeeze()-clean_image.squeeze(), datasample=data['data_samples'][0]),
                #                                         pertubed_image=self.reverse_augment(x=pertubed_image.squeeze(), datasample=data['data_samples'][0]),
                #                                         customize_str=step,
                #                                         attack_proposals=torch.cat(accum_proposals, dim=0),
                #                                         image_path=data['data_samples'][0].img_path,
                #                                         exp_name=_exp_name)
            step += 1

        # 这里用了squeeze实际上是只作为一张图片
        pertub_tensor = pertubed_image.squeeze() - clean_image.squeeze()
        adv_tensor = pertubed_image.squeeze()

        pertub = self.reverse_augment(x=pertub_tensor, datasample=data['data_samples'][0])
        adv_image = self.reverse_augment(x=adv_tensor, datasample=data['data_samples'][0])

        if log_info:
            print(f"Generate adv compeleted! Cost iterations {step}.")

        return pertub, adv_image
