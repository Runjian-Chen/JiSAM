from .detector3d_template_joint_training import Detector3DTemplateJointTraining
import numpy as np
import torch
from collections import defaultdict
import copy
from ..model_utils.joint_training_utils import MinNormSolver, gradient_normalizers
from ..dense_heads import CenterHead, TransFusionHead

class SECONDNetJointTraining(Detector3DTemplateJointTraining):
    def __init__(self, model_cfg, num_class, dataset, domains):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, domains = domains)
        self.module_list = self.build_networks()
        if self.domain_alignment_cfg is not None:
            self.init_memory(self.domain_alignment_cfg)

    def forward(self, batch_dict):
        # domain set in this batch
        self.batch_domains = []
        for domain in self.domains:
            if domain in batch_dict['points'].keys():
                self.batch_domains.append(domain)
        batch_dict['batch_domains'] = self.batch_domains

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        head_output_dict = {'batch_size': batch_dict['batch_size'], 'total_batch_size': batch_dict['total_batch_size']}
        if self.training:
            for domain in self.batch_domains:
                batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == domain)[0]

                head_input_dict = {
                    'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain],
                    'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain]
                }
                head_output_dict[domain] = getattr(self, 'dense_head_'+domain)(head_input_dict)

                # Todo: more flexible config for manipulating data domain
                # Todo: none class for outlier classes
                if self.head_augmentation and self.global_step > self.head_augmentation_warmup and domain == 'SimulationDataset':
                    # transfer original annotation into annotation system of corresponding domain
                    original_gt_boxes = batch_dict['gt_boxes'][batch_idx_in_domain]
                    original_classes = original_gt_boxes[:, :, -1]
                    updated_gt_boxes = original_gt_boxes.clone()
                    bs, max_box = original_classes.shape
                    original_classes = original_classes.view(-1)
                    mask = original_classes > 0
                    simulation_class_names = self.class_names['SimulationDataset']
                    # infer detection results and get loss from the head
                    updated_classes = torch.zeros_like(original_classes)
                    simulation_box_class_names = [simulation_class_names[idx] for idx in (original_classes[mask] - 1).long()]
                    updated_simulation_box_class_names = [self.head_augmentation_class_mapping[box_class] for box_class in simulation_box_class_names]
                    updated_simulation_box_class_idx = [self.class_names['NuScenesDataset'].index(updated_class_name)+1 for updated_class_name in updated_simulation_box_class_names]
                    updated_classes[mask] = torch.tensor(updated_simulation_box_class_idx).to(updated_classes.device).float()
                    updated_classes = updated_classes.resize(bs, max_box)
                    updated_gt_boxes[:, :, -1] = updated_classes

                    head_augmentation_input_dict = {
                        'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain] if not self.head_augmentation_detach_simulation_bev_feature else batch_dict['spatial_features_2d'][batch_idx_in_domain].detach(),
                        'gt_boxes': updated_gt_boxes
                    }
                    head_output_dict['head_augmentation'] = self.dense_head_NuScenesDataset(head_augmentation_input_dict)

        else:
            for domain in self.test_domains:
                batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == domain)[0]
                head_input_dict = {
                    'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain],
                    'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain]
                }
                head_output_dict[domain] = getattr(self, 'dense_head_'+domain)(head_input_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(head_output_dict, batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, head_output_dict, batch_dict):
        disp_dict = {}
        tb_dict = {}
        loss_list = []
        for domain in self.batch_domains:
            loss_second_, tb_dict_ = getattr(self, 'dense_head_' + domain).get_loss()
            tb_dict.update({
                'loss_second_' + domain: loss_second_.item(),
            })
            tb_dict.update({
                k + '_' + domain: v for k, v in tb_dict_.items()
            })

            loss_list.append(getattr(self, f'domain_weight_{domain}') * loss_second_.unsqueeze(0) *
                              head_output_dict['batch_size'][domain])

        loss = torch.sum(torch.cat(loss_list)) / head_output_dict['total_batch_size']

        if self.domain_alignment_cfg is not None:
            feature_dict, coors_dict, gt_boxes_dict = self.prepare_domain_alignment_features(batch_dict)
            domain_alignment_loss = self.domain_alignment_loss(feature_dict, coors_dict, gt_boxes_dict)
            if domain_alignment_loss is not None:
                tb_dict.update(
                    {'loss_domain_alignment': domain_alignment_loss.item()}
                )
                loss += self.domain_alignment_loss_weight * domain_alignment_loss

        return loss, tb_dict, disp_dict

    def prepare_domain_alignment_features(self, batch_dict):
        feature_dict = defaultdict(dict)
        coors_dict = defaultdict(dict)
        gt_boxes_dict = {}
        for feature_source in self.domain_alignment_feature_source:
            for domain in self.batch_domains:
                if feature_source == 'bev':
                    index_in_domain = [idx for idx, domain_ in enumerate(batch_dict['domain_list']) if domain_ == domain]
                    bev_features_in_domain = batch_dict['spatial_features_2d'][index_in_domain] # [B, C, H, W]
                    bev_grid_position = self.bev_grid_position.clone().repeat(bev_features_in_domain.shape[0], 1, 1) # [B, H * W, 2]
                    bev_coors = []
                    for batch_id in range(bev_grid_position.shape[0]):
                        bev_coors.append(torch.cat([bev_grid_position.new_ones(bev_grid_position.shape[1], 1) * batch_id, bev_grid_position[batch_id], bev_grid_position.new_zeros(bev_grid_position.shape[1], 1)], dim=-1))
                    bev_coors = torch.cat(bev_coors, dim=0)

                    # change from 0, 2, 3, 1 to 0, 3, 2, 1
                    feature_dict[feature_source][domain] = bev_features_in_domain.permute(0, 3, 2, 1)
                    feature_dict[feature_source][domain] = feature_dict[feature_source][domain].reshape(-1, feature_dict[feature_source][domain].shape[-1])
                    coors_dict[feature_source][domain] = bev_coors
                    gt_boxes_dict[domain] = batch_dict['gt_boxes'][index_in_domain]

                    return feature_dict, coors_dict, gt_boxes_dict
                else:
                    raise NotImplementedError