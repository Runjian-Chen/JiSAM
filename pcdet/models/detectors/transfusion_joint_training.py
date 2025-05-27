from .detector3d_template_joint_training import Detector3DTemplateJointTraining
import numpy as np
import torch
from collections import defaultdict
import copy
from ..model_utils.joint_training_utils import MinNormSolver, gradient_normalizers
from ..dense_heads import CenterHead, TransFusionHead

class TransFusionJointTraining(Detector3DTemplateJointTraining):
    def __init__(self, model_cfg, num_class, dataset, domains):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, domains = domains)
        self.module_list = self.build_networks()
        if self.domain_alignment_cfg is not None:
            self.init_memory(self.domain_alignment_cfg)

    def forward(self, batch_dict):
        '''

        :param batch_dict:
                    'points': dict {domain: torch.tensor}
                    'voxel_coords': dict {domain: torch.tensor}
                    'voxels': dict {domain: torch.tensor}
                    'voxel_num_points': dict {domain: torch.tensor}

        :return:
        '''
        # domain set in this batch
        self.batch_domains = []
        for domain in self.domains:
            if domain in batch_dict['points'].keys():
                self.batch_domains.append(domain)
        batch_dict['batch_domains'] = self.batch_domains

        if self.conv_input_alignment is not None:
            batch_dict['conv_input_alignment_sources'] = self.conv_input_alignment_sources
            batch_dict['conv_input_alignment_targets'] = self.conv_input_alignment_targets
            batch_dict['conv_input_alignment_sources'] = self.conv_input_alignment_sources

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training and getattr(self, 'MGDA_UB', False):
            if len(self.batch_domains) > 1 or (self.domain_alignment_cfg is not None and self.global_step > self.memory_warmup):
                try:
                    bev_features = batch_dict['spatial_features_2d'].clone().detach()
                    bev_features.requires_grad = True
                    MGDA_UB_input_dict = {
                        'bev_features': bev_features,
                        'domain_list': copy.deepcopy(batch_dict['domain_list']),
                        'batch_size': batch_dict['batch_size'],
                        'total_batch_size': batch_dict['total_batch_size'],
                        'gt_boxes': copy.deepcopy(batch_dict['gt_boxes'])
                    }
                    self.MGDA_UB_generate_scales(MGDA_UB_input_dict)
                except:
                    setattr(self, f'domain_weight_{self.batch_domains[0]}', 1.0)
            else:
                setattr(self, f'domain_weight_{self.batch_domains[0]}', 1.0)

        head_output_dict = {'batch_size': batch_dict['batch_size'], 'total_batch_size': batch_dict['total_batch_size']}
        # Seperate heads for different domains
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

                if self.test_with_both_head:
                    batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == 'SimulationDataset')[0]
                    fake_simulation_batch_dict = {
                        'points': {'SimulationDataset': torch.cat([batch_dict['points'][domain][:, :4],batch_dict['points'][domain][:, 5:6]], dim=-1)},
                        'lidar_aug_matrix': batch_dict['lidar_aug_matrix'],
                        'use_lead_xyz': batch_dict['use_lead_xyz'],
                        'voxels': {'SimulationDataset': torch.cat([batch_dict['voxels'][domain][:, :, :3],batch_dict['voxels'][domain][:, :, 4:5]], dim=-1)},
                        'voxel_coords': {'SimulationDataset': batch_dict['voxel_coords'][domain]},
                        'voxel_num_points': {'SimulationDataset': batch_dict['voxel_num_points'][domain]},
                        'domains': ['SimulationDataset' for n in range(len(batch_dict['domains']))],
                        'domain_set': ['SimulationDataset'],
                        'batch_size': {'SimulationDataset': batch_dict['batch_size'][domain]},
                        'batch_domains': ['SimulationDataset'],
                    }
                    for cur_module in self.module_list:
                        fake_simulation_batch_dict = cur_module(fake_simulation_batch_dict)
                    head_input_dict_simulation = {'spatial_features_2d': fake_simulation_batch_dict['spatial_features_2d'],'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain]}
                    head_output_dict_sim = self.dense_head_SimulationDataset(head_input_dict_simulation)
                    final_box_dicts_sim = head_output_dict_sim['final_box_dicts']
                    for batch_id in range(len(final_box_dicts_sim)):
                        pred_scores = final_box_dicts_sim[batch_id]['pred_scores']
                        score_mask = pred_scores > self.test_with_both_head_score_threshold
                        motorcycle_mask = final_box_dicts_sim[batch_id]['pred_labels'] == 4
                        mask = torch.logical_or(score_mask.bool(), motorcycle_mask.bool())
                        pred_boxes_sim = final_box_dicts_sim[batch_id]['pred_boxes'][mask]
                        pred_scores_sim = pred_scores[mask]
                        pred_labels_sim = final_box_dicts_sim[batch_id]['pred_labels'][mask]
                        # pred_boxes = head_output_dict[domain]['final_box_dicts'][batch_id]['pred_boxes']
                        # pred_scores = head_output_dict[domain]['final_box_dicts'][batch_id]['pred_scores']
                        # pred_labels = head_output_dict[domain]['final_box_dicts'][batch_id]['pred_labels']
                        # ''' try direct merging'''
                        # pred_boxes = torch.cat([pred_boxes, pred_boxes_sim], dim=0)
                        # pred_scores = torch.cat([pred_scores, pred_scores_sim * self.test_with_both_head_sim_scores_weight], dim=0)
                        # for box_id, pred_label_sim in enumerate(pred_labels_sim):
                        #     if self.class_names['SimulationDataset'][pred_label_sim.item() - 1] in self.test_with_both_head_class_mapping.keys():
                        #         pred_labels_sim[box_id] = self.class_names[domain].index(
                        #             self.test_with_both_head_class_mapping[
                        #                 self.class_names['SimulationDataset'][pred_label_sim.item() - 1]]) + 1
                        #
                        # pred_labels = torch.cat([pred_labels, pred_labels_sim], dim=0)
                        # head_output_dict[domain]['final_box_dicts'][batch_id]['pred_boxes'] = pred_boxes
                        # head_output_dict[domain]['final_box_dicts'][batch_id]['pred_scores'] = pred_scores
                        # head_output_dict[domain]['final_box_dicts'][batch_id]['pred_labels'] = pred_labels

                        for box_id, pred_label_sim in enumerate(pred_labels_sim):
                            if self.class_names['SimulationDataset'][pred_label_sim.item() - 1] == 'motorcycle' and 'motorcycle' not in self.class_names['NuScenesDataset']:
                                pred_labels_sim[box_id] = 10
                            elif self.class_names['SimulationDataset'][pred_label_sim.item() - 1] in self.test_with_both_head_class_mapping.keys():
                                pred_labels_sim[box_id] = self.class_names[domain].index(self.test_with_both_head_class_mapping[self.class_names['SimulationDataset'][pred_label_sim.item() - 1]]) + 1

                            min_index = torch.argmin(head_output_dict[domain]['final_box_dicts'][batch_id]['pred_scores'])
                            head_output_dict[domain]['final_box_dicts'][batch_id]['pred_boxes'][min_index] = pred_boxes_sim[box_id]
                            head_output_dict[domain]['final_box_dicts'][batch_id]['pred_scores'][min_index] = pred_scores_sim[box_id] * self.test_with_both_head_sim_scores_weight
                            head_output_dict[domain]['final_box_dicts'][batch_id]['pred_labels'][min_index] = pred_labels_sim[box_id]

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(head_output_dict, batch_dict)

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict, head_output_dict)
            return pred_dicts, recall_dicts

    def MGDA_UB_generate_scales(self, input_dict):
        grads = {}
        loss_data = {}
        for domain in self.batch_domains:
            batch_idx_in_domain = np.where(np.array(input_dict['domain_list']) == domain)[0]
            head_input_dict = {
                'spatial_features_2d': input_dict['bev_features'][batch_idx_in_domain],
                'gt_boxes': input_dict['gt_boxes'][batch_idx_in_domain]
            }
            if isinstance(getattr(self, 'dense_head_' + domain), TransFusionHead):
                loss_domain = getattr(self, 'dense_head_' + domain)(head_input_dict)['loss']
            else:
                head_output_dict = getattr(self, 'dense_head_' + domain)(head_input_dict)
                loss_domain, _ = getattr(self, 'dense_head_' + domain).get_loss()
            loss_domain.backward()
            loss_data[domain] = loss_domain.item()
            grads[domain] = input_dict['bev_features'].grad.clone().detach()
            input_dict['bev_features'].grad.data.zero_()

        if self.domain_alignment_cfg is not None:
            domain_alignment_input_dict = {
                'spatial_features_2d': input_dict['bev_features'],
                'gt_boxes': input_dict['gt_boxes'],
                'domain_list': input_dict['domain_list']
            }

            feature_dict, coors_dict, gt_boxes_dict = self.prepare_domain_alignment_features(domain_alignment_input_dict)
            domain_alignment_loss = self.domain_alignment_loss(feature_dict, coors_dict, gt_boxes_dict)
            if domain_alignment_loss is not None:
                domain_alignment_loss.backward()
                loss_data['domain_alignment'] = domain_alignment_loss.item()
                grads['domain_alignment'] = input_dict['bev_features'].grad.clone().detach()
                input_dict['bev_features'].grad.data.zero_()

        # Normalize all the gradients
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for task in grads.keys():
            grads[task] = grads[task] / gn[task]

        # Frank-Wolfe iteration to compute scales
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[task] for task in grads.keys()])
        for idx, task in enumerate(grads.keys()):
            if task == 'domain_alignment':
                self.domain_alignment_loss_weight = len(grads.keys()) * float(sol[idx])
            else:
                setattr(self, f'domain_weight_{task}', len(grads.keys()) * float(sol[idx]))

    def get_training_loss(self,head_output_dict, batch_dict):
        disp_dict = {}
        tb_dict = {}
        loss_trans = []

        for domain in self.batch_domains:
            if isinstance(getattr(self, 'dense_head_' + domain), TransFusionHead):
                loss_trans_, tb_dict_ = head_output_dict[domain]['loss'], head_output_dict[domain]['tb_dict']
                tb_dict.update({
                    'loss_trans_' + domain: loss_trans_.item(),
                })
                tb_dict.update({
                    k + '_' + domain: v for k, v in tb_dict_.items()
                })

                loss_trans.append( getattr(self, f'domain_weight_{domain}') * loss_trans_.unsqueeze(0) * head_output_dict['batch_size'][domain])

            elif isinstance(getattr(self, 'dense_head_' + domain), CenterHead):
                loss_center_, tb_dict_ = getattr(self, 'dense_head_' + domain).get_loss()
                tb_dict.update({
                    'loss_center_' + domain: loss_center_.item(),
                })
                tb_dict.update({
                    k + '_' + domain: v for k, v in tb_dict_.items()
                })

                loss_trans.append(getattr(self, f'domain_weight_{domain}') * loss_center_.unsqueeze(0) *
                                  head_output_dict['batch_size'][domain])
            else:
                raise NotImplementedError



        loss = torch.sum(torch.cat(loss_trans)) / head_output_dict['total_batch_size']

        if self.domain_alignment_cfg is not None:
            feature_dict, coors_dict, gt_boxes_dict = self.prepare_domain_alignment_features(batch_dict)
            domain_alignment_loss = self.domain_alignment_loss(feature_dict, coors_dict, gt_boxes_dict)
            if domain_alignment_loss is not None:
                tb_dict.update(
                    {'loss_domain_alignment': domain_alignment_loss.item()}
                )
                loss += self.domain_alignment_loss_weight * domain_alignment_loss

        if self.conv_input_alignment is not None and self.global_step > self.conv_input_alignment_warmup:
            conv_input_alignment_loss = 0
            for source_domain, target_domain in zip(self.conv_input_alignment_sources, self.conv_input_alignment_targets):
                if  source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor' in batch_dict.keys():
                    if self.conv_input_alignment_target_grad_only:
                        conv_input_alignment_loss += self.conv_input_alignment_loss(batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor'].detach(), batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_target_tensor'])
                    else:
                        conv_input_alignment_loss += self.conv_input_alignment_loss(batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor'], batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_target_tensor'])
            tb_dict.update(
                {'loss_conv_input_alignment': conv_input_alignment_loss.item() if conv_input_alignment_loss > 0 else 0}
            )

            loss += self.conv_input_alignment_loss_weight * conv_input_alignment_loss

        if self.head_augmentation and self.global_step > self.head_augmentation_warmup and 'head_augmentation' in head_output_dict.keys():
            loss_trans_head_augmentation, tb_dict_head_augmentation = head_output_dict['head_augmentation']['loss'], head_output_dict['head_augmentation']['tb_dict']
            tb_dict.update({
                'loss_trans_head_augmentation': loss_trans_head_augmentation.item(),
            })
            tb_dict.update({
                k + '_head_augmentation': v for k, v in tb_dict_head_augmentation.items()
            })

            loss += loss_trans_head_augmentation * self.head_augmentation_weight

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

    def post_processing(self, batch_dict, head_output_dict):
        assert len(self.test_domains) == 1, 'Currently only one test dataset is supported'

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = head_output_dict['total_batch_size']
        final_pred_dict = head_output_dict[self.test_domains[0]]['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
