from .detector3d_template_joint_training import Detector3DTemplateJointTraining
import numpy as np
import torch
from ...utils.spconv_utils import spconv
from collections import defaultdict

class PVRCNNPlusPlusJointTraining(Detector3DTemplateJointTraining):
    def __init__(self, model_cfg, num_class, dataset, domains):
        super(PVRCNNPlusPlusJointTraining, self).__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, domains=domains)
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

        if self.conv_input_alignment is not None:
            batch_dict['conv_input_alignment_sources'] = self.conv_input_alignment_sources
            batch_dict['conv_input_alignment_targets'] = self.conv_input_alignment_targets

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        head_output_dict = {}

        if self.training:
            acc_batch_cnt = 0
            for domain in self.batch_domains:
                batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == domain)[0]
                multi_scale_3d_features_in_domain = {}
                multi_scale_3d_features = batch_dict['multi_scale_3d_features']
                for layer, sp_tensor in multi_scale_3d_features.items():
                    features = sp_tensor._features
                    indices = sp_tensor.indices
                    features_in_domain = []
                    indices_in_domain = []
                    for batch_idx in batch_idx_in_domain:
                        features_in_domain.append(features[indices[:,0] == batch_idx])
                        indices_in_domain.append(indices[indices[:,0] == batch_idx])
                    features_in_domain = torch.cat(features_in_domain, dim=0)
                    indices_in_domain = torch.cat(indices_in_domain, dim=0)
                    indices_in_domain[:, 0] -= acc_batch_cnt
                    sparse_shape = sp_tensor.spatial_shape

                    multi_scale_3d_features_in_domain[layer] = spconv.SparseConvTensor(
                        features=features_in_domain.squeeze(0),
                        indices=indices_in_domain.squeeze(0),
                        spatial_shape=sparse_shape,
                        batch_size=len(batch_idx_in_domain)
                    )

                head_input_dict = {
                    'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain],
                    'spatial_features': batch_dict['spatial_features'][batch_idx_in_domain],
                    'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain],
                    'batch_size': batch_dict['batch_size'][domain],
                    'points': batch_dict['points'][domain],
                    'spatial_features_stride': batch_dict['spatial_features_stride'],
                    'multi_scale_3d_strides': batch_dict['multi_scale_3d_strides'],
                    'multi_scale_3d_features': multi_scale_3d_features_in_domain
                }
                head_dict = getattr(self, 'dense_head_' + domain)(head_input_dict)
                if self.training:
                    targets_dict = getattr(self, 'roi_head_' + domain).assign_targets(head_dict)
                    head_dict['rois'] = targets_dict['rois']
                    # if torch.sum(head_dict['rois']) == 0:
                    #     print(domain)
                    #     print(getattr(self, 'dense_head_' + domain).forward_ret_dict['pred_dicts'])
                    head_dict['roi_labels'] = targets_dict['roi_labels']
                    head_dict['roi_targets_dict'] = targets_dict
                    num_rois_per_scene = targets_dict['rois'].shape[1]
                    if 'roi_valid_num' in head_dict:
                        head_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(head_dict['batch_size'])]

                head_dict['domain'] = domain
                if self.pfe_raw_point_alignment is not None and domain in self.pfe_raw_point_alignment_sources:
                    head_dict['pfe_raw_point_alignment'] = True
                    head_dict['pfe_raw_point_alignment_target_domain'] = self.pfe_raw_point_alignment_targets[self.pfe_raw_point_alignment_sources.index(domain)]
                else:
                    head_dict['pfe_raw_point_alignment'] = False

                head_dict = getattr(self, 'pfe')(head_dict)
                head_dict = getattr(self, 'point_head_' + domain)(head_dict)
                head_dict = getattr(self, 'roi_head_' + domain)(head_dict)

                head_output_dict[domain] = head_dict
                acc_batch_cnt += len(batch_idx_in_domain)

        else:
            test_domain = self.test_domains[0]
            batch_dict['batch_size'] = batch_dict['batch_size'][test_domain]
            batch_dict['points'] = batch_dict['points'][test_domain]
            batch_dict = getattr(self, 'dense_head_' + test_domain)(batch_dict)
            if self.training:
                targets_dict = getattr(self, 'roi_head_' + test_domain).assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_targets_dict'] = targets_dict
                num_rois_per_scene = targets_dict['rois'].shape[1]
                if 'roi_valid_num' in batch_dict:
                    batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

            batch_dict['domain'] = test_domain
            batch_dict['pfe_raw_point_alignment'] = False

            batch_dict = getattr(self, 'pfe' )(batch_dict)
            batch_dict = getattr(self, 'point_head_' + test_domain)(batch_dict)
            batch_dict = getattr(self, 'roi_head_' + test_domain)(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict['batch_size'], batch_dict['total_batch_size'], batch_dict, head_output_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_size, total_batch_size, batch_dict = None, head_output_dict=None):
        disp_dict = {}
        tb_dict = {}
        loss_rpn = []
        loss_point = []
        loss_rcnn = []

        for domain in self.batch_domains:
            loss_rpn_, tb_dict_ = getattr(self, 'dense_head_'+domain).get_loss()
            if getattr(self, 'point_head_'+domain, None) is not None:
                loss_point_, tb_dict_ = getattr(self, 'point_head_'+domain).get_loss(tb_dict_)
            else:
                loss_point_ = 0
            loss_rcnn_, tb_dict_ = getattr(self, 'roi_head_'+domain).get_loss(tb_dict_)

            loss_rpn.append(loss_rpn_.unsqueeze(0) * getattr(self, f'domain_weight_{domain}') * batch_size[domain])
            loss_point.append(loss_point_.unsqueeze(0) * getattr(self, f'domain_weight_{domain}') * batch_size[domain])
            loss_rcnn.append(loss_rcnn_.unsqueeze(0) * getattr(self, f'domain_weight_{domain}') * batch_size[domain])

            tb_dict.update({
                k + '_' + domain: v for k, v in tb_dict_.items()
            })

        loss_rpn = torch.sum(torch.cat(loss_rpn)) / total_batch_size
        loss_point = torch.sum(torch.cat(loss_point)) / total_batch_size
        loss_rcnn = torch.sum(torch.cat(loss_rcnn)) / total_batch_size

        loss = loss_rpn + loss_point + loss_rcnn

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
                    conv_input_alignment_loss += self.conv_input_alignment_loss(batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor'], batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_target_tensor'])
            tb_dict.update(
                {'loss_conv_input_alignment': conv_input_alignment_loss.item() if conv_input_alignment_loss > 0 else 0}
            )

            loss += self.conv_input_alignment_loss_weight * conv_input_alignment_loss

        for domain in self.batch_domains:
            domain_dict = head_output_dict[domain]
            if self.pfe_raw_point_alignment is not None and self.global_step > self.pfe_raw_point_alignment_warmup:
                pfe_raw_point_alignment_loss = 0
                for source_domain, target_domain in zip(self.pfe_raw_point_alignment_sources,
                                                        self.pfe_raw_point_alignment_targets):
                    if source_domain + '_pfe_raw_point_align_to_' + target_domain + '_source_tensor' in domain_dict.keys():
                        pfe_raw_point_alignment_loss += self.pfe_raw_point_alignment_loss(
                            domain_dict[source_domain + '_pfe_raw_point_align_to_' + target_domain + '_source_tensor'],
                            domain_dict[source_domain + '_pfe_raw_point_align_to_' + target_domain + '_target_tensor'])
                tb_dict.update(
                    {'loss_pfe_raw_point_alignment': pfe_raw_point_alignment_loss.item() if pfe_raw_point_alignment_loss > 0 else 0}
                )

                loss += self.pfe_raw_point_alignment_loss_weight * pfe_raw_point_alignment_loss

        return loss, tb_dict, disp_dict

    def prepare_domain_alignment_features(self, batch_dict):
        feature_dict = defaultdict(dict)
        coors_dict = defaultdict(dict)
        gt_boxes_dict = {}
        for feature_source in self.domain_alignment_feature_source:
            for domain in self.batch_domains:
                if feature_source == 'bev':
                    index_in_domain = [idx for idx, domain_ in enumerate(batch_dict['domain_list']) if
                                       domain_ == domain]
                    bev_features_in_domain = batch_dict['spatial_features_2d'][index_in_domain]  # [B, C, H, W]
                    bev_grid_position = self.bev_grid_position.clone().repeat(bev_features_in_domain.shape[0], 1,
                                                                              1)  # [B, H * W, 2]
                    bev_coors = []
                    for batch_id in range(bev_grid_position.shape[0]):
                        bev_coors.append(torch.cat(
                            [bev_grid_position.new_ones(bev_grid_position.shape[1], 1) * batch_id,
                             bev_grid_position[batch_id], bev_grid_position.new_zeros(bev_grid_position.shape[1], 1)],
                            dim=-1))
                    bev_coors = torch.cat(bev_coors, dim=0)

                    feature_dict[feature_source][domain] = bev_features_in_domain.permute(0, 2, 3, 1)
                    feature_dict[feature_source][domain] = feature_dict[feature_source][domain].reshape(-1,
                                                                                                        feature_dict[feature_source][domain].shape[
                                                                                                            -1])
                    coors_dict[feature_source][domain] = bev_coors
                    gt_boxes_dict[domain] = batch_dict['gt_boxes'][index_in_domain]

                    return feature_dict, coors_dict, gt_boxes_dict
                else:
                    raise NotImplementedError
