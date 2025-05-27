from .detector3d_template_joint_training import Detector3DTemplateJointTraining
import numpy as np
import torch
from collections import defaultdict

class CenterPointJointTraining(Detector3DTemplateJointTraining):
    def __init__(self, model_cfg, num_class, dataset, domains):
        super(CenterPointJointTraining, self).__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, domains=domains)
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

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        head_output_dict = {}
        # Seperate heads for different domains
        if self.training:
            for domain in self.batch_domains:
                batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == domain)[0]

                head_input_dict = {
                    'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain],
                    'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain],
                    'batch_size': batch_dict['batch_size'][domain],
                }
                head_output_dict[domain] = getattr(self, 'dense_head_'+domain)(head_input_dict)

        else:
            for domain in self.test_domains:
                batch_idx_in_domain = np.where(np.array(batch_dict['domain_list']) == domain)[0]
                head_input_dict = {
                    'spatial_features_2d': batch_dict['spatial_features_2d'][batch_idx_in_domain],
                    'gt_boxes': batch_dict['gt_boxes'][batch_idx_in_domain],
                    'batch_size': batch_dict['batch_size'][domain],
                }
                head_output_dict[domain] = getattr(self, 'dense_head_'+domain)(head_input_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict['batch_size'], batch_dict['total_batch_size'], batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict, head_output_dict)
            return pred_dicts, recall_dicts


    def get_training_loss(self, batch_size, total_batch_size, batch_dict=None):
        disp_dict = {}
        tb_dict = {}
        loss_rpn = []

        for domain in self.batch_domains:
            loss_rpn_, tb_dict_ = getattr(self, 'dense_head_' + domain).get_loss()
            tb_dict.update(
                {'loss_rpn_' + domain: loss_rpn_.item()}
            )
            tb_dict.update({
                k + '_' + domain: v for k, v in tb_dict_.items()
            })
            loss_rpn.append(getattr(self, f'domain_weight_{domain}') * loss_rpn_.unsqueeze(0) *
                                  batch_size[domain])

        loss = torch.sum(torch.cat(loss_rpn)) / total_batch_size

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
                if source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor' in batch_dict.keys():
                    conv_input_alignment_loss += self.conv_input_alignment_loss(batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_source_tensor'], batch_dict[source_domain + '_conv_input_align_to_' + target_domain + '_target_tensor'])
            tb_dict.update(
                {'loss_conv_input_alignment': conv_input_alignment_loss.item() if conv_input_alignment_loss > 0 else 0}
            )

            loss += self.conv_input_alignment_loss_weight * conv_input_alignment_loss

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict, head_output_dict):
        assert len(self.test_domains) == 1, 'Currently only one test dataset is supported'

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['total_batch_size']
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
