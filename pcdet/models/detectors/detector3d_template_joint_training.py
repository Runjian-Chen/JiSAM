import torch
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import vfe, pfe
import os
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from ..model_utils import model_nms_utils
from pcdet.models.model_utils.joint_training_utils import ShapeContext
import math
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.utils import common_utils

class Detector3DTemplateJointTraining(torch.nn.Module):
    def __init__(self, model_cfg, num_class, dataset, domains):
        '''

        :param model_cfg:
        :param num_class: dict {'domain': 'num_class'}
        :param dataset:
        :param domains:
        '''
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = {}
        self.class_names = {}
        for domain in domains:
            self.num_class[domain] = num_class[domain]
            self.class_names[domain] = getattr(dataset, domain).class_names
        self.dataset = dataset
        self.domains = domains
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'backbone_2d',
            'dense_head',  'point_head', 'roi_head'
        ]

        self.test_domains = model_cfg.TEST_DOMAINS
        self.test_with_both_head = model_cfg.get('TEST_WITH_BOTH_HEAD', False)
        self.test_with_both_head_sim_scores_weight = model_cfg.get('TEST_WITH_BOTH_HEAD_SIM_SCORES_WEIGHT', 0.1)
        self.test_with_both_head_score_threshold = model_cfg.get('TEST_WITH_BOTH_HEAD_SCORE_THRESHOLD', 0.5)
        self.test_with_both_head_class_mapping = model_cfg.get('TEST_WITH_BOTH_HEAD_CLASS_MAPPING', {'motorcycle': 'motorcycle', 'cyclist': 'bicycle'
              })
        assert len(self.test_domains) == 1, 'Only one test domain is supported now'

        self.domain_alignment_cfg = model_cfg.get('DOMAIN_ALIGNMENT', None)

        weight_cfg = model_cfg.get('DOMAIN_WEIGHT', None)
        for domain in self.domains:
            if weight_cfg is None:
                setattr(self, f'domain_weight_{domain}', 1.0)
            else:
                weight = weight_cfg.get(domain, 1.0)
                if weight > 0:
                    setattr(self, f'domain_weight_{domain}', weight)
                else:
                    self.MGDA_UB = True

        self.conv_input_alignment = model_cfg.get('CONV_INPUT_ALIGNMENT', None)
        if self.conv_input_alignment is not None:
            self.conv_input_alignment_sources = self.conv_input_alignment.get('SOURCES')
            self.conv_input_alignment_targets = self.conv_input_alignment.get('TARGETS')
            self.conv_input_alignment_loss_weight = self.conv_input_alignment.get('WEIGHT', 0.1)
            loss_type = self.conv_input_alignment.get('LOSS', 'L2')
            if loss_type == 'L2':
                self.conv_input_alignment_loss = torch.nn.MSELoss()
            elif loss_type == 'L1':
                self.conv_input_alignment_loss = torch.nn.L1Loss()
            else:
                raise NotImplementedError

            self.conv_input_alignment_warmup = self.conv_input_alignment.get('WARMUP', 0)
            self.conv_input_alignment_target_grad_only = self.conv_input_alignment.get('TARGET_GRAD_ONLY', False)

        self.pfe_raw_point_alignment = model_cfg.get('PFE_RAW_POINT_ALIGNMENT', None)
        if self.pfe_raw_point_alignment is not None:
            self.pfe_raw_point_alignment_sources = self.pfe_raw_point_alignment.get('SOURCES')
            self.pfe_raw_point_alignment_targets = self.pfe_raw_point_alignment.get('TARGETS')
            self.pfe_raw_point_alignment_loss_weight = self.pfe_raw_point_alignment.get('WEIGHT', 0.1)
            loss_type = self.pfe_raw_point_alignment.get('LOSS', 'L2')
            if loss_type == 'L2':
                self.pfe_raw_point_alignment_loss = torch.nn.MSELoss()
            elif loss_type == 'L1':
                self.pfe_raw_point_alignment_loss = torch.nn.L1Loss()
            else:
                raise NotImplementedError

            self.pfe_raw_point_alignment_warmup = self.conv_input_alignment.get('WARMUP', 0)
            self.pfe_raw_point_alignment_target_grad_only = self.conv_input_alignment.get('TARGET_GRAD_ONLY', False)

        self.head_augmentation = model_cfg.get('HEAD_AUGMENTATION', False)
        self.head_augmentation_class_mapping = model_cfg.get('HEAD_AUGMENTATION_CLASS_MAPPING', None)
        self.head_augmentation_warmup = model_cfg.get('HEAD_AUGMENTATION_WARMUP', 4000)
        self.head_augmentation_weight = model_cfg.get('HEAD_AUGMENTATION_WEIGHT', 1.0)
        self.head_augmentation_detach_simulation_bev_feature = model_cfg.get('HEAD_AUGMENTATION_DETACH_SIMULATION_BEV_FEATURE', False)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': {domain: getattr(self.dataset, domain).point_feature_encoder.num_point_features for domain in self.domains},
            'num_point_features': {domain: getattr(self.dataset, domain).point_feature_encoder.num_point_features for domain in self.domains},
            'grid_size': {domain: getattr(self.dataset, domain).grid_size for domain in self.domains},
            'point_cloud_range': {domain: getattr(self.dataset, domain).point_cloud_range for domain in self.domains},
            'voxel_size': {domain: getattr(self.dataset, domain).voxel_size for domain in self.domains},
            'depth_downsample_factor': {domain: getattr(self.dataset, domain).depth_downsample_factor for domain in self.domains},
            'domains': self.domains
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            if module is not None and module_name in ['dense_head', 'point_head', 'roi_head']:
                for domain in self.domains:
                    self.add_module(module_name + '_' + domain, module[domain])
            else:
                self.add_module(module_name, module)

        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor'],
            domains=model_info_dict['domains']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            domains=model_info_dict['domains'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size'],
            domains=model_info_dict['domains'],
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None),
            domains=model_info_dict['domains'],
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features'],
            domains=model_info_dict['domains']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict

        dense_head_dict = {}

        for domain in self.model_cfg.DENSE_HEAD.keys():
            specific_domain_cfg = getattr(self.model_cfg.DENSE_HEAD, domain)
            dense_head_module = dense_heads.__all__[specific_domain_cfg.NAME](
                model_cfg = specific_domain_cfg,
                input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else specific_domain_cfg.INPUT_FEATURES,
                num_class = self.num_class[domain] if not specific_domain_cfg.CLASS_AGNOSTIC else 1,
                class_names = self.class_names[domain],
                grid_size = model_info_dict['grid_size'][domain],
                point_cloud_range = model_info_dict['point_cloud_range'][domain],
                predict_boxes_when_training = self.model_cfg.get('ROI_HEAD', False),
                voxel_size = model_info_dict['voxel_size'][domain] if 'voxel_size' in model_info_dict else False
            )
            dense_head_dict[domain] = dense_head_module

        return dense_head_dict, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        point_head_dict = {}

        for domain in self.model_cfg.POINT_HEAD.keys():
            specific_domain_cfg = getattr(self.model_cfg.POINT_HEAD, domain)

            if specific_domain_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
                num_point_features = model_info_dict['num_point_features_before_fusion']
            else:
                num_point_features = model_info_dict['num_point_features']

            point_head_module = dense_heads.__all__[specific_domain_cfg.NAME](
                model_cfg=specific_domain_cfg,
                input_channels=num_point_features,
                num_class=self.num_class[domain] if not specific_domain_cfg.CLASS_AGNOSTIC else 1,
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            )

            point_head_dict[domain] = point_head_module

        return point_head_dict, model_info_dict


    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict

        roi_head_dict = {}

        for domain in self.model_cfg.ROI_HEAD.keys():
            specific_domain_cfg = getattr(self.model_cfg.ROI_HEAD, domain)
            roi_head_module = roi_heads.__all__[specific_domain_cfg.NAME](
                model_cfg=specific_domain_cfg,
                input_channels=model_info_dict['num_point_features'],
                backbone_channels=model_info_dict.get('backbone_channels', None),
                point_cloud_range=model_info_dict['point_cloud_range'][domain],
                voxel_size=model_info_dict['voxel_size'][domain],
                num_class=self.num_class[domain] if not specific_domain_cfg.CLASS_AGNOSTIC else 1,
            )
            roi_head_dict[domain] = roi_head_module

        return roi_head_dict, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def init_memory(self, domain_alignment_cfg):
        # Initialize the scene partitioner
        partitioner_cfg = domain_alignment_cfg.PARTITIONER
        self.scene_partitioner = ShapeContext(r1 = partitioner_cfg.R1, r2 = partitioner_cfg.R2, nbins_xy = partitioner_cfg.NXY, nbins_zy = partitioner_cfg.NZY)
        self.num_scene_partition = self.scene_partitioner.partitions

        # Initialize memory for each class in each partition separately for each feature source
        for feature_source, feature_channels in zip(domain_alignment_cfg.FEATURE_SOURCE, domain_alignment_cfg.FEATURE_CHANNELS):
            if domain_alignment_cfg.get('MEMORY_INIT_BY_ZERO', False):
                self.register_buffer(f'target_domain_{feature_source}_feature_memory', torch.zeros(len(self.class_names[self.test_domains[0]]), domain_alignment_cfg.NUM_DIR_BINS, self.num_scene_partition, feature_channels))
            else:
                self.register_buffer(f'target_domain_{feature_source}_feature_memory', torch.randn(len(self.class_names[self.test_domains[0]]), domain_alignment_cfg.NUM_DIR_BINS, self.num_scene_partition, feature_channels))
            self.add_module(f'local_feature_embedding_{feature_source}', torch.nn.Sequential(torch.nn.Linear(6, feature_channels // 2), torch.nn.ReLU(), torch.nn.Linear(feature_channels // 2, feature_channels)))
            grid_pooling_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=feature_channels, config=domain_alignment_cfg.GRID_POOL)
            self.add_module(f'grid_pooling_layer_{feature_source}', grid_pooling_layer)

            GRID_SIZE = domain_alignment_cfg.GRID_POOL.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out
            self.separate_fc_layers = False
            if domain_alignment_cfg.GRID_POOL.get('DEEPER_FC_LAYERS', False):
                if domain_alignment_cfg.GRID_POOL.get('SEPARATE_FC_LAYERS', False):
                    self.separate_fc_layers = True
                    for domain in self.domains:
                        self.add_module(f'fc_layer_{feature_source}_{domain}',
                                        torch.nn.Sequential(torch.nn.Linear(pre_channel, feature_channels, bias=False),
                                                            torch.nn.BatchNorm1d(feature_channels), torch.nn.ReLU(),
                                                            torch.nn.Linear(feature_channels, feature_channels,
                                                                            bias=False),
                                                            torch.nn.BatchNorm1d(feature_channels), torch.nn.ReLU()))
                else:
                    self.add_module(f'fc_layer_{feature_source}', torch.nn.Sequential(torch.nn.Linear(pre_channel, feature_channels, bias=False), torch.nn.BatchNorm1d(feature_channels), torch.nn.ReLU(), torch.nn.Linear(feature_channels, feature_channels, bias=False), torch.nn.BatchNorm1d(feature_channels), torch.nn.ReLU()))
            else:
                self.add_module(f'fc_layer_{feature_source}', torch.nn.Sequential(torch.nn.Linear(pre_channel, feature_channels, bias=False), torch.nn.ReLU()))


        # Create mapping to memory
        self.supported_domains = [domain for domain in self.domains if domain not in self.test_domains]
        for domain in self.supported_domains:
            self.register_buffer(f'mapping_class_to_memory_{domain}', -1 * torch.ones(len(self.class_names[domain]), dtype=torch.long))
            for class_id, class_name in enumerate(self.class_names[domain]):
                mapped_class_name = domain_alignment_cfg.CLASS_MAPPING[domain].get(class_name, class_name)
                if not mapped_class_name in self.class_names[self.test_domains[0]]:
                    getattr(self, f'mapping_class_to_memory_{domain}')[class_id] = -100
                    continue
                getattr(self, f'mapping_class_to_memory_{domain}')[class_id] = self.class_names[self.test_domains[0]].index(mapped_class_name)

        self.domain_alignment_feature_source = domain_alignment_cfg.FEATURE_SOURCE
        self.num_dir_bins = domain_alignment_cfg.NUM_DIR_BINS
        self.memory_momentum = domain_alignment_cfg.MEMORY_MOMENTUM
        self.memory_warmup = domain_alignment_cfg.get('MEMORY_WARMUP', 0)
        self.object_feature_pooling_grid_size = domain_alignment_cfg.GRID_POOL.GRID_SIZE
        self.domain_alignment_loss_weight = domain_alignment_cfg.WEIGHT
        loss_func = domain_alignment_cfg.get('LOSS_FUNC', 'L1')
        if loss_func == 'L1':
            self.domain_alignment_loss_function = torch.nn.L1Loss(reduction=domain_alignment_cfg.get('LOSS_REDUCTION', 'mean'))
        elif loss_func == 'L2':
            self.domain_alignment_loss_function = torch.nn.MSELoss(reduction=domain_alignment_cfg.get('LOSS_REDUCTION', 'mean'))
        elif loss_func == 'CONTRASTIVE':
            self.domain_alignment_loss_function = torch.nn.CrossEntropyLoss()
            self.T = domain_alignment_cfg.get('T')
        else:
            raise NotImplementedError
        self.loss_func = loss_func

        self.grid_size = self.vfe.grid_size[self.test_domains[0]]
        self.voxel_size = self.vfe.voxel_size[self.test_domains[0]]
        self.feature_map_stride = domain_alignment_cfg.FEATURE_MAP_STRIDE
        self.point_cloud_range = domain_alignment_cfg.POINT_CLOUD_RANGE
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_grid = self.create_2D_grid(x_size, y_size) # [1, x_size * y_size, 2]
        self.register_buffer('bev_grid_position', self.bev_grid * torch.tensor(self.voxel_size[:2]).unsqueeze(0).unsqueeze(0) * self.feature_map_stride + torch.tensor(self.point_cloud_range[:2]).unsqueeze(0).unsqueeze(0)) # [1, x_size * y_size, 2]

        self.minus_local_information = domain_alignment_cfg.get('MINUS_LOCAL_INFORMATION', False)
        self.alignment_window_size = domain_alignment_cfg.get('ALIGNMENT_WINDOW_SIZE', 1)
        self.domain_alignment_memory_momentum_decay = domain_alignment_cfg.get('MEMORY_MOMENTUM_DECAY', False)
        self.bidirection_domain_alignment = domain_alignment_cfg.get('BIDIRECTION_MEMORY_ALIGNMENT', False)

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def update_memory(self, feature_source, features, class_idx, yaw_bin_idx, scene_partition_idx):
        '''
            feature_source: one of ['bev', 'conv_out', ...]
            features: [N, C]
            class_idx: [N]
            yaw_bin_idx: [N]
            scene_partition_idx: [N]
        '''
        # update memory bank with momentum
        if self.domain_alignment_memory_momentum_decay:
            momentum = self.memory_momentum
            momentum += (1-momentum) * (self.global_step / 10000)
            momentum = torch.clamp(momentum, min=0, max=1)
        else:
            momentum = self.memory_momentum

        getattr(self, f'target_domain_{feature_source}_feature_memory')[class_idx, yaw_bin_idx, scene_partition_idx, :] = momentum * \
                                                                                                                          getattr(self, f'target_domain_{feature_source}_feature_memory')[class_idx, yaw_bin_idx, scene_partition_idx, :] + \
                                                                                                                          (1-momentum) * features

    def extract_object_features(self, feature_source, features, coors, gt_boxes, domain = None):
        '''
            features: [N, C]
            coors: [N, 1 + 3] (bs_idx, x, y, z)
            gt_boxes: [B, N_boxes, 7 + C]
        '''
        batch_size = int(coors[:, 0].max() + 1)
        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            gt_boxes, grid_size=self.domain_alignment_cfg.GRID_POOL.GRID_SIZE
        )  # (BxN_box, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3) # (B, N_box x6x6x6, 3)

        xyz = coors[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = coors[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        _, pooled_features = getattr(self, f'grid_pooling_layer_{feature_source}')(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=features.contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.object_feature_pooling_grid_size ** 3,
            pooled_features.shape[-1]
        )  # (BxN_box, 6x6x6, C)
        if self.separate_fc_layers:
            object_features = getattr(self, f'fc_layer_{feature_source}_{domain}')(pooled_features.reshape(pooled_features.shape[0], -1)) #[BxN_box, C]
        else:
            object_features = getattr(self, f'fc_layer_{feature_source}')(pooled_features.reshape(pooled_features.shape[0], -1)) #[BxN_box, C]
        object_class_idx = gt_boxes[:, :, -1].view(-1) #[BxN_box]
        object_location_size_yaw = gt_boxes[:, :, :7].view(-1, 7) # [BxN_box, 4] xyz yaw

        object_mask = object_class_idx > 0
        features = object_features[object_mask]
        class_idx = object_class_idx[object_mask]
        location_size_yaw = object_location_size_yaw[object_mask]

        # compute scene partition for different objects
        xyz = torch.cat([location_size_yaw.new_zeros(1, 3), location_size_yaw[:, :3]],
                        dim=0)  # append the origin of the coordinate to the list of location
        bins, res_angles_xy, distance_matrix = self.scene_partitioner.compute_partitions(xyz)
        scene_partition_idx = bins[0, 1:]  # [N] indicating bin index for each object
        res_angles_xy = res_angles_xy[0, 1:]  # [N] indicating residual angles of each object in its bin
        distances = distance_matrix[0, 1:]  # [N] indicating distance of each object to the origin

        # compute direction bin indexes and residuals in direction bins
        yaw = location_size_yaw[:, -1]  # [N]
        # angles between 0, 2*pi
        yaw = torch.fmod(yaw + 2 * math.pi, 2 * math.pi)
        yaw_bin_idx = torch.floor(yaw / (2 * math.pi / self.num_dir_bins))  # [N]
        res_yaw = yaw - yaw_bin_idx * (2 * math.pi / self.num_dir_bins)  # [N]

        # aggregate distance and residual information via an MLP head
        local_information = torch.cat([distances.unsqueeze(1), res_angles_xy.unsqueeze(1), location_size_yaw[:,3:6], res_yaw.unsqueeze(1)], dim=1)
        local_feature = getattr(self, f'local_feature_embedding_{feature_source}')(local_information)  # [N, C]

        if self.minus_local_information:
            features -= local_feature
        else:
            # add features about distance and residual information to features
            features += local_feature

        return features, class_idx.long(), yaw_bin_idx.long(), scene_partition_idx.long()

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def prepare_domain_alignment_features(self, batch_dict):
        raise NotImplementedError

    def domain_alignment_loss(self, feature_dict, coors_dict, gt_boxes_dict):
        '''
            feature_dict: feature_source -> domain -> features [N, C]
            coors_dict: feature_source -> domain -> coors [N, 1 + 3] (bs_idx, x, y, z)
            gt_boxes_dict: domain -> gt_boxes [B, N_boxes, 7 + C]
        '''
        # Update memory bank for target domain
        with torch.no_grad():
            for feature_source in self.domain_alignment_feature_source:
                domains_for_memory_update = [self.test_domains[0]]
                for domain in self.batch_domains:
                    if self.bidirection_domain_alignment and domain not in domains_for_memory_update:
                        domains_for_memory_update.append(domain)

                for domain in domains_for_memory_update:
                    if domain in feature_dict[feature_source].keys():
                        features_in_target_domains = feature_dict[feature_source][domain]
                        coors_in_target_domains = coors_dict[feature_source][domain]
                        gt_boxes_in_target_domains = gt_boxes_dict[domain]
                        features, class_idx, yaw_bin_idx, scene_partition_idx = self.extract_object_features(feature_source, features_in_target_domains, coors_in_target_domains, gt_boxes_in_target_domains, domain)
                        # class_idx - 1 because we add 1 to indicate none class in the preparing process
                        class_idx -= 1
                        # conduct class mapping if current domain is not target domain
                        if domain not in self.test_domains:
                            class_idx = getattr(self, f'mapping_class_to_memory_{domain}')[class_idx]
                        # mask out those unmatched class
                        if torch.sum(class_idx < 0) > 0:
                            mask = class_idx >= 0
                            mask = mask.squeeze()
                            class_idx = class_idx[mask]
                            yaw_bin_idx = yaw_bin_idx[mask]
                            scene_partition_idx = scene_partition_idx[mask]
                            features = features[mask]

                        if class_idx.shape[0] > 0:
                            self.update_memory(feature_source, features, class_idx, yaw_bin_idx, scene_partition_idx)

        if self.global_step < self.memory_warmup:
            return None

        loss = []

        domains_for_loss_computation = self.supported_domains
        for domain in self.batch_domains:
            if self.bidirection_domain_alignment and domain not in domains_for_loss_computation:
                domains_for_loss_computation.append(domain)
        # Compute domain feature alignment loss
        for domain in domains_for_loss_computation:
            for feature_source in self.domain_alignment_feature_source:
                if domain in feature_dict[feature_source].keys():
                    features_in_domain = feature_dict[feature_source][domain]
                    coors_in_domain = coors_dict[feature_source][domain]
                    gt_boxes_in_domain = gt_boxes_dict[domain]
                    features, class_idx, yaw_bin_idx, scene_partition_idx = self.extract_object_features(feature_source, features_in_domain, coors_in_domain, gt_boxes_in_domain, domain)
                    if features.shape[0] <= 1 or len(features.shape) < 2:
                        continue
                    # class_idx - 1 because we add 1 to indicate none class in the preparing process
                    class_idx -= 1
                    window_size = self.alignment_window_size

                    # limit yaw bin index in range [0, self.num_dir_bins - 1]
                    yaw_bin_idx = yaw_bin_idx[:, None] + torch.arange(window_size).to(yaw_bin_idx.device) - window_size // 2
                    negative_mask = yaw_bin_idx < 0
                    exceed_mask = yaw_bin_idx >= self.num_dir_bins
                    yaw_bin_idx[negative_mask] = self.num_dir_bins + yaw_bin_idx[negative_mask]
                    yaw_bin_idx[exceed_mask] = yaw_bin_idx[exceed_mask] - self.num_dir_bins
                    # limit scene partition index in range [0, n_partition-1]
                    scene_partition_idx = scene_partition_idx[:, None] + torch.arange(window_size).to(scene_partition_idx.device) - window_size // 2
                    negative_mask = scene_partition_idx < 0
                    exceed_mask = scene_partition_idx >= self.num_scene_partition
                    scene_partition_idx[negative_mask] = self.num_scene_partition + scene_partition_idx[negative_mask]
                    scene_partition_idx[exceed_mask] = scene_partition_idx[exceed_mask] - self.num_scene_partition

                    # conduct class mapping if current domain is not target domain
                    if domain not in self.test_domains:
                        class_idx = getattr(self, f'mapping_class_to_memory_{domain}')[class_idx][:, None]
                    else:
                        class_idx = class_idx[:, None]

                    # mask out those unmatched class
                    if torch.sum(class_idx < 0) > 0:
                        mask = class_idx >= 0
                        mask = mask.squeeze()
                        class_idx = class_idx[mask]
                        yaw_bin_idx = yaw_bin_idx[mask]
                        scene_partition_idx = scene_partition_idx[mask]
                        features = features[mask]

                    if self.loss_func in ['L1', 'L2']:
                        corresponding_features_in_target_domains = getattr(self, f'target_domain_{feature_source}_feature_memory')[class_idx, yaw_bin_idx
                                                                                                                                   , scene_partition_idx , :] # [N, C]
                        if window_size > 1:
                            corresponding_features_in_target_domains, _ = torch.max(corresponding_features_in_target_domains, dim = 1)
                        else:
                            corresponding_features_in_target_domains = torch.mean(corresponding_features_in_target_domains, dim = 1)
                        loss.append(self.domain_alignment_loss_function(features, corresponding_features_in_target_domains))
                    elif self.loss_func in ['CONTRASTIVE']:
                        corresponding_features_in_memory = getattr(self, f'target_domain_{feature_source}_feature_memory')[class_idx, yaw_bin_idx, scene_partition_idx, :].squeeze()  # [N, C]
                        # normalize features
                        corresponding_features_in_memory = torch.nn.functional.normalize(corresponding_features_in_memory, dim=1) # [N, C]
                        features = torch.nn.functional.normalize(features, dim=1) # [N, C]
                        # positive logits
                        l_pos = torch.einsum("nc,nc->n", [features, corresponding_features_in_memory]).unsqueeze(-1)
                        # negative logits
                        l_neg = torch.einsum("nc,kc->nk", [features, features])
                        mask = l_neg.new_ones(l_neg.shape)
                        mask[torch.arange(mask.shape[0]), torch.arange(mask.shape[0])] = 0
                        l_neg = l_neg[mask.bool()].view(l_neg.shape[0], -1)
                        # logits
                        logits = torch.cat([l_pos, l_neg], dim=1)
                        # apply temperature
                        logits /= self.T
                        # labels: positive key indicators
                        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                        loss.append(self.domain_alignment_loss_function(logits, labels) / features.shape[0])
                    else:
                        raise NotImplementedError
        return torch.sum(torch.stack(loss)) if len(loss) > 0 else None