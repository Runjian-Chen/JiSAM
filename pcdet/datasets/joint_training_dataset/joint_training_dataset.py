import pcdet.datasets as datasets
from random import shuffle
import numpy as np
import torch
from collections import defaultdict
from ...utils import common_utils
from itertools import combinations

class JointTrainingDataset(object):
    def __init__(self, dataset_cfg_list, class_names_list, training, root_path_list, logger = None, warmup = False):
        # Currently only one testing dataset is supported
        assert training or ((not training) and (len(dataset_cfg_list) == 1))

        if root_path_list == None:
            root_path_list = [None for i in range(len(dataset_cfg_list))]

        self.domains = []
        self.sample_to_domain_list = []
        self.idx_in_domain = []
        for dataset_cfg, class_names, root_path in zip(dataset_cfg_list, class_names_list, root_path_list):
            dataset_name = dataset_cfg.DATASET
            setattr(self, dataset_name, datasets.__all__[dataset_name](dataset_cfg, class_names, training, root_path, logger))
            dataset_len = len(getattr(self, dataset_name))
            self.domains.append(dataset_name)
            if training:
                for _ in range(dataset_cfg.get('BALANCED_SAMPLE_RATIO', 1)):
                    self.sample_to_domain_list.extend([dataset_name for i in range(dataset_len)])
                    self.idx_in_domain.extend([i for i in range(dataset_len)])
            else:
                self.sample_to_domain_list.extend([dataset_name for i in range(dataset_len)])
                self.idx_in_domain.extend([i for i in range(dataset_len)])



        # Currently different sparse shape is not supported
        sparse_shape_list = []
        for domain in self.domains:
            sparse_shape_list.append(getattr(self, domain).grid_size)
        for pair in combinations(sparse_shape_list, 2):
            assert np.array_equal(pair[0], pair[1]), 'Currently only consistent sparse shape is supported'

        self.shuffle_infos = dataset_cfg.get('SHUFFLE_INFOS', False)
        if training and self.shuffle_infos:
            # Random shuffle all the domain infomation and indexes in each domain
            all_idx = [i for i in range(len(self.sample_to_domain_list))]
            shuffle(all_idx)
            self.sample_to_domain_list = [self.sample_to_domain_list[idx] for idx in all_idx]
            self.idx_in_domain = [self.idx_in_domain[idx] for idx in all_idx]
        else:
            all_idx = [i for i in range(len(self.sample_to_domain_list))]
            self.sample_to_domain_list = [self.sample_to_domain_list[idx] for idx in all_idx]
            self.idx_in_domain = [self.idx_in_domain[idx] for idx in all_idx]

        self.training = training
        self.logger = logger
        self.logger.info(f'Finish building joint training dataset with {len(self.sample_to_domain_list)} samples.')
        self.logger.info(f'There are {len(dataset_cfg_list)} datasets.')
        for dataset_cfg in dataset_cfg_list:
            self.logger.info(f'{dataset_cfg.DATASET}')

        self.warmup = warmup
        if self.warmup:
            self.iter_cnt = 0

    def __len__(self):
        return len(self.idx_in_domain)

    def __getitem__(self, index):
        if not self.warmup or self.iter_cnt > self.warmup:
            # get domain and index inside domain
            domain = self.sample_to_domain_list[index]
            idx = self.idx_in_domain[index]

            data_dict = getattr(self, domain).__getitem__(idx)
            data_dict['domains'] = domain
            data_dict['domain_set'] = self.domains
        else:
            domain = self.sample_to_domain_list[self.iter_cnt]
            idx = self.idx_in_domain[self.iter_cnt]
            # # get domain and index inside domain
            # domain = self.sample_to_domain_list[index]
            # idx = self.idx_in_domain[index]
            # while domain not in ['NuScenesDataset', 'WaymoDataset']:
            #     index = np.random.randint(self.__len__())
            #     domain = self.sample_to_domain_list[index]
            #     idx = self.idx_in_domain[index]
            data_dict = getattr(self, domain).__getitem__(idx)
            data_dict['domains'] = domain
            data_dict['domain_set'] = self.domains
            self.iter_cnt += 1
        return data_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        domain = batch_dict['domains']
        return getattr(self, domain).generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)

    def merge_all_iters_to_one_epoch(self, merge = True, epochs = None):
        raise NotImplementedError

    def disable_joint_training(self, target_dataset = None):
        assert isinstance(target_dataset, str) or (isinstance(target_dataset, list) and len(target_dataset) == 1)

        if isinstance(target_dataset, list):
            target_dataset = target_dataset[0]

        assert getattr(self, target_dataset, None) is not None

        dataset_len = len(getattr(self, target_dataset))
        self.sample_to_domain_list = [target_dataset for i in range(dataset_len)]
        self.domains = [target_dataset]
        self.idx_in_domain = [i for i in range(dataset_len)]

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret_dict = defaultdict(list)
                    for idx, domain in enumerate(data_dict['domains']):
                        ret_dict[domain].append(val[idx])
                    for domain in data_dict['domain_set'][0]:
                        if domain in ret_dict.keys():
                            ret_dict[domain] = np.concatenate(ret_dict[domain], axis=0)
                    ret[key] = ret_dict
                elif key in ['points', 'voxel_coords']:
                    if isinstance(val[0], list):
                        val = [i for item in val for i in item]
                    ret_dict = defaultdict(list)
                    for idx, domain in enumerate(data_dict['domains']):
                        ret_dict[domain].append(val[idx])
                    for domain in data_dict['domain_set'][0]:
                        if domain in ret_dict.keys():
                            coors = []
                            for i, coor in enumerate(ret_dict[domain]):
                                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                                coors.append(coor_pad)
                            ret_dict[domain] = np.concatenate(coors, axis = 0)
                    ret[key] = ret_dict
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    gt_boxes_dict = defaultdict(list)
                    for idx, domain in enumerate(data_dict['domains']):
                        gt_boxes_dict[domain].append(val[idx])
                    gt_boxes_list = []
                    for domain in data_dict['domain_set'][0]:
                        gt_boxes_list.extend(gt_boxes_dict[domain])
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :gt_boxes_list[k].__len__(), :] = gt_boxes_list[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['domains']:
                    ret[key] = val
                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]),
                                                dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len - len(_points)), (0, 0))
                        points_pad = np.pad(_points,
                                            pad_width=pad_width,
                                            mode='constant',
                                            constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['camera_imgs']:
                    ret[key] = torch.stack([torch.stack(imgs, dim=0) for imgs in val], dim=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        batch_size_dict = {domain: 0 for domain in data_dict['domains']}
        for domain in data_dict['domains']:
            batch_size_dict[domain] += 1
        ret['batch_size'] = batch_size_dict
        assert batch_size_ratio == 1

        return ret