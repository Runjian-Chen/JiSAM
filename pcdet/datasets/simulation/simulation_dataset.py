from ..dataset import DatasetTemplate
import copy
import numpy as np
from ...utils.common_utils import rotate_points_along_z
import pickle
import os
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import torch
from tqdm import tqdm
from ...utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

class SimulationDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):

        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)

        self.infos = []
        self.load_interval = self.dataset_cfg.get('LOAD_INTERVAL', 1)
        self.align_to_nuscenes_lidar_coordinate = self.dataset_cfg.get('ALIGN_TO_NUSCENES_LIDAR_COORDINATE', False)
        self.align_to_once_lidar_coordinate = self.dataset_cfg.get('ALIGN_TO_ONCE_LIDAR_COORDINATE', False)
        self.load_points_with_timestamps = self.dataset_cfg.get('LOAD_POINTS_WITH_TIMESTAMPS', True)
        self._merge_all_iters_to_one_epoch = False
        self.include_simulation_data(self.mode)
        self.beam_range_aug = self.dataset_cfg.get('RANGE_BEAM_AUG', False)

        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_simulation_data(self, mode):
        self.logger.info('Loading simulation dataset collected from CARLA')
        simulation_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            simulation_infos.extend(self.load_infos_from_pickle(self.root_path / info_path))

        if self.load_interval > 1:
            simulation_infos = simulation_infos[::self.load_interval]

        self.infos.extend(simulation_infos)
        self.logger.info('Total samples for simulation dataset: %d' % (len(simulation_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        else:
            return len(self.infos)

    def __getitem__(self, index):
        # if all iterations are merged into one epoch
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(info, max_sweeps = self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': str(info['lidar_path']).split('/')[-1].split('.')[0],
            'metadata': {}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = np.ones(info['gt_boxes'].shape[0], dtype=np.bool_)

            # transform gt_boxes into right-hand coordinate system
            gt_boxes = info['gt_boxes'][mask]
            gt_boxes[:, 1] *= -1
            gt_boxes[:, 6] *= -1
            gt_boxes[:, 8] *= -1
            gt_names = info['gt_names'][mask]

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', True):
                assert gt_boxes.shape[-1] == 9
            else:
                gt_boxes = gt_boxes[:, 0:7]

            input_dict.update(
                {'gt_names': gt_names,
                 'gt_boxes': gt_boxes}
            )

        if self.align_to_nuscenes_lidar_coordinate:
            input_dict['points'] = rotate_points_along_z(input_dict['points'][np.newaxis, ...], np.array([np.pi / 2]))[0]
            if 'gt_boxes' in input_dict:
                input_dict['gt_boxes'][:, :3] = rotate_points_along_z(input_dict['gt_boxes'][:, :3][np.newaxis,...], np.array([np.pi / 2]))[0]
                input_dict['gt_boxes'][:, 6] += np.pi / 2
                if input_dict['gt_boxes'].shape[1] > 7:
                    input_dict['gt_boxes'][:, 7:9] = rotate_points_along_z(
                        np.hstack((input_dict['gt_boxes'][:, 7:9], np.zeros((input_dict['gt_boxes'].shape[0], 1))))[np.newaxis, :, :],
                        np.array([np.pi / 2])
                    )[0][:, 0:2]

        if self.align_to_once_lidar_coordinate:
            input_dict['points'] = rotate_points_along_z(input_dict['points'][np.newaxis, ...], np.array([- np.pi / 2]))[0]
            if 'gt_boxes' in input_dict:
                input_dict['gt_boxes'][:, :3] = rotate_points_along_z(input_dict['gt_boxes'][:, :3][np.newaxis, ...], np.array([- np.pi / 2]))[0]
                input_dict['gt_boxes'][:, 6] -= np.pi / 2
                if input_dict['gt_boxes'].shape[1] > 7:
                    input_dict['gt_boxes'][:, 7:9] = rotate_points_along_z(
                        np.hstack((input_dict['gt_boxes'][:, 7:9], np.zeros((input_dict['gt_boxes'].shape[0], 1))))[
                        np.newaxis, :, :],
                        np.array([- np.pi / 2])
                    )[0][:, 0:2]

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib
        data_dict = self.set_lidar_aug_matrix(data_dict)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) < 15:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    def get_lidar_with_sweeps(self, info, max_sweeps=10):

        lidar_path = self.root_path / info['lidar_path']
        points = np.load(lidar_path)['data'] # [N, 4]

        if self.beam_range_aug:
            xyz = points[:, :3]
            std = 0.1
            angle_std = 0.05
            # generate noise
            noise = np.random.normal(loc=0.0, scale=std, size=xyz.shape[0])
            # from xyz to range pitch yaw
            range = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
            theta = np.arccos(xyz[:, 2] / range)  #
            phi = np.sign(xyz[:, 1]) * np.arccos(xyz[:, 0] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2))  # theta
            # add noise to range
            range += noise
            if angle_std > 0:
                theta += np.random.normal(loc=0.0, scale=angle_std, size=xyz.shape[0])
                phi += np.random.normal(loc=0.0, scale=angle_std, size=xyz.shape[0])

            # from range pitch yaw to xyz
            x = range * np.sin(theta) * np.cos(phi)
            y = range * np.sin(theta) * np.sin(phi)
            z = range * np.cos(theta)
            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = z

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        if max_sweeps > 1:
            num_sweeps = min(max_sweeps, len(info['sweeps'])+1)
            for k in np.random.choice(len(info['sweeps']), num_sweeps - 1, replace = False):
                sweep = info['sweeps'][k]
                sweep_points = np.load(self.root_path / sweep['lidar_path'])['data']
                # remove ego points
                mask = ~((np.abs(sweep_points[:, 0]) < 1.0) & (np.abs(sweep_points[:, 1]) < 1.0))
                sweep_points = sweep_points[mask]
                # transform sweep points into current coordinate system
                sweep_points = sweep_points.T
                sweep_points[:3, :] = sweep['transformation_matrix'].dot(np.vstack((sweep_points[:3, :], np.ones(sweep_points.shape[1]))))[:3, :]

                if self.beam_range_aug:
                    sweep_points = sweep_points.T
                    xyz = sweep_points[:, :3]
                    std = 0.1
                    angle_std = 0.05
                    # generate noise
                    noise = np.random.normal(loc=0.0, scale=std, size=xyz.shape[0])
                    # from xyz to range pitch yaw
                    range = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
                    theta = np.arccos(xyz[:, 2] / range)  #
                    phi = np.sign(xyz[:, 1]) * np.arccos(xyz[:, 0] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2))  # theta
                    # add noise to range
                    range += noise
                    if angle_std > 0:
                        theta += np.random.normal(loc=0.0, scale=angle_std, size=xyz.shape[0])
                        phi += np.random.normal(loc=0.0, scale=angle_std, size=xyz.shape[0])

                    # from range pitch yaw to xyz
                    x = range * np.sin(theta) * np.cos(phi)
                    y = range * np.sin(theta) * np.sin(phi)
                    z = range * np.cos(theta)
                    sweep_points[:, 0] = x
                    sweep_points[:, 1] = y
                    sweep_points[:, 2] = z

                    sweep_points = sweep_points.T

                sweep_points_list.append(sweep_points.T)
                sweep_times_list.append(sweep['time_lag'] * np.ones((sweep_points.shape[1], 1)))

        points = np.concatenate(sweep_points_list, axis = 0)
        times = np.concatenate(sweep_times_list, axis = 0)

        # The coordinate system in CARLA simulation is left-handed
        # Thus we need to convert it to right-handed coordinate system
        points[:, 1] *= -1
        if self.load_points_with_timestamps:
            points = np.concatenate((points, times), axis = 1)
        else:
            points = points
        return points

    def load_infos_from_pickle(self, info_path):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos

    def create_groundtruth_database(self, max_sweeps = 10, prefix = None):
        database_save_path = self.root_path / f'{prefix}_gt_database_{max_sweeps}_sweeps'
        db_info_save_path = self.root_path / f'{prefix}_dbinfos_{max_sweeps}_sweeps_train.pkl'
        db_data_save_path = self.root_path / f'{prefix}_gt_database_train_global.npy'

        if db_info_save_path.exists():
            with open(db_info_save_path, 'rb') as f:
                all_db_infos = pickle.load(f)
            save_at = all_db_infos['save_at']
            # stacked_gt_points = np.load(db_data_save_path).tolist()
        else:
            all_db_infos = {}
            save_at = -1
            # stacked_gt_points = []

        database_save_path.mkdir(parents = True, exist_ok = True)
        total_object_cnt = 0
        if save_at > 0:
            # total object count
            raise NotImplementedError

        for idx in tqdm(range(save_at+1, len(self.infos))):
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(info)
            gt_boxes = info['gt_boxes']
            gt_boxes[:,1] *= -1.0
            gt_boxes[:,6] *= -1.0
            gt_boxes[:, 8] *= -1.0

            # align coordinate to nuscenes's
            if self.align_to_nuscenes_lidar_coordinate:
                points[:,:3] = rotate_points_along_z(points[:,:3][np.newaxis, ...], np.array([np.pi / 2]))[0]
                gt_boxes[:, :3] = rotate_points_along_z(gt_boxes[:,:3][np.newaxis, ...], np.array([np.pi/2]))[0]
                gt_boxes[:, 6] += np.pi / 2
                if gt_boxes.shape[1] > 7:
                    gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                        np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                        np.array([np.pi / 2])
                    )[0][:, 0:2]

            # align coordinate to once's
            if self.align_to_once_lidar_coordinate:
                points[:, :3] = rotate_points_along_z(points[:, :3][np.newaxis, ...], np.array([- np.pi / 2]))[0]
                gt_boxes[:, :3] = rotate_points_along_z(gt_boxes[:, :3][np.newaxis, ...], np.array([- np.pi / 2]))[0]
                gt_boxes[:, 6] -= np.pi / 2
                if gt_boxes.shape[1] > 7:
                    gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                        np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                        np.array([- np.pi / 2])
                    )[0][:, 0:2]

            gt_names = info['gt_names']

            box_idx_of_pts = points_in_boxes_gpu(
                torch.from_numpy(points[:, :3]).unsqueeze(0).float().cuda(),
                torch.from_numpy(gt_boxes[:, :7]).unsqueeze(0).float().cuda()
            ).long().squeeze(0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idx_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if gt_points.shape[0] < 10:
                    continue

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                total_object_cnt += 1

                # stacked_gt_points.append(gt_points)
                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': idx, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}

                if gt_names[i] in all_db_infos:
                    all_db_infos[gt_names[i]].append(db_info)
                else:
                    all_db_infos[gt_names[i]] = [db_info]

            if idx % 500 == 0:
                all_db_infos['save_at'] = idx
                with open(db_info_save_path, 'wb') as f:
                    pickle.dump(all_db_infos, f)

                # stacked_gt_points_ = np.concatenate(stacked_gt_points, axis=0)
                # np.save(db_data_save_path, stacked_gt_points_)

            if total_object_cnt > 500000:
                split = total_object_cnt // 500000
                database_save_path = self.root_path / f'{prefix}_gt_database_{max_sweeps}_sweeps_{split}_split'
                database_save_path.mkdir(parents = True, exist_ok = True)



        if 'save_at' in all_db_infos.keys():
            all_db_infos.pop('save_at')

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        # np.save(db_data_save_path, stacked_gt_points)

def create_simulation_info(data_path, max_sweeps, lidar_config, filtering_option = False):
    '''

    :param data_path:
    :param max_sweeps:
    :param lidar_config: currently only support rotation frequency
    :return:
    '''
    infos = []

    for scenario in os.listdir(data_path / 'ego_vehicle/lidar01/'):
        for idx, filename in enumerate(os.listdir(data_path / ('ego_vehicle/lidar01/' + scenario))):

            # scenario id and frame id
            scene_idx = filename.split('.')[0]

            calib_path = data_path / 'ego_vehicle/calib' / scenario / (scene_idx + '.pkl')
            lidar_path = data_path / 'ego_vehicle/lidar01' / scenario / (scene_idx + '.npz')
            label_path = data_path / 'ego_vehicle/label' / scenario / (scene_idx + '.txt')

            # make sure calibration, label and point cloud files exist
            if not (calib_path.exists() and label_path.exists() and lidar_path.exists()):
                print('invalid sample: %s' % scene_idx)
                continue

            # compute point numbers for each object
            points = np.load(lidar_path)['data']
            points[:, 1] = -points[:, 1]
            label_file = data_path / 'ego_vehicle/label/' / scenario / (scene_idx + '.txt')

            with open(label_file) as f:
                lines = [line.rstrip('\n') for line in f]

            gt_bboxes_3d = []
            gt_names = []

            for line in lines:
                if len(line.split(' ')) <= 2:
                    continue

                line_info = line.split(' ')
                if line_info[8] == -1:
                    continue
                cls_label = line_info[0]
                box_info = line_info[1:10]
                box_info = list((map(float, box_info)))
                box_info[1] = -box_info[1]
                # yaw 角从左手系变成右手系
                box_info[6] = -box_info[6]
                gt_bboxes_3d.append(box_info)
                gt_names.append(cls_label)

            gt_bboxes_3d = np.stack(gt_bboxes_3d)
            gt_names = np.stack(gt_names)

            points = torch.from_numpy(points).unsqueeze(0).cuda().float()
            gt_bboxes_3d = torch.from_numpy(gt_bboxes_3d).unsqueeze(0).cuda().float()
            pts_box_idx = points_in_boxes_gpu(points[:,:,:3], gt_bboxes_3d[:,:,:7])
            gt_bboxes_3d = gt_bboxes_3d.squeeze(0)
            points = points.squeeze(0)
            pts_box_idx = pts_box_idx.squeeze(0).cpu().numpy()
            num_lidar_pts = []
            for obj_id in range(gt_bboxes_3d.shape[0]):
                num_lidar_pts.append(np.sum(pts_box_idx == obj_id))
            num_lidar_pts = np.array(num_lidar_pts)

            # filter all the ground truth boxes if required
            if filtering_option:
                # For poles and buildings, recomputes the gt_bboxes
                for obj_id, (name, num_pts) in enumerate(zip(gt_names, num_lidar_pts)):
                    if name in ['pole'] and num_pts > 25:
                        obj_pts = points[pts_box_idx == obj_id]
                        obj_pts = obj_pts[:, :3]
                        obj_pts = obj_pts[obj_pts[:,2] > -1.8]
                        if obj_pts.shape[0] < 25:
                            continue
                        location = torch.mean(obj_pts, dim=0)
                        max_xyz, _ = obj_pts.max(dim=0)
                        min_xyz, _ = obj_pts.min(dim=0)
                        size = max_xyz - min_xyz
                        other_infos = gt_bboxes_3d[obj_id, 6:]
                        new_box = torch.cat([location, size, other_infos])
                        gt_bboxes_3d[obj_id] = new_box
                    if name in ['building'] and num_pts > 25:
                        obj_pts = points[pts_box_idx == obj_id]
                        obj_pts = obj_pts[:, :3]
                        obj_pts = obj_pts[obj_pts[:, 2] > -1.8]
                        if obj_pts.shape[0] < 25:
                            continue
                        xy = gt_bboxes_3d[obj_id,:2]
                        z = torch.mean(obj_pts[:, 2], dim=0, keepdim=True)
                        max_xyz, _ = obj_pts.max(dim=0)
                        min_xyz, _ = obj_pts.min(dim=0)
                        xy_size = gt_bboxes_3d[obj_id, 3:5]
                        z_size = max_xyz[-1:] - min_xyz[-1:]
                        other_infos = gt_bboxes_3d[obj_id, 6:]
                        new_box = torch.cat([xy, z, xy_size, z_size, other_infos])
                        gt_bboxes_3d[obj_id] = new_box

                # compute overlap of all the bounding boxes
                iou = boxes_iou3d_gpu(gt_bboxes_3d[:, :7], gt_bboxes_3d[:, :7]) # [N_box, N_box]
                iou[torch.arange(iou.shape[0]), torch.arange(iou.shape[1])] = 0
                gt_bboxes_3d = gt_bboxes_3d.cpu().numpy()
                iou = iou.cpu().numpy()
                # Filter boxes overlaps with others and have fewer points
                mask = np.zeros(gt_bboxes_3d.shape[0])
                mask = mask.astype(np.bool_)
                for box_id in range(mask.shape[0]):
                    iou_mask = iou[box_id] > 0.4
                    pts_in_boxes = num_lidar_pts[iou_mask]
                    if pts_in_boxes.shape[0] == 0 or num_lidar_pts[box_id] == np.max(pts_in_boxes):
                        mask[box_id] = True
                    else:
                        mask[box_id] = False
                # filter out those with too large size and too few points
                mask = mask & (num_lidar_pts > 25) & (gt_bboxes_3d[:, 3] < 10) & (gt_bboxes_3d[:, 4] < 10)

                gt_bboxes_3d = gt_bboxes_3d[mask]
                gt_names = gt_names[mask]
                num_lidar_pts = num_lidar_pts[mask]

            # transform the boxes back to left-hand coordinate system
            gt_bboxes_3d = gt_bboxes_3d.cpu().numpy()
            gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
            gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6]

            with open(calib_path, 'rb') as f:
                calib_dict = pickle.load(f)
            # world to lidar
            world2lidar = np.linalg.inv(np.dot(calib_dict['ego_to_world'], calib_dict['lidar_to_ego']))

            # generate sweeps information for each sample
            sweeps = []
            frame_id = scene_idx.split('_')[-1]
            frame_id = int(frame_id)
            frame_id_ = copy.deepcopy(frame_id)
            while len(sweeps) < max_sweeps:
                if frame_id_ == 1:
                    break
                else:
                    frame_id_ -= 1
                    sweep = {}
                    sweep_idx = scene_idx[:scene_idx.rfind('_')] + '_' + str(frame_id_).zfill(3)
                    sweep['lidar_path'] = 'ego_vehicle/lidar01/' + scenario + '/' + sweep_idx + '.npz'

                    sweep_calib_path = data_path / 'ego_vehicle/calib' / scenario / (sweep_idx + '.pkl')
                    with open(sweep_calib_path, 'rb') as f:
                        sweep_calib_dict = pickle.load(f)

                    # sweep lidar to world
                    sweep_lidar2world = np.dot(sweep_calib_dict['ego_to_world'], sweep_calib_dict['lidar_to_ego'])

                    # sweep lidar to current lidar
                    sweep_lidar2lidar = np.dot(world2lidar, sweep_lidar2world)

                    sweep['transformation_matrix'] = sweep_lidar2lidar
                    sweep['time_lag'] = (frame_id - frame_id_) * 1.0 / lidar_config['rotation_frequency']
                    sweeps.append(sweep)

            info = {}
            info['lidar_path'] = lidar_path
            info['sweeps'] = sweeps
            info['num_lidar_pts'] = num_lidar_pts
            info['gt_boxes'] = gt_bboxes_3d
            info['gt_names'] = gt_names
            infos.append(info)

            print(f'finish {len(infos)} samples')

    info_path = data_path / ('infos_' + str(len(infos)) + '_samples_10_sweeps.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(infos, f)


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict
    from pathlib import Path
    from ...utils import common_utils

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for dataset')
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--func', type=str, default='create_simulation_infos', help='function to run')
    parser.add_argument('--filtering_option', action='store_true', default=False, help='whether to filter overlapped bounding boxes and adapt gt boxes')
    parser.add_argument('--align_to_nuscenes_lidar_coordinate', action='store_true', default=False, help='whether to align_to_nuscenes_lidar_coordinate')
    parser.add_argument('--align_to_once_lidar_coordinate', action='store_true', default=False, help='whether to align_to_once_lidar_coordinate')
    parser.add_argument('--load_interval', type=int, default=1, help='')

    args = parser.parse_args()

    if args.func == 'create_simulation_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_simulation_info(
            data_path = ROOT_DIR / dataset_cfg.DATA_PATH.replace('../', ''),
            max_sweeps = dataset_cfg.MAX_SWEEPS,
            lidar_config = dataset_cfg.LIDAR_CONFIG,
            filtering_option=args.filtering_option
        )

    elif args.func == 'create_simulation_groundtruth_database':
        assert args.prefix is not None
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        dataset_cfg.ALIGN_TO_NUSCENES_LIDAR_COORDINATE = args.align_to_nuscenes_lidar_coordinate
        dataset_cfg.ALIGN_TO_ONCE_LIDAR_COORDINATE = args.align_to_once_lidar_coordinate
        dataset_cfg.LOAD_INTERVAL = args.load_interval
        simulation_dataset = SimulationDataset(
            dataset_cfg=dataset_cfg,
            class_names=None,
            root_path=ROOT_DIR / dataset_cfg.DATA_PATH.replace('../', ''),
            logger=common_utils.create_logger(), training=True
        )
        simulation_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS, prefix = args.prefix)

    elif args.func == 'visualize_database_for_nuscenes':
        from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
        from pcdet.config import cfg_from_yaml_file
        cfg = EasyDict()
        cfg = cfg_from_yaml_file(args.cfg_file, cfg)
        dataset_cfg = cfg.DATA_CONFIG
        ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
        dataset_cfg.VERSION = 'v1.0-trainval'
        nuscenes_dataset = NuScenesDataset(
            dataset_cfg = dataset_cfg, class_names = cfg.CLASS_NAMES,
            root_path= ROOT_DIR / dataset_cfg.DATA_PATH,
            logger=common_utils.create_logger(), training=True
        )

        sample_data = nuscenes_dataset.__getitem__(0)
        points = sample_data['points']
        gt_boxes = sample_data['gt_boxes']

        np.save('./temp/points.npy', points)
        np.save('./temp/gt_boxes.npy', gt_boxes)
        # from tools.visual_utils.open3d_vis_utils import draw_scenes
        # draw_scenes(points, gt_boxes)

    elif args.func == 'visualize_simulation_scenes':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        dataset_cfg.ALIGN_TO_NUSCENES_LIDAR_COORDINATE = True
        simulation_dataset = SimulationDataset(
            dataset_cfg=dataset_cfg,
            class_names=dataset_cfg.CLASS_NAMES,
            root_path=ROOT_DIR / dataset_cfg.DATA_PATH.replace('../', ''),
            logger=common_utils.create_logger(), training=True
        )

        data_dict = simulation_dataset.__getitem__(99)
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']
        print(data_dict['frame_id'])

        np.save('./tools/temp/points.npy', points)
        np.save('./tools/temp/gt_boxes.npy', gt_boxes)

    elif args.func == 'update_database':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        simulation_dataset = SimulationDataset(
            dataset_cfg=dataset_cfg,
            class_names=['car', 'truck', 'bus', 'motorcycle', 'cyclist', 'pedestrian'],
            root_path=ROOT_DIR / dataset_cfg.DATA_PATH.replace('../', ''),
            logger=common_utils.create_logger(), training=True
        )

        gt_sampler = simulation_dataset.data_augmentor.data_augmentor_queue[0]
        db_infos = gt_sampler.db_infos
        for cur_class in db_infos.keys():
            db_info = db_infos[cur_class]
            for info in tqdm(db_info):
                file_path = gt_sampler.root_path / info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, gt_sampler.sampler_cfg.NUM_POINT_FEATURES])

                obj_points = obj_points[:,:4]

                with open(file_path, 'w') as f:
                    obj_points.tofile(f)

    else:
        raise NotImplementedError

