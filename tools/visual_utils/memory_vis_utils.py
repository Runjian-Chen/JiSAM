import argparse
from pcdet.config import cfg_from_yaml_file_joint_training, cfg
from pcdet.utils import common_utils
import datetime
from pcdet.datasets import build_joint_training_dataloader
import os
from tensorboardX import SummaryWriter
from pcdet.models import build_network_joint_training
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_model(model, ckpt_file, logger):
    model.load_params_from_file(ckpt_file, logger)
    return model

def build_model(cfg, ckpt_file):
    # Currently NuScenes and CARLA Simulation datasets are supported
    supported_datasets = ['NuScenesDataset', 'SimulationDataset']
    dataset_cfg_list = [v for k, v in cfg.DATA_CONFIG.items() if k in supported_datasets]
    class_names_list = [dataset_cfg.CLASS_NAMES for dataset_cfg in dataset_cfg_list]

    os.makedirs('../output/Memory_visualization', exist_ok=True)
    log_file = '../output/Memory_visualization/Memory_visualization_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    train_set, train_loader, train_sampler = build_joint_training_dataloader(
        dataset_cfg_list=dataset_cfg_list,
        class_names_list=class_names_list,
        batch_size=4,
        dist=False, workers=4,
        logger=logger,
        training=True,
        total_epochs=20,
    )

    domains = train_set.domains
    num_class_dict = {}
    for domain in domains:
        num_class_dict[domain] = len(getattr(train_set, domain).class_names)
    cfg.MODEL.TEST_DOMAINS = cfg.TEST_DATASETS

    model = build_network_joint_training(model_cfg=cfg.MODEL, num_class=num_class_dict, dataset=train_set,
                                         domains=domains)

    tb_log = {}
    for domain in model.test_domains:
        tb_log[domain] = SummaryWriter(log_dir=str(f'../output/Memory_visualization/{ckpt_file.split("/")[-1].split(".")[0]}_{domain}'))

    return model, tb_log, logger

def draw_memory_visualization(model, tb_log):
    target_domain = model.test_domains[0]
    feature_source = model.domain_alignment_feature_source
    tensorboard_writer = tb_log[target_domain]

    for source in feature_source:
        memory = getattr(model, f'target_domain_{source}_feature_memory') # [N_class, N_yaw_bin, N_scene_partition, C]
        '''
            Visualize features of cars with the same yaw in different scene partition
        '''
        # For cars
        car_memory = memory[0]
        # Get all the features from different partitions
        features = car_memory.view(-1, car_memory.shape[-1])
        labels = torch.arange(car_memory.shape[1]).unsqueeze(0).repeat(car_memory.shape[0], 1).view(-1)

        tsne_img = visualize_tsne(features, labels, labels, reduce_ratio=0.1)
        tensorboard_writer.add_image(f'memory_visualization/car_with_scene_partition_label', torch.from_numpy(np.copy(tsne_img)).permute(2,0,1), 1)

def visualize_tsne(feats, labels, labels_name, reduce_ratio = 1.0):
    # feats: [N, C]
    # labels: [N] min -1 max N_c

    reduce_len = int(len(feats) * reduce_ratio)
    # idx = torch.randperm(feats.shape[0])[:reduce_len]
    feats = feats[:reduce_len,:]
    labels = labels[:reduce_len]

    feats_pd = pd.DataFrame(feats)
    df_feats = feats_pd

    labels_pd = pd.DataFrame(labels)
    label_list = labels_pd.values.reshape([reduce_len])
    df_feats['label'] = labels_pd

    # Compute tSNE
    tsne = TSNE(init='pca')
    tsne_results = tsne.fit_transform(df_feats.values)

    # Plot TSNE results on image
    fig = plt.figure(figsize=(15, 15))

    color_array = label_list.copy().astype(np.float)
    color_array -= np.min(color_array)
    color_array /= np.max(color_array)
    cm = plt.get_cmap('gist_rainbow')
    styles = ['o', 'P', '*', 'X']

    for i in range(tsne_results.shape[0]):
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=cm(color_array[i]), marker = styles[label_list[i]%4], label=labels_name[int(label_list[i])], s=40)

    plt.xticks([])
    plt.yticks([])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 24},loc='upper right')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str)
    parser.add_argument('--ckpt_file', type=str)

    args = parser.parse_args()
    cfg = cfg_from_yaml_file_joint_training(args.cfg_file, cfg)

    model, tb_log, logger = build_model(cfg, args.ckpt_file)
    model = load_model(model, args.ckpt_file, logger)
    draw_memory_visualization(model, tb_log)