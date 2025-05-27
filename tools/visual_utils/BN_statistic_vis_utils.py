import os
import argparse
from pcdet.config import cfg_from_yaml_file_joint_training, cfg
from pcdet.datasets import build_joint_training_dataloader
from pcdet.utils import common_utils
import datetime
from pcdet.models import build_network_joint_training
from tensorboardX import SummaryWriter
from collections import defaultdict
from pcdet.models.model_utils.joint_training_utils import SeparateBN

def get_bn_layers(model):
    bn_layers = []
    for child in model.children():
        if isinstance(child, SeparateBN):
            bn_layers.append(child)
        else:
            bn_layers.extend(get_bn_layers(child))

    return bn_layers

def get_bn_params(bn_layers):
    bn_params = defaultdict(list)
    for bn_layer in bn_layers:
        for domain in bn_layers[0].domains:
            bn_params[domain].append((getattr(bn_layer, f'running_mean_{domain}'), getattr(bn_layer, f'running_var_{domain}')))
    return bn_params

def draw_bn_statistic(model, tb_log):
    bn_layers = get_bn_layers(model)
    bn_params = get_bn_params(bn_layers)
    for domain in bn_params.keys():
        for layer, (mean, var) in enumerate(bn_params[domain]):
            for channel, (mean_, var_) in enumerate(zip(mean, var)):
                tb_log[domain].add_scalar(f'statistic_visualization/bn_{channel}_channel_mean', mean_, global_step=layer)
                tb_log[domain].add_scalar(f'statistic_visualization/bn_{channel}_channel_var', var_, global_step=layer)

def load_model(model, ckpt_file, logger):
    model.load_params_from_file(ckpt_file, logger)
    return model

def build_model(cfg, ckpt_file):
    # Currently NuScenes and CARLA Simulation datasets are supported
    supported_datasets = ['NuScenesDataset', 'SimulationDataset']
    dataset_cfg_list = [v for k, v in cfg.DATA_CONFIG.items() if k in supported_datasets]
    class_names_list = [dataset_cfg.CLASS_NAMES for dataset_cfg in dataset_cfg_list]

    os.makedirs('../output/BN_statistic_visualization', exist_ok=True)
    log_file = '../output/BN_statistic_visualization/BN_statistic_visualization_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
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
    tb_log = {}
    for domain in domains:
        tb_log[domain] = SummaryWriter(log_dir=str(f'../output/BN_statistic_visualization/{ckpt_file.split("/")[-1].split(".")[0]}_{domain}'))

    num_class_dict = {}
    for domain in domains:
        num_class_dict[domain] = len(getattr(train_set, domain).class_names)
    cfg.MODEL.TEST_DOMAINS = cfg.TEST_DATASETS

    model = build_network_joint_training(model_cfg=cfg.MODEL, num_class=num_class_dict, dataset=train_set,
                                         domains=domains)

    return model, tb_log, logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str)
    parser.add_argument('--ckpt_file', type=str)

    args = parser.parse_args()
    cfg = cfg_from_yaml_file_joint_training(args.cfg_file, cfg)

    model, tb_log, logger = build_model(cfg, args.ckpt_file)
    model = load_model(model, args.ckpt_file, logger)
    draw_bn_statistic(model, tb_log)
