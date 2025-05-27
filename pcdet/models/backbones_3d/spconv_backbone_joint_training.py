import copy
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch
import torch.nn as nn

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out



class VoxelResBackBone8xJointTraining(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, domains, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        use_norm = self.model_cfg.get('USE_NORM', True)
        if use_norm:
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            norm_fn = partial(nn.Identity)

        self.input_channels = input_channels
        self.domains = domains
        self.sparse_shape = copy.deepcopy(grid_size)
        for domain in self.domains:
            self.sparse_shape[domain] = self.sparse_shape[domain][::-1] + [1, 0, 0]

        # build input convolution for different domains
        for domain in self.domains:
            setattr(self, 'conv_input_'+domain, spconv.SparseSequential(
                spconv.SubMConv3d(input_channels[domain], 16, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(16),
                nn.ReLU(),
            ))

        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: dict {domain: int}
                vfe_features: dict {domain: (num_voxels, C)}
                voxel_coords: dict {domain: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]}
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        feature_list = []
        voxel_coords_list = []
        domain_list = []
        acc_batch_size = [0]
        # input convolution for different domains
        for domain in batch_dict['batch_domains']:
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features[domain],
                indices=voxel_coords[domain].int(),
                spatial_shape=self.sparse_shape[domain],
                batch_size=batch_size[domain]
            )

            tensor_after_input_conv = getattr(self, 'conv_input_'+domain)(input_sp_tensor)
            feature_list.append(tensor_after_input_conv._features)
            voxel_coords_list.append(tensor_after_input_conv.indices)
            voxel_coords_list[-1][:, 0] += acc_batch_size[-1]
            # Currently spatial shape should be the same
            sparse_shape = tensor_after_input_conv.spatial_shape

            domain_list.extend([domain for i in range(batch_size[domain])])
            acc_batch_size.append(batch_size[domain] + acc_batch_size[-1])

            # Align conv input if specified
            if batch_dict.get('conv_input_alignment_targets', None) is not None:
                if domain in batch_dict.get('conv_input_alignment_sources'):
                    target_domain = batch_dict.get('conv_input_alignment_targets')[batch_dict.get('conv_input_alignment_sources').index(domain)]
                    target_domain_input_channels = self.input_channels[target_domain]

                    truncated_input_sp_tensor = spconv.SparseConvTensor(
                        features=voxel_features[domain][:, :target_domain_input_channels],
                        indices=voxel_coords[domain].int(),
                        spatial_shape=self.sparse_shape[domain],
                        batch_size=batch_size[domain]
                    )

                    target_tensor_after_input_conv = getattr(self, 'conv_input_'+target_domain)(truncated_input_sp_tensor)

                    batch_dict[domain + '_conv_input_align_to_' + target_domain + '_source_tensor'] = tensor_after_input_conv._features
                    batch_dict[domain + '_conv_input_align_to_' + target_domain + '_target_tensor'] = target_tensor_after_input_conv._features

        fused_tensor = spconv.SparseConvTensor(
            features=torch.cat(feature_list, dim=0).squeeze(0), # if there is only one domain, a new dim will appear
            indices=torch.cat(voxel_coords_list, dim=0).squeeze(0),
            spatial_shape=sparse_shape,
            batch_size=acc_batch_size[-1]
        )

        # ToDo: add domain information in fused tensor and test on transfusion nuscenes

        x_conv1 = self.conv1(fused_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
            'domain_list': domain_list,
            'total_batch_size': acc_batch_size[-1]
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict