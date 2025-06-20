from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
from .spconv_unet import UNetV2
from .spconv_backbone_joint_training import VoxelResBackBone8xJointTraining
from .spconv_backbone_joint_training_with_separate_norm import VoxelResBackBone8xJointTrainingSeparateNorm
from .spconv_backbone_joint_training_ablation import VoxelResBackBone8xJointTrainingAblation

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'VoxelResBackBone8xJointTraining': VoxelResBackBone8xJointTraining,
    'VoxelResBackBone8xJointTrainingSeparateNorm': VoxelResBackBone8xJointTrainingSeparateNorm,
    'VoxelResBackBone8xJointTrainingAblation': VoxelResBackBone8xJointTrainingAblation
}
