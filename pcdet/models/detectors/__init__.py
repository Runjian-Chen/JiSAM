from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion
from .detector3d_template_joint_training import Detector3DTemplateJointTraining
from .transfusion_joint_training import TransFusionJointTraining
from .pv_rcnn_plusplus_joint_training import PVRCNNPlusPlusJointTraining
from .centerpoint_joint_training import CenterPointJointTraining
from .second_net_joint_training import SECONDNetJointTraining
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Detector3DTemplateJointTraining': Detector3DTemplateJointTraining,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
    'TransFusionJointTraining': TransFusionJointTraining,
    'PVRCNNPlusPlusJointTraining': PVRCNNPlusPlusJointTraining,
    'CenterPointJointTraining': CenterPointJointTraining,
    'SECONDNetJointTraining': SECONDNetJointTraining
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

def build_detector_joint_training(model_cfg, num_class, dataset, domains):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, domains=domains
    )

    return model
