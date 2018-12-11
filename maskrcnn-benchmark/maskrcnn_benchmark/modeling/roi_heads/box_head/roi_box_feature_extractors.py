# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler, make_box_pooler


class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        self.pooler = make_box_pooler(cfg)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = make_box_pooler(cfg)
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FPNAdaptiveFeaturePoolingPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPNAdaptiveFeaturePoolingPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = make_box_pooler(cfg)

        parallel_layers = (representation_size, representation_size, representation_size, representation_size)
        self.parallel_block = []
        for layer_idx, layer_features in enumerate(parallel_layers, 1):
            layer_name = "box_fc6_parallel{}".format(layer_idx)
            module = nn.Linear(input_size, layer_features)
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            self.parallel_block.append(layer_name)

        self.fc7 = nn.Linear(representation_size, representation_size)
        nn.init.kaiming_uniform_(self.fc7.weight, a=1)
        nn.init.constant_(self.fc7.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        # FC6 for each level of Feature grid
        for idx, layer_name in enumerate(self.parallel_block):
            x[idx] = x[idx].view(x.size(0), -1)
            x[idx] = F.relu(getattr(self, layer_name)(x[idx]))
        # Fusion (max)
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]

        x = F.relu(self.fc7(x))

        return x


_ROI_BOX_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "FPN2MLPFeatureExtractor": FPN2MLPFeatureExtractor,
    "FPNAdaptiveFeaturePoolingPFeatureExtractor": FPNAdaptiveFeaturePoolingPFeatureExtractor,
}


def make_roi_box_feature_extractor(cfg):
    func = _ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
