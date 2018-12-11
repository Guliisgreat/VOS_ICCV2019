# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler, make_mask_pooler
from maskrcnn_benchmark.layers import Conv2d


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        pooler = make_mask_pooler(cfg)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


class MaskRCNNFPNAdaptiveFeaturePoolingExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNAdaptiveFeaturePoolingExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        pooler = make_mask_pooler(cfg)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        parallel_layers = (256, 256, 256, 256)
        layers = (256, 256, 256)

        self.parallel_block = []
        for layer_idx, layer_features in enumerate(parallel_layers, 1):
            layer_name = "mask_fcn_parallel{}".format(layer_idx)
            module = Conv2d(input_size, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            self.parallel_block.append(layer_name)

        next_feature = 256
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 2):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        # ROI Pooling --> Feature grid for each ROI proposal
        x = self.pooler(x, proposals)
        # FCN for each level of Feature grid
        for idx, layer_name in enumerate(self.parallel_block):
            x[idx] = F.relu(getattr(self, layer_name)(x[idx]))
        # Fusion (max)
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]

        # The rest of three FCN
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x




_ROI_MASK_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
    "MaskRCNNFPNAdaptiveFeaturePoolingExtractor": MaskRCNNFPNAdaptiveFeaturePoolingExtractor,
}


def make_roi_mask_feature_extractor(cfg):
    func = _ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
