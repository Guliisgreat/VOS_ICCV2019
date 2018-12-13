# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler, make_mask_pooler
from maskrcnn_benchmark.layers import Conv2d
from .roi_mask_predictors import make_roi_mask_predictor


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


class PANetMaskBranch(nn.Module):
    """
    Fully connected Fusion head for mask branch in mask_rcnn
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(PANetMaskBranch, self).__init__()

        pooler = make_mask_pooler(cfg)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler


        parallel_layers = (256, 256, 256, 256)
        common_fcn_layers = (256, 256)

        # Parallel Block: 4 parallel fcn1
        self.parallel_block = []
        for layer_idx, layer_features in enumerate(parallel_layers, 1):
            layer_name = "mask_fcn_parallel{}".format(layer_idx)
            module = Conv2d(input_size, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            self.parallel_block.append(layer_name)
        # Common Block: fcn2, fcn3
        self.common_blocks = []
        for layer_idx, layer_features in enumerate(common_fcn_layers, 2):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(layer_features, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            self.common_blocks.append(layer_name)

        layer_features = 256
        # FCN branch: fcn4 + original_mask_predictor
        self.mask_fcn_4 = Conv2d(layer_features, layer_features, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.mask_fcn_4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.mask_fcn_4.bias, 0)

        self.mask_predictor = make_roi_mask_predictor(cfg)

        # FC Branch: conv4_fc, conv5_fc, fc
        # fc_fcn_layers = (256, 128)
        # self.fc_blocks = []
        # for layer_idx, layer_features in enumerate(fc_fcn_layers, 4):
        #     layer_name = "mask_fc_fcn{}".format(layer_idx)
        #     module = Conv2d(layer_features, layer_features, 3, stride=1, padding=1)
        #     # Caffe2 implementation uses MSRAFill, which in fact
        #     # corresponds to kaiming_normal_ in PyTorch
        #     nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        #     nn.init.constant_(module.bias, 0)
        #     self.add_module(layer_name, module)
        #     self.fc_blocks.append(layer_name)

        # FC Branch: conv4_fc, conv5_fc, fc
        self.mask_fc_fcn_4 = Conv2d(256, 256, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.mask_fc_fcn_4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.mask_fc_fcn_4.bias, 0)

        self.mask_fc_fcn_5 = Conv2d(256, 128, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.mask_fc_fcn_5.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.mask_fc_fcn_5.bias, 0)

        fc_layer = 128
        self.pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.mask_resolution = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        self.fc = nn.Linear(fc_layer * self.pooler_resolution * self.pooler_resolution,
                            self.mask_resolution * self.mask_resolution,
                            bias=True)

        nn.init.kaiming_uniform_(self.fc.weight, a=1)
        nn.init.constant_(self.fc.bias, 0)

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
        feature = torch.tensor(x, requires_grad=False)
        # fcn2, fcn3
        for layer_name in self.common_blocks:
            x = F.relu(getattr(self, layer_name)(x))
        # FCN branch
        x_fcn = F.relu(self.mask_fcn_4(x))
        mask_logits_fcn = self.mask_predictor(x_fcn)

        # FC branch
        batch_size = x.size(0)
        # for layer_name in self.fc_blocks:
        #     x = F.relu(getattr(self, layer_name)(x))
        x_fc = F.relu(self.mask_fc_fcn_4(x))
        x_fc = F.relu(self.mask_fc_fcn_5(x_fc))
        mask_logits_fc = F.relu(self.fc(x_fc.view(batch_size, -1)))\
            .view(-1, 1, self.mask_resolution, self.mask_resolution)

        final_mask_logits = mask_logits_fcn + mask_logits_fc
        return feature, final_mask_logits


def make_PANet_mask_branch(cfg):
    func = PANetMaskBranch
    return func(cfg)