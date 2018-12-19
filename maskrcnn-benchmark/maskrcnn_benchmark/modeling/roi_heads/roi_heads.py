# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from maskrcnn_benchmark.modeling.utils_davis import pad_boxes_on_detections


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing

            # if self.training and self.cfg.TRAIN.USE_GT_BOX:
            if self.training and self.cfg.TRAIN.USE_GT_BOX:
                for target, detection in zip(targets, detections):
                    detection.bbox = target.bbox
                    # detection.remove_field("regression_targets")
                    # detection.extra_fields['objectness'] = torch.ones(len(target.bbox)) * 0.95
                    detection.extra_fields['labels'] = torch.ones(len(target.bbox), dtype=torch.int64)

            if not self.training and self.cfg.TEST.PAD_BOX:
                detections = pad_boxes_on_detections(detections, self.cfg.TEST.PAD_SIZE)

            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return x, detections, losses


class CombinedROIHeads_UseGtBox_Test(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads_UseGtBox_Test, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing

            if not self.training and targets is None:
                raise ValueError("In use gt mode, targets should be passed")
            if self.training:
                raise ValueError("Error! it should be in test mode")
            if not self.training:
                for target, detection in zip(targets, detections):
                        detection.bbox = target.bbox
                        detection.extra_fields['scores'] = torch.ones(len(target.bbox)) * 0.95
                        detection.extra_fields['labels'] = torch.ones(len(target.bbox), dtype=torch.int64)
            x, detections, loss_mask = self.mask(mask_features, detections)
            losses.update(loss_mask)
        return x, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))

    # combine individual heads in a single module
    if roi_heads and cfg.TEST.USE_GT_BOX:
        roi_heads = CombinedROIHeads_UseGtBox_Test(cfg, roi_heads)
        return roi_heads
    elif roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
        return roi_heads
    else:
        raise ValueError("No roi_heads")





