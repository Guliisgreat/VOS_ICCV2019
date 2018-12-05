import cv2
import logging
import torch
import numpy as np
from torchvision import transforms as T



class LoadYuwenCheckpoint(object):
    def __init__(
        self,
        model,
        logger=None,
        num_class=2
    ):
        self.model = model
        self.num_class_in_model = num_class
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.model_key_list = self.model.state_dict().keys()





    def load(self, f):
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self.saved_state_dict = checkpoint
        self.check_num_weight_matched()
        self._load_all_parts()
        self.check_num_weight_matched()


    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))


    def _load_all_parts(self):
        self._load_resnet_block()
        self._load_fpn_block()
        self._load_rpn_block()
        self._load_rcnn_block()
        self._load_mask_head_block()


    def _load_resnet_block(self):
        # stem <-> conv1
        for i in self.saved_state_dict:
            i_parts = i.split('.')
            if i_parts[0] == 'resnet_backbone' and i_parts[1] == 'conv1':
                # ''module.backbone.body.stem.conv1.weight'
                #  'resnet_backbone.conv1.conv1.weight'
                name_list = ['backbone', 'body', 'stem', i_parts[2], i_parts[3]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'resnet_backbone' and i_parts[2] == 'layers' and 'downsample' not in i_parts:
                #  backbone.body.layer1.0.conv1.weight
                # resX.layer.Y  --> layer(X-1).Y
                x = int(i_parts[1][-1])
                y = i_parts[3]
                name_list = ["backbone.body","layer"+str(x-1), y, i_parts[4], i_parts[5]]
                assert self.check_keys_exist('.'.join(name_list)) == True, '{}: key not exist'.format('.'.join(name_list))
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'resnet_backbone' and i_parts[2] == 'layers' and 'downsample' in i_parts:
                # esnet_backbone.res2.layers.0.downsample.0.weight = backbone.body.layer1.0.downsample.0.weight
                x = int(i_parts[1][-1])
                y = i_parts[3]
                name_list = ["backbone.body","layer"+str(x-1), y, i_parts[4], i_parts[5], i_parts[6]]
                assert self.check_keys_exist('.'.join(name_list)) == True, '{}: key not exist'.format(
                    '.'.join(name_list))
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))

    def _load_fpn_block(self):
        for i in self.saved_state_dict:
            i_parts = i.split('.')
            if i_parts[0] == 'fpn' and len(i_parts[1])>8 :
                # fpn.fpn_p2_1x1.weight = backbone.fpn.fpn_inner1.weight
                x = int(i_parts[1].split('_')[1][-1])
                name_list = ["backbone",  "fpn", "fpn_inner"+str(x-1), i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, '{}: key not exist'.format(
                    '.'.join(name_list))
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'fpn' and len(i_parts[1])< 8:
                # fpn.fpn_p2.weight = backbone.fpn.fpn_layer1.weight
                x = int(i_parts[1].split('_')[1][-1])
                name_list = ["backbone", "fpn", "fpn_layer" + str(x - 1), i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))


    def _load_rpn_block(self):
        for i in self.saved_state_dict:
            i_parts = i.split('.')
            if i_parts[0] == 'rpn'  and i_parts[1]=='conv_proposal':
                # rpn.conv_proposal.0.weight = rpn.head.conv.weight
                name_list = ["rpn", "head", "conv" , i_parts[3]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'rpn' and i_parts[1] == 'cls_score':
                # rpn.cls_score.weight = rpn.head.cls_logits.weight
                name_list = ["rpn", "head", "cls_logits", i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'rpn' and i_parts[1] == 'bbox_pred':
                # rpn.bbox_pred.weight = rpn.head.bbox_pred.weight
                name_list = ["rpn", "head", "bbox_pred", i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))


    def _load_rcnn_block(self):
        for i in self.saved_state_dict:
            i_parts = i.split('.')
            if i_parts[0] == 'rcnn'  and i_parts[1][0]=='f':
                # rcnn.fc6.0.weight = roi_heads.box.feature_extractor.fc6.weight
                name_list = ["roi_heads", "box", "feature_extractor", i_parts[1], i_parts[3]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'rcnn' and i_parts[1] == 'cls_score':
                #  rcnn.cls_score.weight = roi_heads.box.predictor.cls_score.weigh
                name_list = ["roi_heads", "box", "predictor", i_parts[1], i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'rcnn' and i_parts[1] == 'bbox_pred':
                # rcnn.bbox_pred.weight = roi_heads.box.predictor.bbox_pred.weight
                name_list = ["roi_heads", "box", "predictor", i_parts[1], i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))


    def _load_mask_head_block(self):
        for i in self.saved_state_dict:
            i_parts = i.split('.')
            if i_parts[0] == 'mask_branch' and i_parts[1][:-1] == 'mask_conv':
                x = i_parts[1][-1]
                name_list = ['roi_heads', 'mask', 'feature_extractor', 'mask_fcn'+x, i_parts[3]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'mask_branch' and i_parts[1] == 'mask_deconv1':
                name_list = ['roi_heads', 'mask', 'predictor', 'conv5_mask', i_parts[3]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))
            if i_parts[0] == 'mask_branch' and i_parts[1] == 'mask_score':
                name_list = ['roi_heads', 'mask', 'predictor', 'mask_fcn_logits', i_parts[2]]
                assert self.check_keys_exist('.'.join(name_list)) == True, 'key not exist'
                self.model.state_dict()['.'.join(name_list)] = self.saved_state_dict[i]
                self.print_keys(i, '.'.join(name_list))

    def print_keys(self, a, b):
        print("-" * 20)
        print(a)
        print(b)


    def check_num_weight_matched(self):
        num_checkpoints = len(self.saved_state_dict)
        num_model = 0
        for i in self.model.state_dict().keys():
            i_parts = i.split('.')
            if i_parts[0] == 'rpn' and i_parts[1] == 'anchor_generator':
                continue
            else:
                num_model += 1
        assert num_checkpoints == num_model, \
            "the number of weights in the checkpoint != the number of weights in the model"

    def check_keys_exist(self, key):
        if key in self.model_key_list:
            return True
        else:
            return False













