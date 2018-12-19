# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        num_class=81,
        load_type="Default",

    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.num_class_in_model = num_class
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.load_type = load_type

        self._LOAD_TYPE = {"Default": self._load_model,
                      "ExceptClassificationLayer": self._load_model_except_class_layer,
                      "ExceptMaskBranch": self._load_model_except_mask_branch,
                      "ExceptBoxBranch": self._load_model_except_box_branch,}

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def select_load_type(self):
        func = self._LOAD_TYPE[self.load_type]
        return func

    def load(self, f=None):
        # if self.has_checkpoint():
        #     # override argument with existing checkpoint
        #     f = self.get_checkpoint_file()
        # if not f:
        #     # no checkpoint could be found
        #     self.logger.info("No checkpoint found. Initializing model from scratch")
        #     return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        f = self.select_load_type()
        f(checkpoint)

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        if "model" in checkpoint:
            load_state_dict(self.model, checkpoint.pop("model"))
        else:
            load_state_dict(self.model, checkpoint)

    def _load_model_except_class_layer(self, checkpoint):
        if "model" in checkpoint:
            saved_state_dict = checkpoint['model']
        else:
            saved_state_dict = checkpoint
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            # 'module.roi_heads.mask.predictor.mask_fcn_logits.bias'
            # 'module.roi_heads.box.predictor.cls_score.weight'
            # 'module.roi_heads.box.predictor.bbox_pred.weight'

            i_parts = i.split('.')
            if (i_parts[2] == 'mask' and i_parts[4] == 'mask_fcn_logits') or \
                    (i_parts[2] == 'box' and i_parts[4] == 'cls_score')  or \
                    (i_parts[2] == 'box' and i_parts[4] == 'bbox_pred'):
                continue
            new_params['.'.join(i_parts)] = saved_state_dict[i]
            print('.'.join(i_parts))
        load_state_dict(self.model, new_params)

    def _load_model_except_mask_branch(self, checkpoint):
        if "model" in checkpoint:
            saved_state_dict = checkpoint['model']
        else:
            saved_state_dict = checkpoint
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            # 'module.roi_heads.mask.xx'
            i_parts = i.split('.')
            if i_parts[2] == 'mask':
                continue
            new_params['.'.join(i_parts)] = saved_state_dict[i]
            print('.'.join(i_parts))
        load_state_dict(self.model, new_params)

    def _load_model_except_box_branch(self, checkpoint):
        if "model" in checkpoint:
            saved_state_dict = checkpoint['model']
        else:
            saved_state_dict = checkpoint
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            # 'module.roi_heads.box.xx'
            i_parts = i.split('.')
            if i_parts[2] == 'box':
                continue
            new_params['.'.join(i_parts)] = saved_state_dict[i]
            print('.'.join(i_parts))
        load_state_dict(self.model, new_params)

    def _load_model_part(self, saved_state_dict, name='backbone'):
        for i in saved_state_dict:
            # ''module.backbone.body.stem.conv1.weight'
            i_parts = i.split('.')
            if i_parts[0] == 'module' and i_parts[1] == name:
                self.model.state_dict()['.'.join(i_parts)] = saved_state_dict[i]
                print('.'.join(i_parts))
            if i_parts[0] == name:
                self.model.state_dict()['.'.join(i_parts)] = saved_state_dict[i]
                print('.'.join(i_parts))
        print('Load ' + name +' weight finished.')
        print('-' * 20)


    def _load_model_backbone_and_rpn(self, checkpoint):
        if "model" in checkpoint:
            saved_state_dict = checkpoint['model']
        elif "state_dict" in checkpoint:
            saved_state_dict = checkpoint["state_dict"]
        else:
            saved_state_dict = checkpoint
        self._load_model_part(saved_state_dict, name='backbone')
        self._load_model_part(saved_state_dict, name='rpn')


    def check_shape_of_checkpoints(self, checkpoint):
        if 'model' in checkpoint.keys():
            num_class_in_checkpoints = \
                checkpoint['model']['module.roi_heads.mask.predictor.mask_fcn_logits.weight'].shape[0]
        else:
            return True
        if num_class_in_checkpoints ==  self.num_class_in_model:
            return True
        else:
            return False

class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
