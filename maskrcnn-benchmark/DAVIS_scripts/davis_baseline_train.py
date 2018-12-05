
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import logging
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import Checkpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.logging import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from davis_components.train import LoadYuwenCheckpoint
from utils_davis import tools


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    logger = logging.getLogger("Training")
    with tools.TimerBlock("Loading Experimental setups", logger) as block:
        exp_name = cfg.EXP.NAME
        output_dir = tools.get_exp_output_dir(exp_name, cfg.OUTPUT_DIR)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        validation_period = cfg.SOLVER.VALIDATION_PERIOD

    with tools.TimerBlock("Loading Checkpoints...", logger) as block:
        arguments = {}
        save_to_disk = local_rank == 0
        checkpointer = Checkpointer(model,
                                    save_dir=output_dir,
                                    save_to_disk=save_to_disk,
                                    num_class=cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                                    finetune_class_layer=False)
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)
        arguments["iteration"] = 0

    with tools.TimerBlock("Initializing DAVIS Datasets", logger) as block:
        logger.info("Loading training set...")
        data_loader_train = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        logger.info("Loading valid set...")
        data_loaders_valid = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed,
        )

    do_train(
        model,
        data_loader_train,
        data_loaders_valid[0],
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        validation_period,
        arguments,
        exp_name,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    exp_name = cfg.EXP.NAME
    output_dir = tools.get_exp_output_dir(exp_name, cfg.OUTPUT_DIR)

    logger = setup_logger("Training", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    logger.info("Training is Finished")


if __name__ == "__main__":
    main()

