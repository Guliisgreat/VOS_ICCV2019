# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from tqdm import tqdm

import torch
from torch.distributed import deprecated as dist


from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.tensorboard_logger import TensorboardXLogger

from maskrcnn_benchmark.config import cfg
import os


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader_train,
    data_loaders_valid,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    validation_period,
    arguments,
    exp_name,
):
    logger = logging.getLogger("Training")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    tensorboard_path = os.path.join('../output/tensorboard', exp_name)
    tensorboard_logger = TensorboardXLogger(log_dir=tensorboard_path)

    max_iter = len(data_loader_train)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader_train, start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        meters.update(lr=optimizer.param_groups[0]["lr"])

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        tensorboard_logger.write(meters, iteration, phase='Train')

        if iteration % (validation_period / 10) == 0 or iteration == (max_iter - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % validation_period == 0 and iteration > 0:
            validation(model, data_loaders_valid, device, logger, tensorboard_logger, iteration)

        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    tensorboard_logger.export_to_json()


def validation(
    model,
    data_loader,
    device,
    logger,
    tensorboard_logger,
    iteration
):
    logger.info('-' * 40)
    logger.info("Start Validation")
    meters = MetricLogger(delimiter="  ")
    start_validation_time = time.time()

    max_iter = len(data_loader)


    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, _ = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(total_loss=losses_reduced, **loss_dict_reduced)


    tensorboard_logger.write(meters, iteration, phase='Valid')
    logger.info('Validation:')
    logger.info(
                meters.delimiter.join(
                    [
                        "iter: {iter}",
                        "{meters}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

    total_validation_time = time.time() - start_validation_time
    total_time_str = str(datetime.timedelta(seconds=total_validation_time))
    logger.info(
        "Total Validation time: {} ({:.4f} s / it)".format(
            total_time_str, total_validation_time / (max_iter)
        )
    )
    logger.info('-' * 40)
