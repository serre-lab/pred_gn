#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import os
import pprint
import sys
import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.utils.viz_helpers as viz_helpers
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter

import neptune
import torchvision as tv
from neptune.sessions import Session

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, writer, nep, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    global_iters = data_size*cur_epoch
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        inputs = inputs.cuda(non_blocking=True)
        # labels = torch.repeat_interleave(labels,inputs[i].size(1),0)

        # for i in range(len(inputs)):
        #     if len(inputs[i].shape) > 5:
        #         inputs[i] = inputs[i].view((-1,)+inputs[i].shape[2:])

        # labels = labels.cuda()
        # for key, val in meta.items():
        #     if isinstance(val, (list,)):
        #         for i in range(len(val)):
        #             val[i] = val[i].cuda(non_blocking=True)
        #     else:
        #         meta[key] = val.cuda(non_blocking=True)


        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, global_iters, cfg)
        optim.set_lr(optimizer, lr)

        preds = model(inputs)

        out_keys = list(preds.keys())

        total_loss = preds['total_loss']

        # check Nan Loss.
        misc.check_nan_losses(total_loss)

        # Perform the backward pass.
        optimizer.zero_grad()

        total_loss.backward()

        ####################################################################################################################################
        # check gradients
        # if writer is not None and global_iters%cfg.SUMMARY_PERIOD==0:
        #     n_p = model.module.named_parameters() if hasattr(model,'module') else model.named_parameters()
        #     fig = viz_helpers.plot_grad_flow_v2(n_p)
        #     writer.add_figure('grad_flow/grad_flow', fig, global_iters)
        ####################################################################################################################################

        # Update the parameters.
        optimizer.step()
        
        losses = [preds[k] for k in out_keys]

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:    
            losses = du.all_reduce(losses)
        
        losses = [l.item() for l in losses]

        loss_logs = {}
        for i in range(len(losses)):
            loss_logs[out_keys[i]] = losses[i]
        
        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            lr, inputs[0].size(0) * cfg.NUM_GPUS, **loss_logs
        )

        if writer is not None and global_iters%cfg.LOG_PERIOD==0:
            logger.info(model.conv0[2].weight)
            logger.info(model.conv0[2].bias)
            for k,v in loss_logs.items():
                writer.add_scalar('loss/'+k.strip('loss_'), train_meter.stats[k].get_win_median(), global_iters)
        if nep is not None and global_iters%cfg.LOG_PERIOD==0:
            for k,v in loss_logs.items():
                nep.log_metric(k.strip('loss_'), train_meter.stats[k].get_win_median())

            nep.log_metric('global_iters', global_iters)


        if global_iters%cfg.SUMMARY_PERIOD==0 and du.get_rank()==0 and du.is_master_proc(num_gpus=cfg.NUM_GPUS):

            with torch.no_grad():
                # logger.info(inputs[i].shape)
                # sys.stdout.flush()
                inputs = inputs[:min(3,len(inputs))]
                if 'masks' in meta:
                    frames = model((inputs, meta['masks'][:min(3,len(inputs))]), extra=['frames'])['frames']
                else:
                    frames = model(inputs, extra=['frames'])['frames']

                n_rows = inputs.size(2)-1

                inputs = inputs.transpose(1,2)[:, -n_rows:]
                frames = frames.transpose(1,2)[:, -n_rows:]
                frames = torch.cat([(frames!=inputs)*frames, (frames==inputs)*inputs, torch.zeros_like(frames)], 2)
                inputs = torch.cat([inputs]*3, 2)
                # inputs = inputs*inputs.new(cfg.DATA.STD)[None,None,:,None,None]+inputs.new(cfg.DATA.MEAN)[None,None,:,None,None]
                # frames = frames*frames.new(cfg.DATA.STD)[None,None,:,None,None]+frames.new(cfg.DATA.MEAN)[None,None,:,None,None]
                images = torch.cat([inputs, frames], 1).reshape((-1,) + inputs.shape[2:])

            # grid = tv.utils.make_grid(images, nrow=8, normalize=True)
            # writer.add_image('predictions', images, global_iters)

            tv.utils.save_image(images, os.path.join(cfg.OUTPUT_DIR, 'preds_%d.jpg'%global_iters), nrow=n_rows, normalize=True)

            # del images
            # del frames
            # del inputs

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        global_iters+=1

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, nep, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        # labels = labels.cuda()
        # for key, val in meta.items():
        #     if isinstance(val, (list,)):
        #         for i in range(len(val)):
        #             val[i] = val[i].cuda(non_blocking=True)
        #     else:
        #         meta[key] = val.cuda(non_blocking=True)


        preds = model(inputs)
        
        out_keys = list(preds.keys())
        losses = [preds[k] for k in out_keys]

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:    
            losses = du.all_reduce(losses)
        
        losses = [l.item() for l in losses]

        # Copy the errors from GPU to CPU (sync point).
        loss_logs = {}
        for i in range(len(losses)):
            loss_logs[out_keys[i]] = losses[i]

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            inputs[0].size(0) * cfg.NUM_GPUS,  **loss_logs
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # neptune update
    if nep is not None:
        for k,v in loss_logs.items():
            nep.log_metric('val_' + k.strip('loss_'), val_meter.stats[k].get_global_avg())


    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """


    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    if du.get_rank()==0 and du.is_master_proc(num_gpus=cfg.NUM_GPUS):
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    else:
        writer = None

    if du.get_rank()==0 and du.is_master_proc(num_gpus=cfg.NUM_GPUS) and not cfg.DEBUG:
        tags = []
        if 'TAGS' in cfg and cfg.TAGS !=[]:
            tags=list(cfg.TAGS)
        neptune.set_project('Serre-Lab/motion')

        ######################
        overrides = sys.argv[1:]

        overrides_dict = {}
        for i in range(len(overrides)//2):
            overrides_dict[overrides[2*i]] = overrides[2*i+1]
        overrides_dict['dir'] = cfg.OUTPUT_DIR
        ######################


        if 'NEP_ID' in cfg and cfg.NEP_ID != "":
            session = Session()
            project = session.get_project(project_qualified_name='Serre-Lab/motion')
            nep_experiment = project.get_experiments(id=cfg.NEP_ID)[0]

        else:
            nep_experiment = neptune.create_experiment (name=cfg.NAME,
                                        params=overrides_dict,
                                        tags=tags)
    else:
        nep_experiment=None

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc(num_gpus=cfg.NUM_GPUS):
        misc.log_model_info(model, cfg, is_train=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, writer, nep_experiment, cfg)

        # Compute precise BN stats.
        # if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
        #     calculate_and_update_precise_bn(
        #         train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
        #     )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, nep_experiment, cfg)

        if du.get_rank()==0 and du.is_master_proc(num_gpus=cfg.NUM_GPUS) and not cfg.DEBUG:
            nep_experiment.log_metric('epoch', cur_epoch)
