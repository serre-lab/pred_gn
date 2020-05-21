#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import argparse
import sys
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg

from slowfast.datasets import loader

from test_net import test
from train_net import train

sys.stdout.flush()

import numpy as np

import logging
from slowfast.models import build_model


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # # Inherit parameters from args.
    # if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
    #     cfg.NUM_SHARDS = args.num_shards
    #     cfg.SHARD_ID = args.shard_id
    # if hasattr(args, "rng_seed"):
    #     cfg.RNG_SEED = args.rng_seed
    # if hasattr(args, "output_dir"):
    #     cfg.OUTPUT_DIR = args.output_dir

    # # Create the checkpoint dir.
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR)

    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # overrides = sys.argv[1:]
    
    # overrides_dict = {}
    # for i in range(len(overrides)//2):
    #     overrides_dict[overrides[2*i]] = overrides[2*i+1]   
    # overrides_dict['dir'] = cfg.OUTPUT_DIR
    
    # print(overrides_dict)
        
    # cfg.NUM_GPUS=1
    # # Build the video model and print model statistics.
    # model = build_model(cfg)
    # # misc.log_model_info(model, cfg, is_train=True)

    # import torch

    # input_ = torch.randn([2,3,16,96,96]).cuda()
    
    # output = model(input_)

    # print(output['pred_errors'].shape)

    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    l1_baseline = []
    mse_baseline = []
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        err = inputs[0][:,:,1:] - inputs[0][:,:,:-1]
        copy_baseline_l1 = torch.abs(err).mean([1, 3, 4]) #.view([inputs[0].shape[0], -1])
        copy_baseline_mse = torch.pow(err, 2).mean([1, 3, 4])

        l1_baseline.append(copy_baseline_l1.data.numpy())
        mse_baseline.append(copy_baseline_mse.data.numpy())

        if cur_iter%20 == 0:
            print(cur_iter)
            
    l1_baseline = np.concatenate(l1_baseline)
    mse_baseline = np.concatenate(mse_baseline)
    
    print('l1 copy baseline train', l1_baseline.mean())
    print('mse copy baseline train', mse_baseline.mean())
    np.save('copy_baseline_train_20_timesteps.npy', {'copy_mse': mse_baseline, 'copy_l1': l1_baseline})

    l1_baseline = []
    mse_baseline = []
    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        err = inputs[0][:,:,1:] - inputs[0][:,:,:-1]
        copy_baseline_l1 = torch.abs(err).mean([1, 3, 4]) #.view([inputs[0].shape[0], -1])
        copy_baseline_mse = torch.pow(err, 2).mean([1, 3, 4])

        l1_baseline.append(copy_baseline_l1.data.numpy())
        mse_baseline.append(copy_baseline_mse.data.numpy())
        if cur_iter%20 == 0:
            print(cur_iter) 
    
    l1_baseline = np.concatenate(l1_baseline)
    mse_baseline = np.concatenate(mse_baseline)
    
    print('l1 copy baseline val', l1_baseline.mean())
    print('mse copy baseline val', mse_baseline.mean())
    np.save('copy_baseline_val_20_timesteps.npy', {'copy_mse': mse_baseline, 'copy_l1': l1_baseline})


    

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("forkserver")
    # torch.multiprocessing.set_start_method("spawn")
    
    main()
