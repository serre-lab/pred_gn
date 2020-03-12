
import argparse
import sys
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg

import numpy as np
import torch

from slowfast.datasets import loader
from slowfast.datasets.build import build_dataset
import slowfast.utils.logging as logging

from PIL import Image

import os

logger = logging.get_logger(__name__)

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

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def visualize(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    # logger.info(pprint.pformat(cfg))

    # # Build the video model and print model statistics.
    # model = build_model(cfg)
    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=True)

    # Create the video train and val loaders.
    # train_loader = loader.construct_loader(cfg, "train")
    # val_loader = loader.construct_loader(cfg, "val")

    train_set = build_dataset(cfg.TEST.DATASET, cfg, "train")

    for i in range(5):
        frames, label, _, _ = train_set.get_augmented_examples(i)
        frames = frames[0].permute(0,2,3,4,1)
        logger.info('### Z score ##########')
        logger.info('min')
        logger.info(frames.min())
        logger.info('max')
        logger.info(frames.max())
        logger.info('mean')
        logger.info(frames.mean())
        logger.info('var')
        logger.info(frames.var())
        
        frames = frames * torch.tensor(cfg.DATA.STD)#[None,:,None,None,None]
        frames = frames + torch.tensor(cfg.DATA.MEAN)#[None,:,None,None,None]

        logger.info('### normal ##########')
        logger.info('min')
        logger.info(frames.min())
        logger.info('max')
        logger.info(frames.max())
        logger.info('mean')
        logger.info(frames.mean())
        logger.info('var')
        logger.info(frames.var())
        
        for a in range(frames.size(0)):
            for s in range(frames.size(1)):
                
                im = Image.fromarray((frames[a,s].data.numpy()*255).astype(np.uint8))
                im.save(os.path.join(cfg.OUTPUT_DIR,'example_%d_aug%d_frame_%d.png'%(i,a,s)))



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    visualize(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    # torch.multiprocessing.set_start_method("spawn")
    
    main()