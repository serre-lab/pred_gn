
import argparse
import sys
import torch

import torchvision as tv

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg

import numpy as np
import torch

from slowfast.datasets import loader
from slowfast.datasets.build import build_dataset
import slowfast.utils.logging as logging

from slowfast.models import build_model

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
    logging.setup_logging(cfg)

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

    for i in np.random.choice(len(train_set), 5):
        
        # frames, label, _, _ = train_set.get_augmented_examples(i)
        frames, label, _, _ = train_set[i]
        logger.info(frames[0].shape)
        # frames = frames[0].permute(0,2,3,4,1)
        frames = frames[0].transpose(0,1)#.permute(1,2,3,0)
        logger.info('### Z score ##########')
        logger.info('min')
        logger.info(frames.min())
        logger.info('max')
        logger.info(frames.max())
        logger.info('mean')
        logger.info(frames.mean())
        logger.info('var')
        logger.info(frames.var())
        
        frames = frames * torch.tensor(cfg.DATA.STD)[None,:,None,None] #[None,:,None,None,None]
        frames = frames + torch.tensor(cfg.DATA.MEAN)[None,:,None,None] #[None,:,None,None,None]

        logger.info('### normal ##########')
        logger.info('min')
        logger.info(frames.min())
        logger.info('max')
        logger.info(frames.max())
        logger.info('mean')
        logger.info(frames.mean())
        logger.info('var')
        logger.info(frames.var())
        
        tv.utils.save_image(frames, os.path.join(cfg.OUTPUT_DIR, 'example_%d.jpg'%i), nrow=18, normalize=True)
        # for a in range(frames.size(0)):
        #     for s in range(frames.size(1)):
                
        #         im = Image.fromarray((frames[a,s].data.numpy()*255).astype(np.uint8))
        #         im.save(os.path.join(cfg.OUTPUT_DIR,'example_%d_aug%d_frame_%d.png'%(i,a,s)))


def visualize_activations(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Setup logging format.
    logging.setup_logging(cfg)

    # Print config.
    logger.info("Vizualize activations")
    # logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # Construct the optimizer.
    # optimizer = optim.construct_optimizer(model, cfg)

    logger.info("Load from given checkpoint file.")
    checkpoint_epoch = cu.load_checkpoint(
        cfg.TRAIN.CHECKPOINT_FILE_PATH,
        model,
        cfg.NUM_GPUS > 1,
        optimizer=None,
        inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
        convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
    )

    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=True)

    # Create the video train and val loaders.
    # train_loader = loader.construct_loader(cfg, "train")
    # val_loader = loader.construct_loader(cfg, "val")

    train_set = build_dataset(cfg.TEST.DATASET, cfg, "train")

    

    for i in np.random.choice(len(train_set), 5):
        
        # frames, label, _, _ = train_set.get_augmented_examples(i)
        frames, label, _, _ = train_set[i]
        inputs = frames
        inputs[0] = inputs[0][None,:]
        logger.info(frames[0].shape)
        # frames = frames[0].permute(0,2,3,4,1)
        frames = frames[0].squeeze().transpose(0,1)#.permute(1,2,3,0)
        logger.info(frames.shape)
        tv.utils.save_image(frames, os.path.join(cfg.OUTPUT_DIR, 'example_%d.jpg'%i), nrow=18, normalize=True)
        
        for j in range(len(inputs)):
            inputs[j] = inputs[j].cuda(non_blocking=True)
        with torch.no_grad():
            # logger.info(inputs[i].shape)
            # sys.stdout.flush()
            inputs[0] = inputs[0][:min(3,len(inputs[0]))] 
            output = model(inputs, extra=['frames'])
            
            # frames = frames[0].transpose(0,1)#.permute(1,2,3,0)
            # tv.utils.save_image(frames, os.path.join(cfg.OUTPUT_DIR, 'example_target_%d.jpg'%i), nrow=18, normalize=True)
            
            input_aug = output['input_aug']
            logger.info(input_aug.shape)

            input_aug = input_aug[0].transpose(0,1)
            tv.utils.save_image(input_aug, os.path.join(cfg.OUTPUT_DIR, 'example_input_%d.jpg'%i), nrow=18, normalize=True)
            

            # mix_layer [1, timesteps, layers, activations]
            mix_out = output['mix_layer']#.cpu().data.numpy().squeeze()
            for layer in range(len(mix_out)): 
                logger.info('mix layer %d'%layer)
                logger.info(mix_out[layer].view([18,-1]).mean(1))
                images = mix_out[layer].transpose(1,2).transpose(0,1)
                logger.info(images.shape)
                images = images.reshape((-1,) + images.shape[2:])
                images = (images-images.min())
                images = images/images.max()
                tv.utils.save_image(images, os.path.join(cfg.OUTPUT_DIR, 'example_%d_mix_layer_l%d.jpg'%(i,layer)), nrow=18, normalize=True)


            # BU errors per timestep per layer (choose a random activation or the mean) also write out the mean/norm
            # [1, timesteps, layers, channels, height, width]
            
            bu_errors = output['bu_errors']#.cpu()#.data.numpy().squeeze()
            
            for layer in range(len(bu_errors)): 
                images = bu_errors[layer].transpose(1,2).transpose(0,1)
                images = (images-images.min())
                images = images/images.max()
                logger.info(images.shape)
                images = images.reshape((-1,) + images.shape[2:])
                tv.utils.save_image(images, os.path.join(cfg.OUTPUT_DIR, 'example_%d_bu_errors_l%d.jpg'%(i,layer)), nrow=18, normalize=True)


            # horiz inhibition per timestep per layer (choose a random activation or the mean) also write out the mean/norm
            # [1, timesteps, layers, channels, height, width]
            inhibition = output['H_inh']#.cpu()#.data.numpy().squeeze()
            for layer in range(len(inhibition)): 
                images = inhibition[layer].transpose(1,2).transpose(0,1)
                images = (images-images.min())
                images = images/images.max()
                logger.info(images.shape)
                images = images.reshape((-1,) + images.shape[2:])

                tv.utils.save_image(images, os.path.join(cfg.OUTPUT_DIR, 'example_%d_H_inh_l%d.jpg'%(i,layer)), nrow=18, normalize=True)

            # persistent state in between timesteps
            # [1, timesteps, layers, channels, height, width]
            hidden = output['hidden']#.cpu()#.data.numpy().squeeze()
            for layer in range(len(hidden)): 
                images = hidden[layer].transpose(1,2).transpose(0,1)
                images = (images-images.min())
                images = images/images.max()
                logger.info(images.shape)
                images = images.reshape((-1,) + images.shape[2:])

                tv.utils.save_image(images, os.path.join(cfg.OUTPUT_DIR, 'example_%d_hidden_l%d.jpg'%(i,layer)), nrow=18, normalize=True)


            # inputs = inputs[0].transpose(1,2)[:, -n_rows:]
            # frames = frames.transpose(1,2)[:, -n_rows:]

            # images = torch.cat([inputs, frames], 1).reshape((-1,) + inputs.shape[2:]) 
        
        # grid = tv.utils.make_grid(images, nrow=8, normalize=True)
        # writer.add_image('predictions', images, global_iters)
        
        
        # for a in range(frames.size(0)):
        #     for s in range(frames.size(1)):
                
        #         im = Image.fromarray((frames[a,s].data.numpy()*255).astype(np.uint8))
        #         im.save(os.path.join(cfg.OUTPUT_DIR,'example_%d_aug%d_frame_%d.png'%(i,a,s)))


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    visualize(cfg=cfg)
    # visualize_activations(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    # torch.multiprocessing.set_start_method("spawn")
    
    main()