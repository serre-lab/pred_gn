#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import functools
import json
import math
import numpy as np
import os
import random
import torch
import torch.utils.data as data
from PIL import Image

import slowfast.utils.logging as logging

# from . import ava_helper as ava_helper
# from . import cv2_transform as cv2_transform
# from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY
from .simple_motion_generator import generate_sequence, generate_motion_sequence

logger = logging.get_logger(__name__)


# Simplemotion workflow:

# 1- generate a sequences with a configuration
# 2- add sequence to rootpath as an npy file
# 3- specify file name and the split (train/val/test) to load the dataset
# 4- generate examples with "generate_sequence"

@DATASET_REGISTRY.register()
class Simplemotion(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, cfg, split):
        
        self._split = split

        self.root_path = cfg.DATA.PATH_TO_DATA_DIR
        cfg.DATA.NAME

        if split=='train':
            self.data = np.load(os.path.join(self.root_path, cfg.DATA.NAME+'_train.npy'), allow_pickle=True).item()
            self.main_cfg = self.data['cfg']
            self.data = self.data['specs']
        elif split=='val':
            self.data = np.load(os.path.join(self.root_path, cfg.DATA.NAME+'_val.npy'), allow_pickle=True).item()
            self.main_cfg = self.data['cfg']
            self.data = self.data['specs']
        else:
            #raise error
            self.logger.info('no file specified')

        self.cfg = cfg
        
        self.n_frames = cfg.DATA.NUM_FRAMES
        
        
    def print_summary(self):
        logger.info("=== IntPhys dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self.data)))
        total_frames = self.n_frames * len(self.data)
        logger.info("Number of frames: {}".format(total_frames))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        spec = self.data[index]

        if spec['n_frames'] != self.n_frames:
            spec['n_frames'] = self.n_frames
            spec = generate_motion_sequence(spec)
        
        # use function specified in motion_bench to generate data
        imgs = generate_sequence(spec)
        imgs = imgs.transpose(0,1)
        # spec['shapes']['motion']
        return imgs, 0, index, {}
