#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

import os
import random
import torch
import torch.utils.data
import numpy as np

import slowfast.utils.logging as logging

# from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Mmnist(data.Dataset):
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
        
        if split=='train':
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self.data = os.listdir(os.path.join(self.root_path, 'train'))
            self.n_data_frames = 30
            # self.data = np.load(os.path.join(self.root_path, 'moving_mnist.npy'))
        elif split=='val':
            self._split = 'train'
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self.n_data_frames = 30
            self.data = os.listdir(os.path.join(self.root_path, 'train')) #np.load(os.path.join(self.root_path, 'mnist_test_seq.npy'))

        self.cfg = cfg
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        
        self.n_frames = cfg.DATA.NUM_FRAMES
        
    def print_summary(self):
        logger.info("=== MMnist dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(self.data.shape[0]))
        total_frames = self.data.shape[0]*self.data.shape[1]
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
        
        
        # frame_indices = list(range(1,100+1))
        if self.n_frames < self.n_data_frames:
            start_frame = np.random.choice(self.n_data_frames - self.n_frames)
        else:
            start_frame = 0
        
        # imgs = self.data[index, start_frame:start_frame+self.n_frames]
        imgs = np.load(os.path.join(self.root_path, self._split, self.data[index]))[start_frame:start_frame+self.n_frames]
        
        imgs = torch.as_tensor(imgs).float()
        
        if self._crop_size != imgs.size(-1):
            imgs = torch.nn.functional.interpolate(
                imgs,
                size=(self._crop_size, self._crop_size),
                mode="bilinear",
                align_corners=False,
            )
        imgs = imgs.transpose(0,1)

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        
        return imgs, 0, index, {}
