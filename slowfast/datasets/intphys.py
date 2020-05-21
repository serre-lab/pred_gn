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

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader



# Copy baselines for 20 timesteps training 10000
# l1 copy baseline train 0.01076708
# mse copy baseline train 0.0037651432

# Copy baselines for 20 timesteps validation 5000
# l1 copy baseline val 0.003580888
# mse copy baseline val 0.0008071857


@DATASET_REGISTRY.register()
class Intphys(data.Dataset):
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
            self.data = sorted(os.listdir(self.root_path))[:10000]

        elif split=='val':
            self.data = sorted(os.listdir(self.root_path))[10000:]

        elif split=='test':
            self.data = []
            for block in ['O1', 'O2', 'O3']:
                data = sorted(os.listdir(os.path.join(self.root_path, block)))
                data = [os.path.join(block, folder, video) for video in ['1','2','3','4'] for folder in data]
                self.data = self.data + data

        self.cfg = cfg
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.INTPHYS.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.INTPHYS.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.INTPHYS.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.INTPHYS.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.INTPHYS.TEST_FORCE_FLIP
        
        self.sample_rate_in_clip = cfg.DATA.SAMPLING_RATE
        
        self.n_frames = cfg.DATA.NUM_FRAMES
        
        #self.data = sorted(os.listdir(self.root_path))

        self.im_loader = get_default_image_loader()

    def print_summary(self):
        logger.info("=== IntPhys dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self.data)))
        total_frames = 100 * len(self.data)
        logger.info("Number of frames: {}".format(total_frames))
        # logger.info("Number of key frames: {}".format(len(self)))
        # logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root_path, self.data[index])
        
        # frame_indices = list(range(self.data[index]['segment'][0],self.data[index]['segment'][1]+1))
        # frame_indices = self.data[index]['frame_indices']

        frame_indices = list(range(1,100+1))
        
        # if frame_indices[-1]//self.sample_rate_in_clip<=self.n_frames:
        #     i = 1
        # else:
        #     i = np.random.choice(frame_indices[-1]//self.sample_rate_in_clip-self.n_frames)+1
        i = 1 
        
        if self._split == 'train':
        
            frame_indices = list(range(i, min(i+self.n_frames*self.sample_rate_in_clip, frame_indices[-1]), self.sample_rate_in_clip))

            if len(frame_indices) < self.n_frames:
                frame_indices = frame_indices + [frame_indices[-1]]*(self.n_frames - len(frame_indices))
        else:
            frame_indices = list(range(i,101))
        # clip = self.loader(path, frame_indices)
        clip = []
        # logger.info(frame_indices)
        for i in frame_indices:
            im_path = os.path.join(path,'scene/scene_{:03d}.png'.format(i))
            # logger.info(im_path)
            if os.path.exists(im_path):
                clip.append(self.im_loader(im_path))
            else:
                break
        # logger.info(clip)
        # if self.spatial_transform is not None:
        #     self.spatial_transform.randomize_parameters()
        #     clip = [self.spatial_transform(img) for img in clip]
        imgs = torch.as_tensor(np.stack(clip))
        # T H W C -> T C H W.
        imgs = imgs.permute(0, 3, 1, 2)
        # Preprocess images and boxes.
        imgs, _ = self._images_preprocessing(imgs)
        # T C H W -> C T H W.
        imgs = imgs.permute(1, 0, 2, 3)

        # target = self.data[index]['label']-1
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        return imgs, 0, index, {}

    def _images_preprocessing_cv2(self, imgs):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.

        Returns:
            imgs (tensor): list of preprocessed images.
        """

        height, width, _ = imgs[0].shape

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, _ = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale
            )
            imgs, _ = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC"
            )

            # random flip
            imgs, _ = cv2_transform.horizontal_flip_list(
                0.5, imgs, order="HWC"
            )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            
            imgs, _ = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1 #, boxes=boxes
            )

            if self._test_force_flip:
                imgs, _ = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC" # , boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            # boxes = [
            #     cv2_transform.scale_boxes(
            #         self._crop_size, boxes[0], height, width
            #     )
            # ]

            if self._test_force_flip:
                imgs, _ = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC" # , boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        # boxes = cv2_transform.clip_boxes_to_image(
        #     boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        # )
        return imgs, None

    def _images_preprocessing(self, imgs):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        if self._split == "train":
            # Train split
            imgs, _ = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                # boxes=boxes,
            )

            imgs, _ = transform.random_crop(
                imgs, self._crop_size # , boxes=boxes
            )

            # Random flip.
            imgs, _ = transform.horizontal_flip(0.5, imgs) # , boxes=boxes
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, _ = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                # boxes=boxes,
            )

            # Apply center crop for val split
            imgs, _ = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1 #, boxes=boxes
            )

            if self._test_force_flip:
                imgs, _ = transform.horizontal_flip(1, imgs) #, boxes=boxes
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, _ = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                # boxes=boxes,
            )

            if self._test_force_flip:
                imgs, _ = transform.horizontal_flip(1, imgs) #, boxes=boxes
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        return imgs, None

@DATASET_REGISTRY.register()
class Intphys_seg(Intphys):
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
        super(Intphys_seg, self).__init__(cfg, split)
        if split=='test':
            logger.info('WARNING: test split does not have segmentation labels')
        
        self.name_to_class = {
            "floor": 0,
            "sky": 1,
            "walls": 2,
            "occluder": 3,
            "object": 4
        }

        # self.name_to_class = {
        #     "floor": 1,
        #     "sky": 2,
        #     "walls": 3,
        #     "occluder": 4,
        #     "Cone": 5,
        #     "Sphere": 6,
        #     "Cube": 7,
        # }
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root_path, self.data[index])

        frame_indices = list(range(1,100+1))
        
        i = 1 
        
        frame_indices = list(range(i, min(i+self.n_frames*self.sample_rate_in_clip, frame_indices[-1]), self.sample_rate_in_clip))

        if len(frame_indices) < self.n_frames:
            frame_indices = frame_indices + [frame_indices[-1]]*(self.n_frames - len(frame_indices))
        
        clip = []
        masks = []
        for i in frame_indices:
            im_path = os.path.join(path,'scene/scene_{:03d}.png'.format(i))
            mask_path = os.path.join(path,'masks/masks_{:03d}.png'.format(i))

            if os.path.exists(im_path):
                clip.append(self.im_loader(im_path))
                masks.append(self.im_loader(mask_path))
            else:
                # logger.info('problem in video %d'%index)
                break

        gray_masks = np.stack(masks)
        with open(os.path.join(self.root_path, self.data[index], 'status.json')) as f:
            desc = json.load(f)
            desc = desc['frames']
            desc = [desc[i-1]['masks'] for i in frame_indices]
            mask_pixels = {}
            for f_idx in range(len(frame_indices)):
                for l in desc[f_idx]:
                    if not l in mask_pixels:
                        mask_pixels[l] = {}
                    mask_pixels[l][f_idx] = desc[f_idx][l]

        # group masks into array
        # masks_ = np.ones(gray_masks.shape[0:3])
        gray_masks = gray_masks[:,:,:,0]
        for cls_name in mask_pixels:
            # mask_pixels[cls_name]
            
            # get class number that corresponds to the object

            v = self.name_to_class[[k for k in self.name_to_class.keys() if k in cls_name][0]]
            for f_idx in mask_pixels[cls_name].keys():
                gray_masks[f_idx][gray_masks[f_idx] == mask_pixels[cls_name][f_idx]] = v
            
        gray_masks = gray_masks[:,:,:,None]
        # greyscale to class            
        
        masks = torch.as_tensor(gray_masks).float()
        imgs = torch.as_tensor(np.stack(clip))

        # T H W C -> T C H W.
        imgs = imgs.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        
        # logger.info(masks.shape)
        

        # Preprocess images and boxes.
        imgs, masks = self._images_preprocessing(imgs, masks)

        # T C H W -> C T H W.
        imgs = imgs.permute(1, 0, 2, 3)
        masks = masks.permute(1, 0, 2, 3)

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        return imgs, 0, index, {'masks':masks}

    def _images_preprocessing(self, imgs, masks):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        if self._split == "train":
            # Train split
            
            # logger.info('jit')
            imgs, masks = transform.random_short_side_scale_jitter_seg(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                masks = masks
            )
            # logger.info(masks.shape)

            # logger.info('randcrop')
            imgs, masks = transform.random_crop_seg(
                imgs, self._crop_size , masks=masks 
            )
            
            # logger.info(masks.shape)
            # logger.info('flip')
            # Random flip.
            imgs, masks = transform.horizontal_flip_seg(0.5, imgs, masks=masks) 
            # logger.info(masks.shape)


        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, masks = transform.random_short_side_scale_jitter_seg(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                masks = masks
            )

            # Apply center crop for val split
            imgs, masks = transform.uniform_crop_seg(
                imgs, size=self._crop_size, spatial_idx=1, masks=masks
            )
            
            if self._test_force_flip:
                imgs, masks = transform.horizontal_flip_seg(1, imgs, masks=masks)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, masks = transform.random_short_side_scale_jitter_seg_seg(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                masks = masks
            )

            if self._test_force_flip:
                imgs, masks = transform.horizontal_flip_seg(1, imgs, masks=masks)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        logger.info(masks.shape)

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        return imgs, masks

# if __name__=='__main__':

#     # import yaml
    
#     # with open() as f:
#     #     cfg = yaml.load(f)
#     from ..config.defaults import get_cfg
#     cfg = get_cfg()
#     # Load config from cfg.
#     cfg.merge_from_file('../../configs/HMDB/I3D.yaml')

#     data = Hmdb51(cfg, 'train')
