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


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    segments = []
    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
            segments.append(value['segment'])

    return video_names, annotations, segments


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video=1,
                 sample_duration=None):
    data = load_annotation_data(annotation_path)
    video_names, annotations, segments = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        
        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue
        # frames = os.listdir(video_path)
        # if not frames:
        #     continue
        # n_frames_file_path = os.path.join(video_path, 'n_frames')
        # n_frames = int(load_value_file(n_frames_file_path))
        # if n_frames <= 0:
        #     continue

        
        sample = {
            'video': video_path,
            'segment': segments[i],
            'n_frames': segments[i][-1],
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        # sample['frame_indices'] = list(range(1, len(frames) + 1))
        dataset.append(sample)

    return dataset, idx_to_class




@DATASET_REGISTRY.register()
class Hmdb51(data.Dataset):
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
        # if split=='train':
        #     subset = 'training'
        # elif split=='val':
        #     subset = 'validation'

        self.cfg = cfg
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.HMDB.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.HMDB.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.HMDB.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.HMDB.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.HMDB.TEST_FORCE_FLIP


        self.root_path = cfg.DATA.PATH_TO_DATA_DIR
        self.annotation_path = cfg.HMDB.ANNOTATION_PATH
        self.sample_duration = 0 #cfg.SAMPLE_DURATION
        self.n_samples_for_each_video = 0#cfg.N_SAMPLES
        self.sample_rate = cfg.HMDB.OUT_SAMPLE_RATE
        
        self.n_frames = cfg.DATA.NUM_FRAMES
        assert self.sample_rate<=30 and 30%self.sample_rate==0, 'wrong sample rate value'
        
        self.sample_rate_in_clip = 30//self.sample_rate

        self.data, self.class_names = make_dataset(
            self.root_path, self.annotation_path, self._split, self.n_samples_for_each_video,
            self.sample_duration)

        # self.spatial_transform = spatial_transform
        # self.temporal_transform = temporal_transform
        # self.target_transform = target_transform
        self.loader = get_default_video_loader()

    def print_summary(self):
        logger.info("=== HMDB dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self.data)))
        total_frames = sum(
            video_img_paths['segment'][-1] for video_img_paths in self.data
        )
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
        path = self.data[index]['video']

        frame_indices = list(range(self.data[index]['segment'][0],self.data[index]['segment'][1]+1))
        # frame_indices = self.data[index]['frame_indices']
        
        if frame_indices[-1]//self.sample_rate_in_clip<=self.n_frames:
            i = 1
        else:
            i = np.random.choice(frame_indices[-1]//self.sample_rate_in_clip-self.n_frames)+1

        frame_indices = list(range(i, min(i+self.n_frames*self.sample_rate_in_clip, frame_indices[-1]), self.sample_rate_in_clip))

        if len(frame_indices) < self.n_frames:
            frame_indices = frame_indices + [frame_indices[-1]]*(self.n_frames - len(frame_indices))
        
        clip = self.loader(path, frame_indices)
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

        target = self.data[index]['label']-1
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        return imgs, target, index, {}

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

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        # boxes[:, [0, 2]] *= width
        # boxes[:, [1, 3]] *= height
        # boxes = transform.clip_boxes_to_image(boxes, height, width)

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

        # if not self._use_bgr:
        #     # Convert image format from BGR to RGB.
        #     # Note that Kinetics pre-training uses RGB!
        #     imgs = imgs[:, [2, 1, 0], ...]

        # boxes = transform.clip_boxes_to_image(
        #     boxes, self._crop_size, self._crop_size
        # )

        return imgs, None

# if __name__=='__main__':

#     # import yaml
    
#     # with open() as f:
#     #     cfg = yaml.load(f)
#     from ..config.defaults import get_cfg
#     cfg = get_cfg()
#     # Load config from cfg.
#     cfg.merge_from_file('../../configs/HMDB/I3D.yaml')

#     data = Hmdb51(cfg, 'train')
