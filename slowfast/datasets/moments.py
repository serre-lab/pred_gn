#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)



@DATASET_REGISTRY.register()
class Moments(torch.utils.data.Dataset):
    """
    Moments video loader. Construct the Moments video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Moments video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Moments".format(mode)
        self.mode = mode
        self.cfg = cfg

        

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Moments {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        data_file = 'val' if self.mode == 'test' else self.mode
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(data_file)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        
        self.num_repeated_samples = self.cfg.DATA.NUM_REPEATED_SAMPLES
        self._use_color_augmentation = self.cfg.DATA.COLOR_AUGMENTATION
        
        self._classes = []
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'moments_categories.txt')) as f:
            self._classes = f.readlines()
        self._classes = [c.split(',')[0] for c in self._classes]
        
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.readlines()):
                
                # if clip_idx < 640:
                
                # assert len(path_label.split()) == 4
                path, label = path_label.split(',')[:2]
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, data_file, path)
                    )
                    self._labels.append(self._classes.index(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Moments split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing Moments dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            # temporal_sample_index = (
            #     self._spatial_temporal_idx[index]
            #     // self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            # spatial_sample_index = (
            #     self._spatial_temporal_idx[index]
            #     % self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                self.cfg.DATA.SAMPLING_RATE,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=30,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            
            if self.mode == "train" and self.num_repeated_samples>1:
                out_frames = []
                for i in range(self.num_repeated_samples):
                    sample_frames = frames.clone()

                    if self._use_color_augmentation:
                        # T H W C -> T C H W. for color augmentation
                        sample_frames = sample_frames.permute(0, 3, 1, 2)
                        
                        # sample_frames = transform.color_jitter(
                        sample_frames = transform.random_color_augmentation(
                            sample_frames,
                            img_brightness=0.4,
                            img_contrast=0.4,
                            img_saturation=0.4,
                            img_illumination=0.2
                        )
                        
                        # T C H W -> C T H W.
                        sample_frames = sample_frames.permute(1, 0, 2, 3)

                    else:
                        # T H W C -> C T H W.
                        sample_frames = sample_frames.permute(3, 0, 1, 2)
                    
                    sample_frames = sample_frames - torch.tensor(self.cfg.DATA.MEAN)[:,None,None,None]
                    sample_frames = sample_frames / torch.tensor(self.cfg.DATA.STD)[:,None,None,None]
                    
                    # Perform data augmentation.
                    sample_frames = self.spatial_sampling(
                        sample_frames,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                    )

                    # out_frames.append(sample_frames)
                    out_frames.append(utils.pack_pathway_output(self.cfg, sample_frames))

                # label = torch.Tensor([self._labels[index]]*self.num_repeated_samples)
                label = self._labels[index]
                frames = [torch.stack([o[i] for o in out_frames],0)  for i in range(len(out_frames[0]))]
                
                # frames = utils.pack_pathway_output(self.cfg, out_frames)
                return frames, label, index, {}
            
            else:
                frames = frames - torch.tensor(self.cfg.DATA.MEAN)
                frames = frames / torch.tensor(self.cfg.DATA.STD)
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = self.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )

                label = self._labels[index]
                
                frames = utils.pack_pathway_output(self.cfg, frames)
                return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def get_example_by_class(self, class_idx):
        class_idx = self._label[self._label==class_idx]
        idx = np.random.choice(len(class_idx))
        
        return self.__getitem__(idx) 
    
    ##############################################################################################
    ##############################################################################################

    def get_augmented_examples(self, index):
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            # temporal_sample_index = (
            #     self._spatial_temporal_idx[index]
            #     // self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            # spatial_sample_index = (
            #     self._spatial_temporal_idx[index]
            #     % self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                self.cfg.DATA.SAMPLING_RATE,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=30,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            
            if self.mode == "train" and self.num_repeated_samples>1:
                out_frames = []
                for i in range(self.num_repeated_samples):
                    sample_frames = frames.clone()

                    if self._use_color_augmentation:
                        # T H W C -> T C H W. for color augmentation
                        sample_frames = sample_frames.permute(0, 3, 1, 2)
                        
                        # sample_frames = transform.color_jitter(
                        sample_frames = transform.random_color_augmentation(
                            sample_frames,
                            img_brightness=0.4,
                            img_contrast=0.4,
                            img_saturation=0.4,
                            img_illumination=0.4
                        )

                        # T C H W -> C T H W.
                        sample_frames = sample_frames.permute(1, 0, 2, 3)

                    else:
                        # T H W C -> C T H W.
                        sample_frames = sample_frames.permute(3, 0, 1, 2)
                    
                    sample_frames = sample_frames - torch.tensor(self.cfg.DATA.MEAN)[:,None,None,None]
                    sample_frames = sample_frames / torch.tensor(self.cfg.DATA.STD)[:,None,None,None]
                    
                    # Perform data augmentation.
                    sample_frames = self.spatial_sampling(
                        sample_frames,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                    )
                    # out_frames.append(sample_frames)
                    out_frames.append(utils.pack_pathway_output(self.cfg, sample_frames))

                # label = torch.Tensor([self._labels[index]]*self.num_repeated_samples)
                label = self._labels[index]
                frames = [torch.stack([o[i] for o in out_frames],0)  for i in range(len(out_frames[0]))]
                
                # frames = utils.pack_pathway_output(self.cfg, out_frames)
                return frames, label, index, {}
            
            else:
                frames = frames - torch.tensor(self.cfg.DATA.MEAN)
                frames = frames / torch.tensor(self.cfg.DATA.STD)
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = self.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )

                label = self._labels[index]
                
                frames = utils.pack_pathway_output(self.cfg, frames)
                return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            
            frames = transform.random_spatial_augmentation(frames, rotate=40, shear=0.3)
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
