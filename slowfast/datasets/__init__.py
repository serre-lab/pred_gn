#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .kinetics import Kinetics  # noqa
from .hmdb51 import Hmdb51  # noqa
from .moments import Moments
from .intphys import Intphys, Intphys_seg
from .mmnist import Mmnist
from .simple_motion import Simplemotion
