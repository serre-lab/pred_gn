#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa
from .r3d_builder import ResNet3D
# from .gn_builder import GN_R3D #, GN_R3D_CPC
from .gn_pred_builder import GN_R2D #GN_R3D_CPC
from .gn_vpn_builder import GN_R2D_VPN
