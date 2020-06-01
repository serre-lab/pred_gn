#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa
from .r3d_builder import ResNet3D
# from .gn_builder import GN_R3D #, GN_R3D_CPC
from .gn_pred_builder import GN_R2D #GN_R3D_CPC
# from .gn_vpn_builder import GN_R2D_VPN
from .gn_pred_vpn_builder import GN_R2D_VPN
from .simple_gn_pred_vpn_builder import GN_VPN
from .simple_gn_pred_builder import GN_PRED
from .simple_gn_seg_builder import GN_SEG
from .prednet import PredNet, PredNet_E
from .prednet_hGRU import PredNet_hGRU
from .prednet_GRU import PredNet_GRU
from .prednet_small import SmallPredNet
from .gn_small import SmallGN

