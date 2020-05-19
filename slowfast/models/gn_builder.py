import numpy as np
from functools import partial
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import init

# from .head_helper import ResNetSimpleHead
# from .stem_helper import BasicStem
import slowfast.utils.logging as logging

# from .backbone import Backbone
from .build import MODEL_REGISTRY

# from layers.fgru_base import fGRUCell2 as fGRUCell
# from layers.fgru_base import fGRUCell2_td as fGRUCell_td

# from .rnns import hConvGRUCell, tdConvGRUCell

# from detectron2.layers import (
#     Conv2d,
#     DeformConv,
#     FrozenBatchNorm2d,
#     ModulatedDeformConv,
#     ShapeSpec,
#     get_norm,
# )


logger = logging.get_logger(__name__)


__all__ = [
    "ResNetBlockBase",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "ResNet3D",
]

class TemporalCausalConv3d(nn.Conv3d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=1,
                 groups=1,
                 bias=True):
        super(TemporalCausalConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        print(padding)
        self.causal_padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        #self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input, 
                  (self.causal_padding[0]*2, 0, 
                   self.causal_padding[1], self.causal_padding[1], 
                   self.causal_padding[2], self.causal_padding[2]))
        return super().forward(x)

class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, stride=(2, 2, 2), pooling=(3, 3, 3), norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        #TemporalCausalConv3d
        self.conv1 = TemporalCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=pooling, stride=(2,2,2), padding=(1,1,1))

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool
    
class ResNetSimpleHead(nn.Module):
    """
    ##### Similar to ResNetBasicHead without the pathways
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        input_fan,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetSimpleHead, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        # self.avg_pool
        self.fc = nn.Linear(input_fan, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        
        x = self.avgpool(inputs)
        
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        # Performs fully convolutional inference.
        if not self.training:
            x = self.act(x)
            # x = x.mean([1, 2, 3])

        return x


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=stride,
        padding=(0,1,1),
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.autograd.Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            gala=False,
            spatial_kernel=5,
            hidden_init='zeros',
            r=4,
            grad_method='bptt',
            norm='GroupNorm'):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala

        if self.gala:
            self.u1_channel_gate_0 = nn.Conv3d(
                hidden_size, hidden_size // r, 1)
            self.u1_channel_gate_1 = nn.Conv3d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u1_spatial_gate_0 = nn.Conv3d(
                hidden_size, hidden_size // r, (1,spatial_kernel,spatial_kernel), padding=(0,1,1))
            self.u1_spatial_gate_1 = nn.Conv3d(
                hidden_size // r, 1, (1,spatial_kernel,spatial_kernel), padding=(0,1,1), bias=False)
            self.u1_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1, 1)))
        else:
            self.u1_gate = nn.Conv3d(hidden_size, hidden_size, 1)
            nn.init.xavier_uniform_(self.u1_gate.weight)
        self.u2_gate = nn.Conv3d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, 1, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, 1, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1, 1)))

        # if norm == "":
        #     norm = 'SyncBN'

        # Norm is harcoded to group norm
        if norm == 'GroupNorm':
            norm = 'GN'
        else:
            norm = 'BN'

        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.mu, 1)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            nn.init.uniform_(self.u1_combine_bias, 1, init_timesteps - 1)
            self.u1_combine_bias.data.log()
            self.u2_gate.bias.data = -self.u1_combine_bias.data
        else:
            nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1)
            self.u1_gate.bias.data.log()
            self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, input_, h_, timestep=0, return_extra=[]):
        
        if timestep==0 and h_ is None:
            if self.hidden_init=='identity':
                h_ = input_
            else:
                h_ = torch.zeros_like(input_)

        extra = {}
        if self.gala:
            global_0 = F.softplus(self.u1_channel_gate_0(h_))
            global_1 = self.u1_channel_gate_1(global_0)
            local_0 = F.softplus(self.u1_spatial_gate_0(h_))
            local_1 = self.u1_spatial_gate_1(local_0)
            import pdb; pdb.set_trace()
            g1_t = F.softplus(global_1 * local_1 + self.u1_combine_bias)
        else:
            g1_t = torch.sigmoid(self.u1_gate(h_))
        c1_t = self.bn[0](
            F.conv3d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=(0, self.padding, self.padding)))
        error = F.softplus(c1_t * (self.alpha * h_ + self.mu))
        # if 'error' in return_extra:
        #     extra['error'] = error
        next_state1 = F.softplus(input_ - error)
        if 'error' in return_extra:
            extra['error'] = next_state1
        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv3d(
                next_state1,
                self.w_gate_exc,
                padding=(0, self.padding, self.padding)))
        
        h_ = (1 - g2_t) * h_ + g2_t * h2_t
        if not extra:
            return h_
        else:
            return h_, extra


class tdConvGRUCell(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            diff_size,
            kernel_size,
            batchnorm=True,
            hidden_init='zeros',
            timesteps=8,
            grad_method='bptt',
            norm='GroupNorm'):
        
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.remap_0 = nn.Conv3d(hidden_size, diff_size, 1)
        self.remap_1 = nn.Conv3d(diff_size, input_size, 1)

        self.u1_gate = nn.Conv3d(input_size, input_size, 1)
        self.u2_gate = nn.Conv3d(input_size, input_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(input_size, input_size, 1, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(input_size, input_size, 1, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((input_size, 1, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((input_size, 1, 1, 1)))
        self.mu = nn.Parameter(torch.empty((input_size, 1, 1, 1)))

        self.hidden_init = hidden_init
        # if norm == "":
        #     norm = 'SyncBN'

        # Norm is harcoded to group norm
        # norm = 'GN'
        if norm == 'GroupNorm':
            norm = 'GN'
        else:
            norm = 'BN'
        self.bn = nn.ModuleList(
            [get_norm(norm, input_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        nn.init.xavier_uniform_(self.w_gate_inh)
        nn.init.xavier_uniform_(self.w_gate_exc)

        nn.init.xavier_uniform_(self.u1_gate.weight)
        nn.init.xavier_uniform_(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.mu, 1)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        nn.init.uniform_(self.u1_gate.bias.data, 1, init_timesteps - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, lower_, higher_, timestep=0, return_extra=[]):
        extra = {}
        prev_state2 = F.interpolate(
            higher_,
            size = lower_.shape[2:],
            #scale_factor=2,
            mode="nearest")
        
        prev_state2 = F.softplus(self.remap_0(prev_state2))
        prev_state2 = F.softplus(self.remap_1(prev_state2))
        
        # if timestep==0 and lower_is None:
        #     if self.hidden_init=='identity':

        if 'remap' in return_extra:
            extra['remap'] = prev_state2
        
        g1_t = torch.sigmoid(self.u1_gate(prev_state2))
        c1_t = self.bn[0](
            F.conv3d(
                prev_state2 * g1_t,
                self.w_gate_inh,
                padding=(0, self.padding, self.padding)))
        
        error = F.softplus(c1_t * (self.alpha * prev_state2 + self.mu))
        
        next_state1 = F.softplus(lower_ - error)
        if 'error' in return_extra:
            extra['error'] = next_state1
        # if 'h1' in return_extra:
        #     extra['h1'] = next_state1
            
        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv3d(
                next_state1,
                self.w_gate_exc,
                padding=(0, self.padding, self.padding)))

        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t
        
        if not extra:
            return prev_state2
        else:
            return prev_state2, extra
    
def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):
    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(channels//4, channels),
        }[norm]
    return norm(out_channels)

# @MODEL_REGISTRY.register()
class GN_R3D(nn.Module):
    
    def __init__(self, cfg):
        super(GN_R3D, self).__init__()
        # should include temporal kernel and temporal stride causal padding option 
        __RESNET_PARAMS__ = {
            'resnet10': ['BasicBlock', [[1,64], [1,128], [1,256], [1,512]]],
            'resnet18': ['BasicBlock', [[2,64], [2,128], [2,256], [2,512]]],
            'resnet34': ['BasicBlock', [[3,64], [4,128], [6,256], [3,512]]],
            'resnet50': ['Bottleneck', [[3,64], [4,128], [6,256], [3,512]]],
            'resnet101': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]], 
            'resnet152': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]], ###### fix resnet 152
            'resnet200': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]]
        }
        ########################################################
        # variables
        
        stem_fan_out = 64
        shortcut_type = 'B'
        block_type, stages = __RESNET_PARAMS__[cfg.MODEL.ARCH]
    
        num_classes = cfg.MODEL.NUM_CLASSES
        dropout_rate = cfg.MODEL.DROPOUT_RATE

        recurrent_bn = cfg.GN.RECURRENT_BN # 'GroupNorm'
        grad_method = 'bptt'

        self.pred_gn=True

        # h_units = [[0, 3],
        #            # [1, 3],
        #            [2, 3],
        #            # [3, 3],
        #            [4, 1]]
        
        # td_units = [[0, 1],
        #            [2, 1]]
        
        h_units = cfg.GN.HORIZONTAL_UNITS
        td_units = cfg.GN.TOPDOWN_UNITS
        self.h_units = h_units
        self.td_units = td_units

        ########################################################
        
        assert len(td_units) <= len(h_units)-1, 'wrong number of horizontal or top-down units'
        h_locations = [h[0] for h in h_units]
        for td in td_units:
            assert td[0] in h_locations, 'no horizontal location found for td location' 

        if block_type == 'BasicBlock':
            block = BasicBlock
        elif block_type == 'Bottleneck':
            block = Bottleneck
            
        # self.learned_init = True
        self.hidden_init = 'learned'
        self.stem = BasicStem(3,stem_fan_out)
        self._strides = [4]
        self._out_feature_channels =[stem_fan_out]
        
        self.inplanes = stem_fan_out

        self.stages_and_names = []
        fan_in = stem_fan_out
        for i, stage in enumerate(stages):
            n_blocks, fan_out = stage
            stride = 1 if i==0 else 2
                
            stage_seq = self._make_layer(
                block, fan_out, n_blocks, shortcut_type, stride=stride)

            name = "res" + str(i + 1)
            self.add_module(name, stage_seq)
            self.stages_and_names.append((stage_seq, name))

            fan_in = self.inplanes
            self._out_feature_channels.append(self.inplanes)
            self._strides.append(self._strides[-1]*stride)
            
        self.head = ResNetSimpleHead(fan_in, num_classes, dropout_rate)
        
        # self._h_feature_channels = []
        self.h_units_and_names = []
        horizontal_layers = []
        for i, (loc, ks) in enumerate(h_units):
            # loc, ks = h_params
            fan_in = self._out_feature_channels[loc]
            recurrent = hConvGRUCell(
                input_size=fan_in,
                hidden_size=fan_in,
                kernel_size=ks,
                batchnorm=recurrent_bn,
                # timesteps=timesteps,
                grad_method=grad_method)
            horizontal_name = "horizontal{}".format(loc)
            self.add_module(horizontal_name, recurrent)
            self.h_units_and_names.append((recurrent, horizontal_name))
            horizontal_layers += [[horizontal_name, fan_in]]
        
        self.td_units_and_names = []
        topdown_layers = []
        for i, (loc, ks) in enumerate(td_units[::-1]):
            
            if i==0:
                td_fan_in = fan_in

            fan_in = self._out_feature_channels[loc]
            
            recurrent = tdConvGRUCell(
                input_size=fan_in,
                hidden_size=td_fan_in,
                diff_size=fan_in//2,#diff_fan,
                kernel_size=ks,
                batchnorm=recurrent_bn,
                # timesteps=timesteps,
                grad_method=grad_method)

            topdown_name = "topdown{}".format(loc)
            self.add_module(topdown_name, recurrent)
            self.td_units_and_names.append((recurrent, topdown_name))
            topdown_layers += [[topdown_name, fan_in]]
            
            td_fan_in = fan_in

        pred_fan = self._out_feature_channels[self.h_units[0][0]]
        if self.pred_gn:
            self.final_remap = nn.Sequential(
                nn.Conv3d(fan_in, pred_fan, kernel_size=1,padding=0,stride=1),
                nn.ReLU(),
                nn.Conv3d(pred_fan, pred_fan, kernel_size=1,padding=0,stride=1),
                nn.ReLU(),
            )

        self._out_features = fan_in

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs_):
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]
        x = inputs_
        current_loc = 0
        conv_input = self.stem(x)
        # print(conv_input.shape)
                
        conv_input = self.ds_block(conv_input, current_loc, self.h_units[0][0])
        
        timesteps = conv_input.shape[2]

        current_loc = self.h_units[0][0]
        
        hidden_states= {}
        errors = 0
        
        if self.hidden_init=='learned':
            # h_unit, h_name =  self.h_units_and_names[-1]
            # # x = torch.zeros_like(conv_input[:,:,0])[:,:,None]
            # x = inputs_.new(torch.zeros([inputs_.shape[0],
            #                        self._out_feature_channels[-1],  
            #                        1,    
            #                         inputs_.shape[3]//self._strides[-1], 
            #                         inputs_.shape[4]//self._strides[-1]]))
            # hidden_states[h_name] = x #torch.zeros_like(x)
            
            # hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=0, return_extra=[])
            
            # x = hidden_states[h_name]

            x = torch.zeros_like(conv_input[:,:,0])[:,:,None]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x, current_loc, loc)
                
                hidden_states[h_name] = torch.zeros_like(x)
                
                hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=0, return_extra=[])
                
                x = hidden_states[h_name]
            
                current_loc = loc
            
            for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                loc = int(td_name.strip('topdown'))
                h_name = 'horizontal'+str(loc)
                # print(x.shape)
                
                # hidden_states[h_name] = x.new(torch.zeros([x.shape[0],
                #                    self._out_feature_channels[loc],  
                #                    1,    
                #                     inputs_.shape[3]//self._strides[loc], 
                #                     inputs_.shape[4]//self._strides[loc]]))
        
                hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=0)
                x = hidden_states[h_name]
            
        
        for i in range(timesteps):
            x = conv_input[:,:,0][:,:,None]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x, current_loc, loc)
                
                if i == 0 and h_name not in hidden_states:    
                    hidden_states[h_name] = torch.zeros_like(x)
                
                hidden_states[h_name], extra = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error'])
                # errors = errors + torch.norm(extra['error'].view(extra['error'].shape[0],-1), p=1, dim=1)/1e4
                if i > 0:
                    errors = errors + torch.abs(extra['error'].view(extra['error'].shape[0],-1)).mean(-1)
                x = hidden_states[h_name]
            
                current_loc = loc
        
            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i)
                    x = hidden_states[h_name]
                
                
                pred_error = F.interpolate(x, conv_input.shape[2:], mode='trilinear', align_corners=True)
                pred_error = self.final_remap(pred_error) - conv_input[:,:,1][:,:,None]
                # errors = errors + torch.norm(pred_error.view(pred_error.shape[0],-1), p=1, dim=1)/1e4
                errors = errors + torch.abs(pred_error.view(pred_error.shape[0],-1)).mean(-1)
                conv_input = conv_input[:,:,1:]

        # calculate targets from conv_input (cpc_steps) targets = W*conv_input[min(cpc_steps):]
        # keep them all in memory

        # for each step
            # C should be small
            # if step > min(cpc_steps) calculate target from top layer
            # cpc_targets [B, C, mincpc : timesteps, H, W]
            # for each cpc_step
                # if i < timesteps - cpc_step calculate prediction from top layer
                # cpc_preds [B, C, 0 : timesteps-cpc_step-1, H, W]
                # [preds_step2, preds_step3, preds_step4]

        # calculate CPC
        
        # levels of difficulty
        # easy : across batches (dim = 0)
        # medium : across space within sample (fixed dim = 0, dims = 3,4)
        # hard : across time within sample (fixed dim = 0, fixed dims = 3,4, dim=2) 
        
        # output = targets
        # transpose to B,S,T,C

        # label smoothing -> S,T block diag matrix within B,S block diag matrix
            
        # x = self.head(hidden_states[self.h_units_and_names[-1][1]])
        
        return errors

    def ds_block(self, x, loc_in, loc_out):
        for loc in range(loc_in, loc_out):
            x = self.stages_and_names[loc][0](x)
        
        return x
            
    
# m = GN_R3D(None)

# print(m)

# a = torch.randn([2,3,16,128,128])

# o = m(a)


# @MODEL_REGISTRY.register()
class GN_R3D_CPC(GN_R3D):
    
    def __init__(self, cfg):
        super(GN_R3D_CPC, self).__init__(cfg)

        # CPC weights:
        # w target
        # for each future step:
            # w preds step

        self.cpc_steps = [2,3,4]
        cpc_fan_in = self._out_feature_channels[self.h_units[-1][0]]
        cpc_fan_out = 32 

        # self.W_cpc_target = nn.Conv2d(cpc_fan_in, cpc_fan_out, kernel=1, bias=False)
        # self.W_cpc_target = nn.Linear(cpc_fan_in, cpc_fan_out, bias=False)
        # self.W_cpc_preds = {step: nn.Linear(cpc_fan_in, cpc_fan_out, bias=False) for step in self.cpc_steps}
        # for step in self.cpc_steps:
        #     self.add_module('W_cpc_preds_%d'%step, self.W_cpc_preds[step])
        
        # self.cpc_steps = [2,3,4]
        # cpc_fan_in = self._out_feature_channels[self.h_units[-1][0]]
        # cpc_fan_out = 32 

        # self.W_cpc_target = nn.Conv2d(cpc_fan_in, cpc_fan_out, kernel=1, bias=False)
        self.W_cpc_target = nn.Linear(cpc_fan_in, cpc_fan_out, bias=False)
        self.W_cpc_preds = {}
        for step in self.cpc_steps:
            w = nn.Linear(cpc_fan_in, cpc_fan_out, bias=False)
            self.add_module('W_cpc_preds_%d'%step, w)
            self.W_cpc_preds[step] = w

    def forward(self, inputs_):
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]
        x = inputs_
        current_loc = 0
        conv_input = self.stem(x)
        # print(conv_input.shape)
                
        conv_input = self.ds_block(conv_input, current_loc, self.h_units[0][0])
        
        timesteps = conv_input.shape[2]

        current_loc = self.h_units[0][0]
        
        hidden_states= {}
        errors = 0
        
        if self.hidden_init=='learned':
            # h_unit, h_name =  self.h_units_and_names[-1]
            # # x = torch.zeros_like(conv_input[:,:,0])[:,:,None]
            # x = inputs_.new(torch.zeros([inputs_.shape[0],
            #                        self._out_feature_channels[-1],  
            #                        1,    
            #                         inputs_.shape[3]//self._strides[-1], 
            #                         inputs_.shape[4]//self._strides[-1]]))
            # hidden_states[h_name] = x #torch.zeros_like(x)
            
            # hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=0, return_extra=[])
            
            # x = hidden_states[h_name]

            x = torch.zeros_like(conv_input[:,:,0])[:,:,None]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x, current_loc, loc)
                
                hidden_states[h_name] = torch.zeros_like(x)
                
                hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=0, return_extra=[])
                
                x = hidden_states[h_name]
            
                current_loc = loc
            
            for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                loc = int(td_name.strip('topdown'))
                h_name = 'horizontal'+str(loc)
                # print(x.shape)
                
                # hidden_states[h_name] = x.new(torch.zeros([x.shape[0],
                #                    self._out_feature_channels[loc],  
                #                    1,    
                #                     inputs_.shape[3]//self._strides[loc], 
                #                     inputs_.shape[4]//self._strides[loc]]))
        
                hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=0)
                x = hidden_states[h_name]
            
        cpc_targets = []
        cpc_preds = {step: [] for step in self.cpc_steps}
        for i in range(timesteps):
            x = conv_input[:,:,0][:,:,None]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x, current_loc, loc)
                
                if i == 0 and h_name not in hidden_states:    
                    hidden_states[h_name] = torch.zeros_like(x)
                
                hidden_states[h_name], extra = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error'])
                # errors = errors + torch.norm(extra['error'].view(extra['error'].shape[0],-1), p=1, dim=1)/1e4
                if i > 0:
                    errors = errors + torch.abs(extra['error'].view(extra['error'].shape[0],-1)).mean(-1)
                x = hidden_states[h_name]
            
                current_loc = loc
            self.cpc_fan_out = 32
            if i >= min(self.cpc_steps):
                cpc_targets.append(self.W_cpc_target(x.transpose(1,4)).view([-1,self.cpc_fan_out]))
            for step in self.cpc_steps:
                
                if i < timesteps-step: 
                    cpc_preds[step].append(self.W_cpc_preds[step](x.transpose(1,4)).view([-1,self.cpc_fan_out]))

            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i)
                    x = hidden_states[h_name]
                
                
                pred_error = F.interpolate(x, conv_input.shape[2:], mode='trilinear', align_corners=True)
                pred_error = self.final_remap(pred_error) - conv_input[:,:,1][:,:,None]
                # errors = errors + torch.norm(pred_error.view(pred_error.shape[0],-1), p=1, dim=1)/1e4
                errors = errors + torch.abs(pred_error.view(pred_error.shape[0],-1)).mean(-1)
                conv_input = conv_input[:,:,1:]
        
        output = self.head(hidden_states[self.h_units_and_names[-1][1]].detach()) 
        
        # del hidden_states
        # del conv_input
        # del x
        cpc_loss = 0
        cpc_targets = torch.stack(cpc_targets,0)
        for step in self.cpc_steps:
            if len(cpc_preds[step])>1:
                cpc_preds[step] = torch.cat(cpc_preds[step], 0)
    
                cpc_output = torch.matmul(cpc_targets[step-min(self.cpc_steps):].view([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())
                # label = torch.eye(cpc_preds[step].shape[0])
                
                labels = torch.cumsum(torch.ones_like(cpc_preds[step][:,0]).long(), 0) -1
                cpc_loss = cpc_loss + F.cross_entropy(cpc_output, labels)


        # calculate CPC
        
        # levels of difficulty
        # easy : across batches (dim = 0)
        # medium : across space within sample (fixed dim = 0, dims = 3,4)
        # hard : across time within sample (fixed dim = 0, fixed dims = 3,4, dim=2) 
        
        # output = targets
        # transpose to B,S,T,C

        # label smoothing -> S,T block diag matrix within B,S block diag matrix
            
        # x = self.head(hidden_states[self.h_units_and_names[-1][1]])
        
        return errors, cpc_loss, output
