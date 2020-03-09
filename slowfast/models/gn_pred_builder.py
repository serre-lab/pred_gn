import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from functools import partial

# from layers.fgru_base import fGRUCell2 as fGRUCell
# from layers.fgru_base import fGRUCell2_td as fGRUCell_td

import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from .batch_norm import get_norm

from .rnns import hConvGRUCell, tdConvGRUCell
# from .rnns import hConvGRUCell, tdConvGRUCell

# from detectron2.layers import (
#     Conv2d,
#     DeformConv,
#     FrozenBatchNorm2d,
#     ModulatedDeformConv,
#     ShapeSpec,
#     get_norm,
# )

# from .head_helper import ResNetSimpleHead
# from .stem_helper import BasicStem
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

# from .backbone import Backbone
from .build import MODEL_REGISTRY

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
    def __init__(self, in_channels=3, out_channels=64, stride=(1, 2, 2), pooling=(1, 3, 3), norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        #TemporalCausalConv3d
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False)
            
        self.bn1 = get_norm(norm, out_channels) # nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=pooling, stride=(1,2,2), padding=(0,1,1))

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


def conv1x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm="BN"):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = get_norm(norm, planes) # nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = get_norm(norm, planes) # nn.BatchNorm3d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm="BN"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm, planes) # nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=False)
        self.bn2 = get_norm(norm, planes) # nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, planes * 4) # nn.BatchNorm3d(planes * 4)
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

    
# def get_norm(norm, out_channels):
#     """
#     Args:
#         norm (str or callable):
#     Returns:
#         nn.Module or None: the normalization layer
#     """
#     if isinstance(norm, str):
#         if len(norm) == 0:
#             return None
#         norm = {
#             "BN": nn.BatchNorm3d,
#             "GN": lambda channels: nn.GroupNorm(channels//4, channels),
#         }[norm]
#     return norm(out_channels)

@MODEL_REGISTRY.register()
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

        recurrent_bn = cfg.GN.RECURRENT_BN 
        feedforward_bn = cfg.GN.FEEDFORWARD_BN
        grad_method = 'bptt'

        self.pred_gn = cfg.PREDICTIVE.ENABLE
        self.cpc_gn = cfg.PREDICTIVE.CPC
        # h_units = [[0, 3],
        #            # [1, 3],
        #            [2, 3],
        #            # [3, 3],
        #            [4, 1]]
        
        # td_units = [[0, 1],
        #            [2, 1]]
        
        h_units = cfg.GN.HORIZONTAL_UNITS
        td_units = cfg.GN.TOPDOWN_UNITS
        gala = cfg.GN.GALA

        self.hidden_init = cfg.GN.HIDDEN_INIT # 'learned'

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
        
        self.stem = BasicStem(3,stem_fan_out, norm=feedforward_bn)
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
        self.horizontal_norms={}
        for i, (loc, ks) in enumerate(h_units):
            # loc, ks = h_params
            fan_in = self._out_feature_channels[loc]
            recurrent = hConvGRUCell(
                input_size=fan_in,
                hidden_size=fan_in,
                kernel_size=ks,
                batchnorm=True,
                timesteps=8,
                gala=gala,
                spatial_kernel=3,
                less_softplus=False,
                r=4,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn)
            horizontal_name = "horizontal{}".format(loc)
            self.add_module(horizontal_name, recurrent)
            self.h_units_and_names.append((recurrent, horizontal_name))
            horizontal_layers += [[horizontal_name, fan_in]]

            horizonta_norm = get_norm('GN', fan_in)
            self.add_module("horizontal_norm{}".format(loc),horizonta_norm)
            self.horizontal_norms[horizontal_name] = horizonta_norm
        
        self.td_units_and_names = []
        topdown_layers = []
        for i, (loc, ks) in enumerate(td_units[::-1]):
            
            if i==0:
                td_fan_in = fan_in

            fan_in = self._out_feature_channels[loc]
            
            recurrent = tdConvGRUCell(
                fan_in=fan_in,
                td_fan_in=td_fan_in,
                diff_fan_in= fan_in - (fan_in-td_fan_in)//2,
                kernel_size=ks,
                gala=gala,
                batchnorm=True,
                timesteps=8,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn)

            topdown_name = "topdown{}".format(loc)
            self.add_module(topdown_name, recurrent)
            self.td_units_and_names.append((recurrent, topdown_name))
            topdown_layers += [[topdown_name, fan_in]]
            
            td_fan_in = fan_in

        pred_fan = self._out_feature_channels[self.h_units[0][0]]
        if self.pred_gn:
            # self.final_remap = nn.Sequential(
            #     nn.Conv3d(fan_in, pred_fan, kernel_size=1,padding=0,stride=1),
            #     nn.ReLU(),
            #     nn.Conv3d(pred_fan, pred_fan, kernel_size=1,padding=0,stride=1),
            #     nn.ReLU(),
            # )

            self.final_remap = nn.Sequential(
                nn.ConvTranspose2d(fan_in,pred_fan,3, stride=2, padding=1, output_padding=1),
                get_norm(feedforward_bn+'2D', pred_fan),
                nn.ReLU(),
                nn.ConvTranspose2d(pred_fan,pred_fan//2, 3, stride=2, padding=1, output_padding=1),
                get_norm(feedforward_bn+'2D', pred_fan//2),
                nn.ReLU(),
                nn.Conv2d(pred_fan//2, 3, 3, padding=1)
            )
        
        if self.cpc_gn:
            cpc_fan_out = cfg.PREDICTIVE.CPC_FAN_OUT
            self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS # [2,4,8]
            cpc_fan_in = self._out_feature_channels[self.h_units[-1][0]] 
            self.cpc_fan_out = cpc_fan_out

            self.W_cpc_target = nn.Linear(cpc_fan_in, cpc_fan_out, bias=False)
            self.W_cpc_preds = {}
            for step in self.cpc_steps:
                w = nn.Linear(cpc_fan_in, cpc_fan_out, bias=False)
                self.add_module('W_cpc_preds_%d'%step, w)
                self.W_cpc_preds[step] = w

        self._out_features = fan_in

        # change init when you add a new layer

        for name, m in self.named_modules():
            if 'horizontal' not in name and 'topdown' not in name:
                if isinstance(m, nn.Conv3d): 
                    # nn.init.kaiming_normal(m.weight, mode='fan_out')
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)): # , nn.GroupNorm 
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm="BN"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=stride,
                            bias=False), 
                    get_norm(norm, planes * block.expansion) ) #nn.BatchNorm3d(planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm=norm))
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
                
        conv_input = self.ds_block(conv_input, current_loc, self.h_units[0][0])
        
        timesteps = conv_input.shape[2]
        current_loc = self.h_units[0][0]
        
        hidden_states= {}
        if self.pred_gn:
            errors = 0
        if self.cpc_gn:
            cpc_targets = []
            cpc_preds = {step: [] for step in self.cpc_steps}
        output = {}

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

            x = torch.zeros_like(conv_input[:,:,0])
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x[:,:,None], current_loc, loc).squeeze(2)
                
                hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name] = h_unit(F.softplus(x), hidden_states[h_name], timestep=0)
                
                x = hidden_states[h_name]

                x = self.horizontal_norms[h_name](x)
                x = F.relu_(x)
            
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
            x = conv_input[:,:,0]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x[:,:,None], current_loc, loc).squeeze(2)
                
                if i == 0 and h_name not in hidden_states:    
                    hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name], extra = h_unit(F.softplus(x), hidden_states[h_name], timestep=i, return_extra=['error'])
                # errors = errors + torch.norm(extra['error'].view(extra['error'].shape[0],-1), p=1, dim=1)/1e4
                # if i > 0:
                #     errors = errors + torch.abs(extra['error'].view(extra['error'].shape[0],-1)).mean(-1)
                
                x = hidden_states[h_name]

                x = self.horizontal_norms[h_name](x)
                x = F.relu_(x)
            
                current_loc = loc
            
            if self.cpc_gn:
                if i >= min(self.cpc_steps):
                    cpc_targets.append(self.W_cpc_target(x.transpose(1,3).detach()).view([-1,self.cpc_fan_out]))
                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        cpc_preds[step].append(self.W_cpc_preds[step](x.transpose(1,3)).view([-1,self.cpc_fan_out]))

            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i)
                    x = hidden_states[h_name]

                # prediction error -> next step lower layer is detached to avoid gradients flowing through lower layers
                # pred_error = F.interpolate(x, conv_input.shape[2:], mode='trilinear', align_corners=True)
                # pred_error = self.final_remap(pred_error) - conv_input[:,:,1][:,:,None].detach()
                # errors = errors + torch.abs(pred_error.view(pred_error.shape[0],-1)).mean(-1)
                
                ## change architecture to take different locations into account
                if self.pred_gn:
                    pred_error = F.smooth_l1_loss(self.final_remap(x), inputs_[:,:,i+1]) # conv_input[:,:,1][:,:,None].detach()
                    errors = errors + pred_error

                conv_input = conv_input[:,:,1:]
        
        logits = self.head(hidden_states[self.h_units_and_names[-1][1]][:,:,None].detach()) 
        output['logits'] = logits
        # del hidden_states
        # del conv_input
        # del x
        if self.pred_gn:
            output['pred_errors'] = errors

        if self.cpc_gn:
            # calculate CPC
        
            # levels of difficulty
            # easy : across batches (dim = 0)
            # medium : across space within sample (fixed dim = 0, dims = 3,4)
            # hard : across time within sample (fixed dim = 0, fixed dims = 3,4, dim=2) 
            
            # label smoothing -> S,T block diag matrix within B,S block diag matrix
            
            cpc_loss = 0
            cpc_targets = torch.stack(cpc_targets,0)
            for step in self.cpc_steps:
                if len(cpc_preds[step])>1:
                    cpc_preds[step] = torch.cat(cpc_preds[step], 0)
        
                    cpc_output = torch.matmul(cpc_targets[step-min(self.cpc_steps):].view([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

                    labels = torch.cumsum(torch.ones_like(cpc_preds[step][:,0]).long(), 0) -1
                    cpc_loss = cpc_loss + F.cross_entropy(cpc_output, labels)
            
            output['cpc_loss'] = cpc_loss

        # {'logits': output, 
        # 'cpc_loss': cpc_loss, 
        # 'pred_errors': errors}
        return output

    def ds_block(self, x, loc_in, loc_out):
        for loc in range(loc_in, loc_out):
            x = self.stages_and_names[loc][0](x)
        
        return x
            

# @MODEL_REGISTRY.register()
class GN_R3D_CPC(GN_R3D):
    
    def __init__(self, cfg):
        super(GN_R3D_CPC, self).__init__(cfg)

        cpc_fan_out = cfg.PREDICTIVE.CPC_FAN_OUT
        self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS # [2,4,8]
        cpc_fan_in = self._out_feature_channels[self.h_units[-1][0]] 
        self.cpc_fan_out = cpc_fan_out

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

            x = torch.zeros_like(conv_input[:,:,0])
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x[:,:,None], current_loc, loc).squeeze(2)
                
                hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=0) # F.softplus(x)
                
                x = hidden_states[h_name]

                x = self.horizontal_norms[h_name](x)
                x = F.relu_(x)
            
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
            x = conv_input[:,:,0]
            current_loc = self.h_units[0][0]
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if j > 0:
                    x = self.ds_block(x[:,:,None], current_loc, loc).squeeze(2)
                
                if i == 0 and h_name not in hidden_states:    
                    hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name], extra = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error']) # F.softplus(x)
                # errors = errors + torch.norm(extra['error'].view(extra['error'].shape[0],-1), p=1, dim=1)/1e4
                # if i > 0:
                #     errors = errors + torch.abs(extra['error'].view(extra['error'].shape[0],-1)).mean(-1)
                
                x = hidden_states[h_name]

                x = self.horizontal_norms[h_name](x)
                x = F.relu_(x)
            
                current_loc = loc
            
            if i >= min(self.cpc_steps):
                cpc_targets.append(self.W_cpc_target(x.transpose(1,3).detach()).view([-1,self.cpc_fan_out]))
            for step in self.cpc_steps:
                if i < timesteps-step: 
                    cpc_preds[step].append(self.W_cpc_preds[step](x.transpose(1,3)).view([-1,self.cpc_fan_out]))

            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i)
                    x = hidden_states[h_name]

                # prediction error -> next step lower layer is detached to avoid gradients flowing through lower layers
                # pred_error = F.interpolate(x, conv_input.shape[2:], mode='trilinear', align_corners=True)
                # pred_error = self.final_remap(pred_error) - conv_input[:,:,1][:,:,None].detach()
                # errors = errors + torch.abs(pred_error.view(pred_error.shape[0],-1)).mean(-1)
                
                ## change architecture to take different locations into account
                pred_error = F.smooth_l1_loss(self.final_remap(x), inputs_[:,:,i+1]) # conv_input[:,:,1][:,:,None].detach()
                errors = errors + pred_error

                conv_input = conv_input[:,:,1:]
        
        output = self.head(hidden_states[self.h_units_and_names[-1][1]][:,:,None].detach()) 
        
        # del hidden_states
        # del conv_input
        # del x
        cpc_loss = 0
        cpc_targets = torch.stack(cpc_targets,0)
        for step in self.cpc_steps:
            if len(cpc_preds[step])>1:
                cpc_preds[step] = torch.cat(cpc_preds[step], 0)
    
                cpc_output = torch.matmul(cpc_targets[step-min(self.cpc_steps):].view([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

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
        
        return {'logits': output, 
                'cpc_loss': cpc_loss, 
                'pred_errors': errors}


