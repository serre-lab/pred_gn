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


from .gn_helper import  TemporalCausalConv3d, \
                        BasicBlock, \
                        Bottleneck, \
                        BasicStem, \
                        ResNetSimpleHead, \
                        SpatialTransformer, \
                        conv1x3x3, \
                        downsample_basic_block
                        
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

from .build import MODEL_REGISTRY

__all__ = [
    "GN_R2D_VPN",
]

@MODEL_REGISTRY.register()
class GN_R2D_VPN(nn.Module):
    
    def __init__(self, cfg):
        super(GN_R2D_VPN, self).__init__()
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
                block, fan_out, n_blocks, shortcut_type, stride=stride, norm=feedforward_bn)

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
                get_norm(feedforward_bn, pred_fan), #+'2D'
                nn.ReLU(),
                nn.ConvTranspose2d(pred_fan,pred_fan//2, 3, stride=2, padding=1, output_padding=1),
                get_norm(feedforward_bn, pred_fan//2), #+'2D'
                nn.ReLU(),
                nn.Conv2d(pred_fan//2, 3, 3, padding=1)
            )

        self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
        fan_in = self._out_feature_channels[self.h_units[-1][0]] 
        # self.spatial_transform = SpatialTransformer(fan_in=fan_in)
        # self.feature_transform = nn.Conv2d(fan_in, kernel_size=1)
        self.spatial_transforms = {}
        self.feature_transforms = {}
        for step in self.cpc_steps:
            ws = SpatialTransformer(fan_in=fan_in)
            wf = nn.Conv2d(fan_in, fan_in, kernel_size=1)
            self.add_module('spatial_transform_%d'%step, ws)
            self.add_module('feature_transform_%d'%step, wf)
            self.spatial_transforms[step] = ws
            self.feature_transforms[step] = wf

        self._out_features = fan_in

        # change init when you add a new layer

        for name, m in self.named_modules():
            if 'horizontal' not in name and 'topdown' not in name:
                if isinstance(m, (nn.Conv3d,nn.Conv2d, nn.ConvTranspose2d)): 
                    # nn.init.kaiming_normal(m.weight, mode='fan_out')
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    m.bias.data.zero_()
                elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)): # , nn.GroupNorm 
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

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
            layers.append(block(self.inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, inputs_, return_frames=False):
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
            if return_frames:
                frames = []
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

                if (x>10000).any():
                    logger.info('variable %s at timestep %d out of bound'%(h_name,i))
                x = self.horizontal_norms[h_name](x)
                x = F.relu_(x)
                
                current_loc = loc

            if self.cpc_gn:
                if i >= min(self.cpc_steps):
                    cpc_targets.append(x.detach())
                
                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        cpc_preds[step].append(self.spatial_transforms[step](x, self.feature_transforms[step](x)))

            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i)
                    x = hidden_states[h_name]
                    if (x>10000).any():
                        logger.info('variable %s at timestep %d out of bound'%(td_name,i))

                # prediction error -> next step lower layer is detached to avoid gradients flowing through lower layers
                # pred_error = F.interpolate(x, conv_input.shape[2:], mode='trilinear', align_corners=True)
                # pred_error = self.final_remap(pred_error) - conv_input[:,:,1][:,:,None].detach()
                # errors = errors + torch.abs(pred_error.view(pred_error.shape[0],-1)).mean(-1)
                
                ## change architecture to take different locations into account
                if self.pred_gn:
                    frame = self.final_remap(x)
                    if return_frames:
                        
                        frames.append(frame)
                    pred_error = F.smooth_l1_loss(frame, inputs_[:,:,i+1]) # conv_input[:,:,1][:,:,None].detach()
                    errors = errors + pred_error

                conv_input = conv_input[:,:,1:]
        
        logits = self.head(hidden_states[self.h_units_and_names[-1][1]][:,:,None].detach()) 
        output['logits'] = logits
        # del hidden_states
        # del conv_input
        # del x
        if self.pred_gn:
            output['pred_errors'] = errors
            if return_frames:
                frames = torch.stack(frames, 2)
                output['frames'] = frames

        if self.cpc_gn:

            cpc_loss = 0
            cpc_targets = torch.stack(cpc_targets,-1).transpose(1,4)
            
            for step in self.cpc_steps:
                if len(cpc_preds[step])>1:
                    cpc_preds[step] = torch.stack(cpc_preds[step], -1).transpose(1,4)
                    # .permute(1,0,3,4,2) #T B C H W -> B T H W C
                    # logger.info(cpc_preds[step].shape)
                    cpc_preds[step] = cpc_preds[step].reshape([-1,cpc_preds[step].shape[-1]]) # -> N C
                    # logger.info(cpc_targets[:,step-min(self.cpc_steps):].shape)
                    cpc_output = torch.matmul(cpc_targets[:,step-min(self.cpc_steps):].reshape([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

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
            

# differentiable warping + feature transformation
class SpatialTransformer(nn.Module):
    def __init__(self, fan_in):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network

        self.localization = nn.Sequential(
            nn.Conv2d(fan_in, 64, kernel_size=5),
            #nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x, input_trans=None):
        if input_trans == None:
            input_trans = x 
        xs = self.localization(x)
        xs = F.relu(F.max_pool2d(xs, kernel_size=xs.size()[2:]))

        xs = xs.view(-1, xs.shape[1])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input_trans.size(), align_corners=True)
        x = F.grid_sample(input_trans, grid, align_corners=True)

        return x

