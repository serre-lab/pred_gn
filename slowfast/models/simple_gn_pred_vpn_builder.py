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

from .batch_norm import get_norm, GroupNorm

from .rnns import hConvGRUCell, tdConvGRUCell, tdConvGRUCell_err


from .gn_helper import  TemporalCausalConv3d, \
                        BasicBlock, \
                        Bottleneck, \
                        BasicStem, \
                        ResNetSimpleHead, \
                        SpatialTransformer, \
                        conv1x3x3, \
                        downsample_basic_block
                        
import kornia
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

from .build import MODEL_REGISTRY

__all__ = [
    "GN_VPN",
]

@MODEL_REGISTRY.register()
class GN_VPN(nn.Module):
    
    def __init__(self, cfg):
        super(GN_VPN, self).__init__()
        # should include temporal kernel and temporal stride causal padding option 
        # __RESNET_PARAMS__ = {
        #     'resnet10': ['BasicBlock', [[1,64], [1,128], [1,256], [1,512]]],
        #     'resnet18': ['BasicBlock', [[2,64], [2,128], [2,256], [2,512]]],
        #     'resnet34': ['BasicBlock', [[3,64], [4,128], [6,256], [3,512]]],
        #     'resnet50': ['Bottleneck', [[3,64], [4,128], [6,256], [3,512]]],
        #     'resnet101': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]], 
        #     'resnet152': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]], ###### fix resnet 152
        #     'resnet200': ['Bottleneck', [[3,64], [4,128], [23,256], [3,512]]]
        # }
        ########################################################
        # variables
        
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


        
        # self.color_aug = kornia.augmentation.ColorJitter(brightness= 0.05, contrast= 0.05, saturation = 0.05, hue = 0.05)
        # self.affine_aug = kornia.augmentation.RandomAffine(degrees=(-5,5), translate=(2/224,2/224), scale=(0.95, 1.05), shear=(-1,1))
        self.color_aug = kornia.augmentation.ColorJitter(brightness= 0.1, contrast= 0.1, saturation = 0.1, hue = 0.2, return_transform=True)
        self.affine_aug = kornia.augmentation.RandomAffine(degrees=(-3,3), translate=(5/224,5/224), scale=(0.9, 1.1), shear=(-0.02,0.02), return_transform=True)

        # self._strides = [1]
        # self._out_feature_channels =[3]
        
        # self.inplanes = 3

        # self.stages_and_names = []
        # fan_in = stem_fan_out
        # for i, stage in enumerate(stages):
        #     n_blocks, fan_out = stage
        #     # stride = 1 if i==0 else 2
                
        #     stage_seq = nn.Sequential(
        #         nn.Conv2d(fan_in, fan_out, 3, stride=1, padding=1),
        #         nn.MaxPool2d(2)
        #     )
            
        #     name = "res" + str(i + 1)
        #     self.add_module(name, stage_seq)
        #     self.stages_and_names.append((stage_seq, name))

        #     self.inplanes = fan_out
        #     fan_in = self.inplanes
        #     self._out_feature_channels.append(self.inplanes)
        #     self._strides.append(self._strides[-1]*2)
        
        ### small expansion of the image
        self.stem = nn.Conv2d(3,8,3,padding=1)
        self._strides = [1]
        self._out_feature_channels =[8]
        
        fan_in = 8
        self.feedforward_units = []
        self.h_units_and_names = []
        horizontal_layers = []

        for i, (_, ks) in enumerate(h_units):
            # loc, ks = h_params
            recurrent = hConvGRUCell(
                input_size=fan_in,
                hidden_size=fan_in,
                kernel_size=ks,
                batchnorm=True,
                timesteps=18,
                gala=gala,
                spatial_kernel=3,
                less_softplus=False,
                r=4,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn)
            horizontal_name = "horizontal{}".format(i)
            self.add_module(horizontal_name, recurrent)
            self.h_units_and_names.append((recurrent, horizontal_name))
            horizontal_layers += [[horizontal_name, fan_in]]
            if i< len(h_units)-1:
                if i==0:
                    fan_out = 48
                else:
                    fan_out = fan_in*2
                
                feedforward = nn.Sequential(
                    nn.Conv2d(fan_in, fan_out, 3, stride=1, padding=1),
                    nn.MaxPool2d(2)
                )
                self._out_feature_channels.append(fan_out)
                self._strides.append(self._strides[-1]*2)
                self.add_module('feedforward{}'.format(i), feedforward)
                self.feedforward_units.append(feedforward)
                fan_in = fan_out
        
        if cfg.SUPERVISED:
            self.run_classification = True
            self.head = ResNetSimpleHead(fan_in, num_classes, dropout_rate)
        else:
            self.run_classification = False
        self.td_units_and_names = []
        topdown_layers = []
        for i, (loc, ks) in enumerate(td_units[::-1]):
            
            if i==0:
                td_fan_in = fan_in

            fan_in = self._out_feature_channels[loc]
            
            recurrent = tdConvGRUCell_err(
                fan_in=fan_in,
                td_fan_in=td_fan_in,
                diff_fan_in= fan_in - (fan_in-td_fan_in)//2,
                kernel_size=ks,
                gala=gala,
                batchnorm=True,
                timesteps=18,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn,
                spatial_transform=True,
                gn_remap=False)

            topdown_name = "topdown{}".format(loc)
            self.add_module(topdown_name, recurrent)
            self.td_units_and_names.append((recurrent, topdown_name))
            topdown_layers += [[topdown_name, fan_in]]
            
            td_fan_in = fan_in

        pred_fan = self._out_feature_channels[self.h_units[0][0]]
        if self.pred_gn:

            # self.final_remap = nn.Sequential(
            #     nn.ConvTranspose2d(fan_in,pred_fan,3, stride=2, padding=1, output_padding=1),
            #     get_norm(feedforward_bn, pred_fan), #+'2D'
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(pred_fan,pred_fan//2, 3, stride=2, padding=1, output_padding=1),
            #     get_norm(feedforward_bn, pred_fan//2), #+'2D'
            #     nn.ReLU()
            # )
            # self.final_remap = nn.Sequential(
            #     nn.Conv2d(3, 3, 3, padding=1),
            #     nn.ReLU()
            # )
            # self.fb_warp = SpatialTransformer(fan_in=fan_in)
            self.output_conv = nn.Conv2d(pred_fan, 3, 3, padding=1)

        if self.cpc_gn:
            self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
            fan_in = self._out_feature_channels[self.h_units[-1][0]] 
            # self.spatial_transform = SpatialTransformer(fan_in=fan_in)
            # self.feature_transform = nn.Conv2d(fan_in, kernel_size=1)
            self.spatial_transforms = {}
            self.feature_transforms = {}
            for step in self.cpc_steps:
                ws = SpatialTransformer(fan_in=fan_in)
                wf = nn.Conv2d(fan_in, fan_in, kernel_size=1)
                self.add_module('bu_warp_%d'%step, ws)
                self.add_module('bu_feat_%d'%step, wf)
                self.spatial_transforms[step] = ws
                self.feature_transforms[step] = wf

        self._out_features = fan_in

        # change init when you add a new layer

        self.group_norm_layers=[]
         
        for name, m in self.named_modules():
            if 'horizontal' not in name and 'topdown' not in name:
                if isinstance(m, (nn.Conv3d,nn.Conv2d, nn.ConvTranspose2d)): 
                    # nn.init.kaiming_normal(m.weight, mode='fan_out')
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, GroupNorm)): # , nn.GroupNorm 
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.zero_()
            if isinstance(m, GroupNorm):
                self.group_norm_layers.append(m)


    def forward(self, inputs_, extra=[], autoreg=False):
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]

        for gnl in self.group_norm_layers:
            gnl.reset_stats()

        current_loc = 0
        aug_inputs = []
        
        timesteps = inputs_.size(2)

        if self.training:
            o_ ,aug_c = self.color_aug(inputs_[:,:,0])
            o_ ,aug_a = self.affine_aug(o_)
            # tr = kornia.linalg.compose_transformations(aug_c, aug_a)
            aug_inputs.append(o_)

            for i in range(1,inputs_.shape[2]):
                #aug_inputs.append(kornia.warp_perspective(inputs_[:,:,i], tr, dsize=inputs_.shape[-2:]))
                o_ = self.affine_aug(self.color_aug(inputs_[:,:,i], params=aug_c)[0], params=aug_a)[0]
                # o_ ,aug_a = self.affine_aug(o_, params=aug_a)
                aug_inputs.append(o_)
            aug_inputs = torch.stack(aug_inputs, 2)

            # for i in range(inputs_.shape[0]):
            #     aug_inputs.append(self.affine_aug(self.color_aug(inputs_[i].transpose(0,1))).transpose(0,1))
                
            #     # b = kornia.warp_perspective(a, tr_c, dsize=a.shape[-2:])
            #     # c = kornia.warp_perspective(b, tr_a, dsize=a.shape[-2:])
            # aug_inputs = torch.stack(aug_inputs, 0)
        else:
            aug_inputs = inputs_

        
        conv_input = aug_inputs

        current_loc = self.h_units[0][0]

        hidden_states = {}
        bu_errors = {}

        # mix_layer_out = []
        # bu_errors_out =[]
        # H_inh_out = []
        # hidden_out = []

        if self.pred_gn:
            errors = 0
            if 'frames' in extra:
                frames = []
        if self.cpc_gn:
            cpc_targets = []
            cpc_preds = {step: [] for step in self.cpc_steps}
        
        output = {}

        # if 'input_aug' in extra:
        output['input_aug'] = aug_inputs
        
        ##########################################################################################
        # learned initialization

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
                    # x = self.ds_block(x[:,:,None], current_loc, loc).squeeze(2)
                    x = self.ds_block(x, current_loc, loc)
                
                hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name], extra_h = h_unit(F.softplus(x), hidden_states[h_name], timestep=0, return_extra=['error'])
                bu_errors[h_name] = extra_h['error']
                x = hidden_states[h_name]

                # x = F.relu_(x)
            
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
        
                hidden_states[h_name], extra_td = td_unit(hidden_states[h_name], x, timestep=0, return_extra=['error'])
                x = extra_td['error']

        ##########################################################################################
        # gammanet loop

        for i in range(timesteps):
            if autoreg and i>timesteps//2:
                x = frame
            else:
                x = conv_input[:,:,0]
            current_loc = self.h_units[0][0]
            
            if 'hidden_errors' in extra:
                hidden_errors.append([])

            ### small expansion of the input
            x = self.stem(x)

            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                
                if i == 0 and h_name not in hidden_states:    
                    ### zeros initialization
                    hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                    ### identity initialization
                    # hidden_states[h_name] = F.softplus(x)
                
                # hidden_states[h_name], extra_h = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error'])
                # bu_errors[h_name] = extra_h['error']
                
                ##### bu errors are minimized explicitly
                hidden_states[h_name], extra_h = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error_']) # , 'inh', 'mix_layer'
                bu_errors[h_name] = extra_h['error_']

                ########### for viz
                # bu_errors_out.append(extra_h['error_'])
                # H_inh_out.append(extra_h['inh'])
                # mix_layer_out.append(extra_h['mix_layer'])
                ###########

                if 'hidden_errors' in extra:
                    hidden_errors[-1].append(bu_errors[h_name].mean().detach())

                ### R is passed up
                # x = hidden_states[h_name]
                ### E is passed up
                x = bu_errors[h_name]

                # if (x>10000).any():
                #     logger.info('variable %s at timestep %d out of bound'%(h_name,i))
                # x = F.relu_(x)
                
                if j < len(self.h_units_and_names)-1:
                    x = self.feedforward_units[j](x)

                current_loc = loc

            if self.cpc_gn:
                if i >= min(self.cpc_steps):
                    cpc_targets.append(x) #.detach()
                
                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        cpc_preds[step].append(self.spatial_transforms[step](x, self.feature_transforms[step](x)))

            if i <timesteps-1:
                for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                    loc = int(td_name.strip('topdown'))
                    h_name = 'horizontal'+str(loc)
                    
                    ### prediction is top down inh + errors 
                    # hidden_states[h_name], extra_td = td_unit(hidden_states[h_name], x, bu_errors[h_name], timestep=i, return_extra=['inh'])
                    # x = extra_td['inh']

                    ### prediction is top down inh + errors, bu errors are minimized
                    hidden_states[h_name], extra_td = td_unit(hidden_states[h_name], x, F.softplus(bu_errors[h_name]), timestep=i, return_extra=['inh'])
                    x = extra_td['inh']

                    ### prediction is top down inh + errors, no errors passed horiz
                    # hidden_states[h_name], extra_td = td_unit(hidden_states[h_name], x, None, timestep=i, return_extra=['inh'])
                    # x = extra_td['inh']
                    
                    ### prediction is top down rep + errors
                    # hidden_states[h_name] = td_unit(hidden_states[h_name], x, bu_errors[h_name], timestep=i)
                    # x = hidden_states[h_name]

                    ### prediction is top down rep + errors
                    # hidden_states[h_name] = td_unit(hidden_states[h_name], x, F.softplus(bu_errors[h_name]), timestep=i)
                    # x = hidden_states[h_name]
                    
                    # if (x>1e6).any():
                    #     logger.info('variable %s at timestep %d out of bound: %f'%(h_name,i, x.max().item()))

                if self.pred_gn:
                    # frame = self.final_remap(x)
                    # frame = self.output_conv(self.fb_warp(x,input_trans=self.final_remap(x)))
                    # frame = self.output_conv(self.final_remap(x))
                    
                    ### frame readout is used for training
                    frame = self.output_conv(x)

                    ### frame readout is used for vizualization
                    # frame = self.output_conv(x.detach())

                    if 'frames' in extra:
                        frames.append(frame)
                    
                    # pred_error = F.smooth_l1_loss(frame, inputs_[:,:,i+1])
                    
                    ### pred error on thresholded values
                    # pred_error = F.l1_loss(F.hardtanh(frame, 0,1), inputs_[:,:,i+1])

                    ### pred error on logits
                    pred_error = F.l1_loss(frame, inputs_[:,:,i+1])

                    errors = errors + pred_error
            #         if i==0:
            #             errors = errors + pred_error
            #         else:
            #             bu_pred_error = F.l1_loss(bu_errors['horizontal0'], torch.zeros_like(bu_errors['horizontal0']))
            #             errors = errors + pred_error + bu_pred_error
            # else:
            #     bu_pred_error = F.l1_loss(bu_errors['horizontal0'], torch.zeros_like(bu_errors['horizontal0']))
            #     errors = errors + bu_pred_error
            

            ########### for viz
            # for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
            #     hidden_out.append(hidden_states[h_name]) 
            ###########

            ### important
            conv_input = conv_input[:,:,1:]

        if self.run_classification:
            logits = self.head(hidden_states[self.h_units_and_names[-1][1]][:,:,None].detach()) 
            output['logits'] = logits
        # del hidden_states
        # del conv_input
        # del x
        if self.pred_gn:
            output['pred_errors'] = errors / (timesteps-1)
            if 'frames' in extra:
                frames = torch.stack(frames, 2)
                frames = F.relu(frames)
                frames[frames>1] = 1 
                output['frames'] = frames

        # if 'hidden_errors' in extra:
        #     output['hidden_errors'] = torch.Tensor(hidden_errors)

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

        # mix_layer_out = [torch.stack(mix_layer_out[i::5], 0) for  i in range(5)]
        # bu_errors_out = [torch.stack(bu_errors_out[i::5], 0) for  i in range(5)] #torch.stack(bu_errors_out, 1)
        # H_inh_out = [torch.stack(H_inh_out[i::5], 0) for  i in range(5)] # torch.stack(H_inh_out, 1)
        # hidden_out = [torch.stack(hidden_out[i::5], 0) for  i in range(5)] # torch.stack(hidden_out, 1)

        # output['mix_layer'] = mix_layer_out
        # output['bu_errors'] = bu_errors_out
        # output['H_inh'] = H_inh_out
        # output['hidden'] = hidden_out

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

        self.loc = nn.Sequential(
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
        xs = self.loc(x)
        xs = F.relu(F.max_pool2d(xs, kernel_size=xs.size()[2:]))

        xs = xs.view(-1, xs.shape[1])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input_trans.size(), align_corners=True)
        x = F.grid_sample(input_trans, grid, align_corners=True)

        return x

