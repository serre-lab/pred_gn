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
                        conv1x3x3, \
                        downsample_basic_block
                        
import kornia
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

from .build import MODEL_REGISTRY

__all__ = [
    "GN_SEG",
]

@MODEL_REGISTRY.register()
class GN_SEG(nn.Module):
    
    def __init__(self, cfg):
        super(GN_SEG, self).__init__()

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

        n_seg_classes = cfg.MODEL.SEG_CLASSES

        self.hidden_init = cfg.GN.HIDDEN_INIT # 'learned'

        self.h_units = h_units
        self.td_units = td_units

        ########################################################
        
        assert len(td_units) <= len(h_units)-1, 'wrong number of horizontal or top-down units'
        h_locations = [h[0] for h in h_units]
        for td in td_units:
            assert td[0] in h_locations, 'no horizontal location found for td location' 
        
        self.color_aug = kornia.augmentation.ColorJitter(brightness= 0.1, contrast= 0.1, saturation = 0.1, hue = 0.2, return_transform=True)
        self.affine_aug = kornia.augmentation.RandomAffine(degrees=(-3,3), translate=(5/224,5/224), scale=(0.9, 1.1), shear=(-0.02,0.02), return_transform=True)
        
        ### small expansion of the image
        self.stem = nn.Conv2d(3,24,3,padding=1)
        self._strides = [1]
        self._out_feature_channels = [24] # [cfg.DATA.INPUT_CHANNEL_NUM[0]]
        
        fan_in = 24 # cfg.DATA.INPUT_CHANNEL_NUM[0] # 3
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
                norm=recurrent_bn,
                bottom_layer=False)
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
            
            recurrent = tdConvGRUCell(
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
                spatial_transform=False,
                gn_remap=False)

            topdown_name = "topdown{}".format(loc)
            self.add_module(topdown_name, recurrent)
            self.td_units_and_names.append((recurrent, topdown_name))
            topdown_layers += [[topdown_name, fan_in]]
            
            td_fan_in = fan_in

        pred_fan = self._out_feature_channels[self.h_units[0][0]]
        
        # if self.pred_gn:
        #     # self.output_conv = nn.Conv2d(pred_fan, cfg.DATA.INPUT_CHANNEL_NUM[0], 3, padding=1)
        #     self.output_conv = nn.Sequential(
        #         nn.Conv2d(pred_fan, pred_fan, 3, padding=1),
        #         get_norm(feedforward_bn, pred_fan), #+'2D'
        #         nn.ReLU(),
        #         nn.Conv2d(pred_fan, cfg.DATA.INPUT_CHANNEL_NUM[0], 3, padding=1),
        #     )
        
        self.output_seg = nn.Sequential(
                nn.Conv2d(pred_fan, pred_fan, 3, padding=1),
                get_norm(feedforward_bn, pred_fan), #+'2D'
                nn.ReLU(),
                nn.Conv2d(pred_fan, pred_fan, 3, padding=1),
                get_norm(feedforward_bn, pred_fan), #+'2D'
                nn.ReLU(),
                nn.Conv2d(pred_fan, n_seg_classes, 1)
            )

        if self.cpc_gn:
            self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
            fan_in = self._out_feature_channels[self.h_units[-1][0]] 

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
        inputs_, masks = inputs_
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]

        
        for gnl in self.group_norm_layers:
            gnl.reset_stats()

        current_loc = 0
        aug_inputs = []
        
        timesteps = inputs_.size(2)

        conv_input = inputs_

        hidden_states = {}
        bu_errors = {}

        # mix_layer_out = []
        # bu_errors_out =[]
        # H_inh_out = []
        # hidden_out = []

        if self.pred_gn:
            frame_errors = 0
            errors = 0

            acc = 0
            iou = 0
            if 'frames' in extra:
                frames = []
        if self.cpc_gn:
            cpc_targets = []
            cpc_preds = {step: [] for step in self.cpc_steps}
        
        output = {}

        # if 'input_aug' in extra:
            # output['input_aug'] = aug_inputs

        ##########################################################################################
        # gammanet loop

        for i in range(timesteps):
            
            x = conv_input[:,:,i]
            
            x = self.stem(x)

            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                 
                ##### bu errors are minimized explicitly
                # if i>0 and j==0 and self.pred_gn:
                #     hidden_states[h_name], extra_h = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error_', 'inh']) # , 'mix_layer'
                #     bu_errors[h_name] = extra_h['error_']

                #     # frame = F.hardtanh(extra_h['inh'], 0, 1)
                #     frame = F.hardtanh(self.output_conv(extra_h['inh']), 0, 1)
                    
                #     bu_pred_error = F.l1_loss(frame, inputs_[:,:,i])
                
                #     errors = errors + bu_pred_error

                #     if 'frames' in extra:
                #         frames.append(frame)

                #     frame_error = F.mse_loss(frame, inputs_[:,:,i])

                #     frame_errors = frame_errors + frame_error
                # else:

                if i == 0:
                    hidden_states[h_name] = F.softplus(torch.zeros_like(x))
                
                hidden_states[h_name] = h_unit(x, hidden_states[h_name], timestep=i, return_extra=[]) # 'error_' , 'mix_layer'
                # bu_errors[h_name] = extra_h['error_']

                x = hidden_states[h_name]
                if j < len(self.h_units_and_names)-1:
                    x = self.feedforward_units[j](x)

            
            for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                loc = int(td_name.strip('topdown'))
                h_name = 'horizontal'+str(loc)
                
                hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i, return_extra=[]) # , extra_td , 'inh', 'mix_layer'
                
                x = hidden_states[h_name]

            output_seg = self.output_seg(x)

            
            classes, weight = masks[:,:,i].unique(return_counts=True)
            if classes.shape[0]< output_seg.shape[1]:
                weights = torch.zeros([output_seg.shape[1]])
                k = 0
                for cl in classes:
                    weights[cl.long()] = weight[k].float()
                    k+=1
            else:
                weights = weight.float()
            weights = 1/weights
            weights = weights/weights.sum()

            seg_loss = F.cross_entropy(output_seg, masks[:,:,i].long().squeeze(1), weight = weights.to(output_seg.device))
            errors = errors + seg_loss
            
            pred = output_seg.max(1)[1]

            iou = iou + intersectionAndUnion(pred.long(), masks[:,:,i].long().squeeze(1), output_seg.shape[1])

            acc = acc + accuracy(pred.long(), masks[:,:,i].long().squeeze(1))

            if 'frames' in extra:
                frames.append((pred[:,None].float()/pred.max()))

            if self.cpc_gn:
                if i >= min(self.cpc_steps):
                    cpc_targets.append(x)
                
                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        cpc_preds[step].append(self.spatial_transforms[step](x, self.feature_transforms[step](x)))

        if self.run_classification:
            logits = self.head(hidden_states[self.h_units_and_names[-1][1]][:,:,None].detach()) 
            output['logits'] = logits

        if self.pred_gn:
            output['pred_errors'] = errors / (timesteps-1)
            output['Acc'] = acc / (timesteps-1)
            output['IoU'] = iou.float() / (timesteps-1)

            # output['frame_errors'] = frame_errors / (timesteps-1)
            if 'frames' in extra:
                frames = torch.stack(frames, 2)
            #     # frames = F.relu(frames)
            #     # frames[frames>1] = 1 
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

        # logger.info(len(mix_layer_out))
        # mix_layer_out = [torch.stack(mix_layer_out[i::3], 0) for  i in range(3)]
        # bu_errors_out = [torch.stack(bu_errors_out[i::4], 0) for  i in range(4)] #torch.stack(bu_errors_out, 1)
        # H_inh_out = [torch.stack(H_inh_out[i::4], 0) for  i in range(4)] # torch.stack(H_inh_out, 1)
        # hidden_out = [torch.stack(hidden_out[i::4], 0) for  i in range(4)] # torch.stack(hidden_out, 1)

        # output['mix_layer'] = mix_layer_out
        # output['bu_errors'] = bu_errors_out
        # output['H_inh'] = H_inh_out
        # output['hidden'] = hidden_out

        # {'logits': output, 
        # 'cpc_loss': cpc_loss, 
        # 'pred_errors': errors}
        return output

def accuracy(preds, label):
    acc_sum = (1 * (preds == label)).sum() / (np.prod(label.shape) + 1e-10) 
    return acc_sum

def intersectionAndUnion(imPred, imLab, numClass):

    score = 0
    n_ = 0
    for i in range(numClass):
        l = (imLab == i)
        # logger.info('label')
        # logger.info(l.any())
        if l.any():
            p = (imPred == i)
            # logger.info('pred')
            # logger.info(p.any())

            intersection = (p == l)*1 * ((p > 0)*1)
            union = (p != l)*1 + intersection

            score = score + intersection.sum().float()/union.sum().float()
            
            n_ += 1
    
    return score / n_


# differentiable warping + feature transformation
# class SpatialTransformer(nn.Module):
#     def __init__(self, fan_in):
#         super(SpatialTransformer, self).__init__()

#         # Spatial transformer localization-network

#         self.loc = nn.Sequential(
#             nn.Conv2d(fan_in, 64, kernel_size=5),
#             #nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(64, 32, kernel_size=3),
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(32, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#     # Spatial transformer network forward function
#     def forward(self, x, input_trans=None):
#         if input_trans == None:
#             input_trans = x 
#         xs = self.loc(x)
#         xs = F.relu(F.max_pool2d(xs, kernel_size=xs.size()[2:]))

#         xs = xs.view(-1, xs.shape[1])
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, input_trans.size(), align_corners=True)
#         x = F.grid_sample(input_trans, grid, align_corners=True)

#         return x

class SpatialTransformer(nn.Module):
    def __init__(self, fan_in):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network

        
        self.loc = nn.Sequential(
            nn.Conv2d(fan_in, 32, kernel_size=3, padding=1, bias=False),
            #nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32*4*4, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        nn.init.xavier_normal_(self.loc[0].weight)
        if self.loc[0].bias is not None:
            self.loc[0].bias.data.zero_()
        
        nn.init.xavier_normal_(self.fc_loc[0].weight)
        self.fc_loc[0].bias.data.zero_()
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x, input_trans=None):
        if input_trans == None:
            input_trans = x 
        xs = self.loc(x)

        xs = F.relu(F.adaptive_max_pool2d(xs, output_size=(4,4)))

        xs = xs.view(xs.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input_trans.size(), align_corners=True)
        x = F.grid_sample(input_trans, grid, align_corners=True)

        return x

