
import numpy as np
from functools import partial
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import slowfast.utils.logging as logging

import kornia

from .batch_norm import GroupNorm, get_norm
from .build import MODEL_REGISTRY
from .gn_helper import ResNetSimpleHead
from .rnns import CBP_penalty, hConvGRUCell, tdConvGRUCell, tdConvGRUCell_err

# from .gn_helper import  TemporalCausalConv3d, \
#                         BasicBlock, \
#                         Bottleneck, \
#                         BasicStem, \
#                         ResNetSimpleHead, \
#                         SpatialTransformer, \
#                         conv1x3x3, \
#                         downsample_basic_block



logger = logging.get_logger(__name__)


__all__ = [
    "SmallGN",
]

@MODEL_REGISTRY.register()
class SmallGN(nn.Module):
    
    def __init__(self, cfg):
        super(SmallGN, self).__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        dropout_rate = cfg.MODEL.DROPOUT_RATE

        recurrent_bn = cfg.GN.RECURRENT_BN 
        feedforward_bn = cfg.GN.FEEDFORWARD_BN
        grad_method = 'bptt'
        
        layer_channels = cfg.PREDNET.LAYERS

        h_units = cfg.GN.HORIZONTAL_UNITS
        td_units = cfg.GN.TOPDOWN_UNITS
        gala = cfg.GN.GALA

        losses = {}
        for i in range(len(cfg.PREDNET.LOSSES[0])):
            losses[cfg.PREDNET.LOSSES[0][i]] = cfg.PREDNET.LOSSES[1][i]

        self.losses = losses
        
        self.evals = cfg.PREDNET.EVALS

        self.cpc_gn = cfg.PREDICTIVE.CPC

        self.hidden_init = cfg.GN.HIDDEN_INIT # 'learned'

        self.h_units = h_units
        self.td_units = td_units

        self.cbp_penalty = True
        self.cbp_cutoff = 0.9
        self.cbp_penalty_weight = 5e-6
        self.rnn_norm = True
        self.warp_td = False
        self.pixel_loss = F.l1_loss

        pred_fan = 1

        ########################################################
        
        assert len(td_units) <= len(h_units)-1, 'wrong number of horizontal or top-down units'
        h_locations = [h[0] for h in h_units]
        for td in td_units:
            assert td[0] in h_locations, 'no horizontal location found for td location' 

        fan_in = layer_channels[0]
        
        ### small expansion of the image
        
        self.stem = nn.Conv2d(pred_fan,fan_in,1) #24, 8, 3
        self._strides = [1]
        self._out_feature_channels = [fan_in] # [cfg.DATA.INPUT_CHANNEL_NUM[0]]
        
        self.feedforward_units = []
        self.h_units_and_names = []
        horizontal_layers = []

        for i, (_, ks) in enumerate(h_units):
            # loc, ks = h_params
            recurrent = hConvGRUCell(
                input_size=fan_in,
                hidden_size=fan_in,
                kernel_size=ks,
                batchnorm=self.rnn_norm, # True
                timesteps=cfg.DATA.NUM_FRAMES,
                gala=gala,
                spatial_kernel=3,
                less_softplus=False,
                r=4,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn,
                bottom_layer=(i==0))
            horizontal_name = "horizontal{}".format(i)
            self.add_module(horizontal_name, recurrent)
            self.h_units_and_names.append((recurrent, horizontal_name))
            horizontal_layers += [[horizontal_name, fan_in]]
            
            if i< len(h_units)-1:
                fan_out = layer_channels[i+1]
                
                feedforward = nn.Sequential(
                    nn.MaxPool2d(3, stride=2),
                    nn.Conv2d(fan_in, fan_out, 3, stride=1, padding=1),
                )
                self._out_feature_channels.append(fan_out)
                self._strides.append(self._strides[-1]*2)
                self.add_module('feedforward{}'.format(i), feedforward)
                self.feedforward_units.append(feedforward)
                fan_in = fan_out

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
                batchnorm=self.rnn_norm, # True
                timesteps=cfg.DATA.NUM_FRAMES,
                init=nn.init.orthogonal_,
                grad_method=grad_method,
                norm=recurrent_bn,
                spatial_transform=self.warp_td,
                gn_remap=False)

            topdown_name = "topdown{}".format(loc)
            self.add_module(topdown_name, recurrent)
            self.td_units_and_names.append((recurrent, topdown_name))
            topdown_layers += [[topdown_name, fan_in]]
            
            td_fan_in = fan_in
        
        fan_out = layer_channels[0]
        self.output_conv = nn.Sequential(
            nn.Conv2d(fan_out, fan_out, 1),
            nn.ReLU(),
            nn.Conv2d(fan_out, pred_fan, 1),
        )

        # if self.cpc_gn:
        #     fan_out = layer_channels[0]
        #     self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
        
        #     self.spatial_transforms = {}
        #     self.feature_transforms = {}
        #     for step in self.cpc_steps:
        #         ws = SpatialTransformer(fan_in=fan_out)
        #         wf = nn.Conv2d(fan_out, fan_out, kernel_size=1)
        #         self.add_module('bu_warp_%d'%step, ws)
        #         self.add_module('bu_feat_%d'%step, wf)
        #         self.spatial_transforms[step] = ws
        #         self.feature_transforms[step] = wf

        self.cpc_gn = cfg.PREDICTIVE.CPC
        if self.cpc_gn:
            self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
            
            fan_in = R_channels[-1] 
            
            wf = nn.Conv2d(fan_in, fan_in//4, kernel_size=1)
            init.orthogonal_(wf.weight)
            init.constant_(wf.bias, 0)

            self.add_module('cpc_target_layer', wf)

            self.cpc_pred_layer = {}
            for step in self.cpc_steps:
                wf = nn.Conv2d(fan_in, fan_in//4, kernel_size=1)
                init.orthogonal_(wf.weight)
                init.constant_(wf.bias, 0)

                self.add_module('cpc_pred_layer_%d'%step, wf)
                self.cpc_pred_layer[step] = wf

        self.group_norm_layers=[]
         
        for name, m in self.named_modules():
            if 'horizontal' not in name and 'topdown' not in name:
                if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)): 
                    nn.init.orthogonal_(m.weight)                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, GroupNorm)): # , nn.GroupNorm 
                    nn.init.constant_(m.bias, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            if isinstance(m, GroupNorm):
                self.group_norm_layers.append(m)


    def forward(self, inputs_, extra=[], autoreg=False):
        if isinstance(inputs_, list):
            inputs_ = inputs_[0]

        timesteps = inputs_.size(2)
        
        for gnl in self.group_norm_layers:
            gnl.reset_stats()

        current_loc = 0
        aug_inputs = []

        conv_input = inputs_

        

        ############################
        # FeedForward loss variables
        ############################
        outputs = {}
        
        if 'CPC' in self.losses:
            cpc_targets = []
            cpc_preds = {step: [] for step in self.cpc_steps}
        
            outputs['cpc_targets'] = cpc_targets
            outputs['cpc_preds'] = cpc_preds
        
        hidden_states = {}
        bu_errors = {}
        # hidden_errors = []

        if self.pred_gn:
            frame_errors = 0
            errors = 0
            if 'frames' in extra:
                frames = []
        
        if self.cpc_gn:
            cpc_targets = []
            cpc_preds = {step: [] for step in self.cpc_steps}
        
        output = {}

        ##########################################################################################
        ### hidden init
        
        if self.activity_shapes == {}:
            x = self.stem(conv_input[:,:,0])
            x = F.softplus(torch.ones_like(x)/2.0)
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                hidden_states[h_name] = x
                self.activity_shapes[h_name] = list(x.shape)
                if j < len(self.h_units_and_names)-1:
                    x = F.softplus(torch.ones_like(self.feedforward_units[j](x))/2.0)

        else:
            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                hidden_states[h_name] = F.softplus(conv_input.new_ones(self.activity_shapes[h_name])/2.0)
        
        ##########################################################################################
        # gammanet loop

        for i in range(timesteps):
            
            for j, (td_unit, td_name) in enumerate(self.td_units_and_names):
                loc = int(td_name.strip('topdown'))
                h_name = 'horizontal'+str(loc)
                
                hidden_states[h_name] = td_unit(hidden_states[h_name], x, timestep=i, return_extra=[]) # , extra_td , 'inh', 'mix_layer'
                
                x = hidden_states[h_name]

            
            bottom_h = self.output_conv(x)
            
            gt_frame = conv_input[:,:,i]
            frame = torch.argmax(bottom_h, dim=1)[:,None] if bottom_h.shape[1] != 1 else bottom_h

            # if autoreg and i>timesteps//2:
            #     x = frame
            # else:
            #     x = conv_input[:,:,i]
            
            # if hidden_states['horizontal0'].requires_grad and self.cbp_penalty and i == timesteps-2:
            #     prev_hidden = hidden_states['horizontal0']
                
            # if hidden_states['horizontal0'].requires_grad and self.cbp_penalty and i == timesteps-1:
            #     cbp_penalty = CBP_penalty(hidden_states['horizontal0'],
            #                                 prev_hidden,
            #                                 tau=self.cbp_cutoff) 

            # if autoreg and i>timesteps//2:
            #     x = torch.zeros_like(frame)
            #     # x = frame
            # else:

            x = conv_input[:,:,i] - frame
            # x = conv_input[:,:,i]
            
            x = self.stem(x)

            for j, (h_unit, h_name) in enumerate(self.h_units_and_names):
                loc = int(h_name.strip('horizontal'))
                 
                ##### bu errors are minimized explicitly
                # if i>0 and j==0 and self.pred_gn:
                #     hidden_states[h_name], extra_h = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error_', 'inh']) # , 'mix_layer'
                #     bu_errors[h_name] = extra_h['error_']

                #     # frame = F.hardtanh(extra_h['inh'], 0, 1)
                #     frame = F.hardtanh(self.output_conv(extra_h['inh'].detach()), 0, 1)
                    
                #     bu_pred_error = F.l1_loss(frame, inputs_[:,:,i])
                
                #     errors = errors + bu_pred_error

                #     if 'frames' in extra:
                #         frames.append(frame)

                #     frame_error = F.mse_loss(frame, inputs_[:,:,i])

                #     frame_errors = frame_errors + frame_error
                # else:

                hidden_states[h_name], extra_h = h_unit(x, hidden_states[h_name], timestep=i, return_extra=['error_']) # , 'mix_layer'
                x = extra_h['error_']
                
                bu_errors[h_name] = x.mean([-1,-2])

                if j < len(self.h_units_and_names)-1:
                    x = self.feedforward_units[j](x)

                if i>0:
                    bu_pred_error = self.pixel_loss(x, torch.zeros_like(x))
                    if j> 0: 
                        errors = errors + bu_pred_error*0.2
                    else:
                        errors = errors + bu_pred_error
            
            ############################
            # FeedForward loss variables
            ############################
            R = R_top 
            if 'CPC' in self.losses:
                # CPC between R:t and R:t+4
                if i >= min(self.cpc_steps)+1:
                    outputs['cpc_targets'].append(self.cpc_target_layer(R))
                    # print('targ', i)

                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        # print('preds', i, step)
                        outputs['cpc_preds'][step].append(self.cpc_pred_layer[step](R))

            ###########################
            # Gather loop variables
            ###########################
            outputs = self.get_loss_t(outputs, t, bu_errors, A_0, A_h_0, extra, time_steps)
        
        ###########################
        # Calculate loss and process output
        ###########################
        outputs = self.get_loss(input, outputs, extra)

        # output['pred_errors'] = errors / (timesteps-1)
        # output['frame_errors'] = frame_errors / (timesteps-1)
        # if 'frames' in extra:
        #     frames = torch.stack(frames, 2)
        #     # frames = F.relu(frames)
        #     # frames[frames>1] = 1 
        #     output['frames'] = frames

        # if self.cpc_gn:
        #     cpc_loss = 0
        #     cpc_targets = torch.stack(cpc_targets,-1).transpose(1,4)
            
        #     for step in self.cpc_steps:
        #         if len(cpc_preds[step])>1:
        #             cpc_preds[step] = torch.stack(cpc_preds[step], -1).transpose(1,4)
        #             # .permute(1,0,3,4,2) #T B C H W -> B T H W C
        #             # logger.info(cpc_preds[step].shape)
        #             cpc_preds[step] = cpc_preds[step].reshape([-1,cpc_preds[step].shape[-1]]) # -> N C
        #             # logger.info(cpc_targets[:,step-min(self.cpc_steps):].shape)
        #             cpc_output = torch.matmul(cpc_targets[:,step-min(self.cpc_steps):].reshape([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

        #             labels = torch.cumsum(torch.ones_like(cpc_preds[step][:,0]).long(), 0) -1
        #             cpc_loss = cpc_loss + F.cross_entropy(cpc_output, labels)

        #     output['cpc_loss'] = cpc_loss

        return outputs
    
    def get_loss_t(self, outputs, i, E_seq, frame, A_hat, extra, timesteps):
        
        if i == 0:
            
            ############################
            # Feedback loss variables
            ############################
            if 'CrossEntropy' in self.losses:
                outputs['ce_loss'] = []
            if 'FocalLoss' in self.losses:
                outputs['focal_loss'] = []
            if 'FocalLoss' in self.losses:
                outputs['focal_loss'] = []
            if 'BinaryCrossEntropy' in self.losses:
                outputs['bce_loss'] = []
            if 'WeightedBinaryCrossEntropy' in self.losses:
                outputs['bce_loss'] = []
            if 'L1Loss' in self.losses:
                outputs['errors'] = []
            
            ############################
            # Evaluation
            ############################
            outputs['mse'] = []
            outputs['IoU'] = []
            outputs['prec'] = []
            outputs['recall'] = []
            outputs['f1s'] = []
            outputs['balacc'] = []
            outputs['frames'] = []

        else:
            ############################
            # Feedback loss variables
            ############################
            if 'CrossEntropy' in self.losses:
                pos = ((frame>=0.2)*1).sum()
                neg = ((frame<0.2)*1).sum()
                pos = pos/(pos+neg)
                neg = neg/(pos+neg)
                fb_loss = F.cross_entropy(A_hat.permute([0,2,3,1]).reshape([-1,2]), ((frame<0.2)*1).long().flatten(), weight=A_hat.new([pos, neg]))
                outputs['ce_loss'].append(fb_loss)
                
            if 'FocalLoss' in self.losses:
                fb_loss = losses.focal_loss(A_hat, (frame>=0.2)*1, gamma=2.0, reduce=True)
                outputs['focal_loss'].append(fb_loss)

            if 'BinaryCrossEntropy' in self.losses:
                fb_loss = F.binary_cross_entropy_with_logits(A_hat, frame, reduction='mean')
                outputs['bce_loss'].append(fb_loss)
            
            if 'WeightedBinaryCrossEntropy' in self.losses:    
                fb_loss = losses.weighted_bce_with_logits(A_hat, frame, reduction='mean')
                outputs['bce_loss'].append(fb_loss)

            if 'L1Loss' in self.losses:
                # fb_loss = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for k,e in E_seq.keys()], 1)
                fb_loss = torch.Tensor([e for k,e in E_seq.keys()]).mean()
                outputs['errors'].append(fb_loss)
            
            
            ############################
            # Auxilary losses
            ############################
            A_hat = A_hat.data #detach()
            frame = frame.data #detach()

            ############################
            # Evaluation
            ############################

            if 'FocalLoss' in self.losses or 'CrossEntropy' in self.losses:
                outputs['mse'].append(F.mse_loss(F.softmax(A_hat, dim=1)[:,1][:,None], frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                balacc, precision, recall, f1s = losses.acc_scores((frame>0.2).long(), A_hat)

                A_hat = torch.argmax(A_hat, dim=1)[:,None].float()
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                
            elif 'BinaryCrossEntropy' in self.losses or 'WeightedBinaryCrossEntropy' in self.losses:
                A_hat = torch.sigmoid(A_hat)
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))

                balacc, precision, recall, f1s = losses.metric_scores((frame>0.2).long().byte(), (A_hat>0.2).long().byte())
                balacc = balacc * 100
            else:
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                balacc, precision, recall, f1s = losses.metric_scores((frame>0.2).long().byte(), (A_hat>0.2).long().byte())
                balacc = balacc * 100

            outputs['prec'].append(precision)
            outputs['recall'].append(recall)
            outputs['f1s'].append(f1s)
            outputs['balacc'].append(balacc)

            if 'frames' in extra:
                outputs['frames'].append(A_hat)

        return outputs

    def get_loss(self, input, outputs, extra):

        output = {}
        total_loss = 0
        ############################
        # Feedback Loss
        ############################ 
        if 'CrossEntropy' in self.losses:
            output['ce_loss'] = torch.stack(outputs['ce_loss']).mean()
            total_loss += output['ce_loss'] * self.losses['CrossEntropy']
            
        if 'FocalLoss' in self.losses:
            output['focal_loss'] = torch.stack(outputs['focal_loss']).mean()
            total_loss += output['focal_loss'] * self.losses['FocalLoss']
       
        if 'BinaryCrossEntropy' in self.losses:
            output['bce_loss'] = torch.stack(outputs['bce_loss']).mean()
            total_loss += output['bce_loss'] * self.losses['BinaryCrossEntropy']
        
        if 'WeightedBinaryCrossEntropy' in self.losses:
            output['bce_loss'] = torch.stack(outputs['bce_loss']).mean()
            total_loss += output['bce_loss'] * self.losses['WeightedBinaryCrossEntropy']
        
        if 'L1Loss' in self.losses:
            output['errors'] = torch.stack(outputs['errors'], 2)*torch.Tensor([1]+[0.1]*(self.n_layers-1)).to(outputs['errors'][0].device)[None,:,None]
            output['errors'] = output['errors'].sum(1).mean()
            total_loss += output['errors'] * self.losses['L1Loss']

        ############################
        # Feedforward Loss
        ############################ 
        if 'CPC' in self.losses:
            cpc_loss = losses.cpc_loss(outputs['cpc_preds'], outputs['cpc_targets'])
            output['cpc_loss'] = cpc_loss
            total_loss += output['cpc_loss'] * self.losses['CPC']

        ############################
        # Auxilary Losses
        ############################ 
        output['total_loss'] = total_loss
        
        ############################
        # Evaluation
        ############################ 
        output['mse'] = torch.stack(outputs['mse'], 1).mean() - F.mse_loss(input[:,:,2:], input[:,:,1:-1])
        output['IoU'] = torch.stack(outputs['IoU'], 1).float().mean() #
        
        output['prec'] = torch.stack(outputs['prec'], 0).float().mean()
        output['recall'] = torch.stack(outputs['recall'], 0).float().mean()
        output['f1s'] = torch.stack(outputs['f1s'], 0).float().mean()
        output['balacc'] = torch.stack(outputs['balacc'], 0).float().mean()


        if 'frames' in extra:
            output['frames'] = torch.stack(outputs['frames'], 2)
        
        return output
    

# differentiable warping + feature transformation
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


logger = logging.get_logger(__name__)


__all__ = [
    "SmallPredNet",
]

def get_recurrent_cell(name):
    return getattr(rnns, name)


@MODEL_REGISTRY.register()
class SmallPredNet(nn.Module):
    def __init__(self, cfg):
        super(SmallPredNet, self).__init__()
        
        losses = {}
        for i in range(len(cfg.PREDNET.LOSSES[0])):
            losses[cfg.PREDNET.LOSSES[0][i]] = cfg.PREDNET.LOSSES[1][i]

        self.losses = losses
        
        self.evals = cfg.PREDNET.EVALS

        rnn_cell_name = cfg.PREDNET.CELL

        R_channels = cfg.PREDNET.LAYERS # (3, 48, 96, 192)
        A_channels = cfg.PREDNET.LAYERS # (3, 48, 96, 192)
        A_channels[0] = 1

        # print(self.losses, self.evals)
        
        # self.losses={
        #     'FocalLoss': 1,
        #     'CPC': 1e-2,
        #     'smooth_l1_loss': 1,
        # }
        # self.evals=[
        #     'mse',
        #     'Acc',
        #     'IoU'
        # ]
        
        self.r_channels = R_channels + [0] # # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)

        rnn_cell = get_recurrent_cell(rnn_cell_name)

        # top_losses = ['', ]
        # bottom_losses

        for i in range(self.n_layers):
            cell = rnn_cell(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            if i == 0:
                if 'FocalLoss' in self.losses or 'CrossEntropy' in self.losses:
                    fan_out = 2

                    # temporary
                    conv = nn.Sequential(
                        nn.Conv2d(self.r_channels[i], self.r_channels[i], 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(self.r_channels[i], fan_out, 1),
                    )
                    
                    init.orthogonal_(conv[0].weight)
                    init.constant_(conv[0].bias, 0)

                    init.xavier_normal_(conv[2].weight)
                    init.constant_(conv[2].bias, 0)

                elif 'BinaryCrossEntropy' in self.losses or 'WeightedBinaryCrossEntropy' in self.losses:
                    fan_out = 1
                    conv = nn.Conv2d(self.r_channels[i], fan_out, 3, padding=1) #2 for focal loss
                    init.xavier_normal_(conv.weight)
                    # init.constant_(conv.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))
                    init.constant_(conv.bias, 0)

                else:
                    conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU()) #2 for focal loss
                    conv.add_module('satlu', SatLU())
                    init.orthogonal_(conv[0].weight)
                    init.constant_(conv[0].bias, 0)

            else:
                conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
                init.orthogonal_(conv[0].weight)
                init.constant_(conv[0].bias, 0)

            setattr(self, 'conv{}'.format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            init.orthogonal_(update_A[0].weight)
            init.constant_(update_A[0].bias, 0)

            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

        self.cpc_gn = cfg.PREDICTIVE.CPC
        if self.cpc_gn:
            self.cpc_steps = cfg.PREDICTIVE.CPC_STEPS
            
            fan_in = R_channels[-1] 
            
            wf = nn.Conv2d(fan_in, fan_in//4, kernel_size=1)
            init.orthogonal_(wf.weight)
            init.constant_(wf.bias, 0)

            self.add_module('cpc_target_layer', wf)

            self.cpc_pred_layer = {}
            for step in self.cpc_steps:
                wf = nn.Conv2d(fan_in, fan_in//4, kernel_size=1)
                init.orthogonal_(wf.weight)
                init.constant_(wf.bias, 0)

                self.add_module('cpc_pred_layer_%d'%step, wf)
                self.cpc_pred_layer[step] = wf
                
    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input, meta=None, extra=[], autoreg=False):
        
        ###########################
        # Initialize
        ###########################
        
        if isinstance(input, list):
            input = input[0]

        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = input.new_zeros([batch_size, 2*self.a_channels[l], w, h])
            R_seq[l] = input.new_zeros([batch_size, self.r_channels[l], w, h])
            
            w = w//2
            h = h//2

        time_steps = input.size(2)
        total_error = []
        frames = []
        frame_error = 0

        outputs = {}
        
        ###########################
        # Loop
        ###########################
        
        for t in range(time_steps):
            A = input[:,:,t] #t
            A = A.float()
            
            ###########################
            # Top Down
            ###########################
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx

                if l == self.n_layers-1:
                    R_top = R

            ###########################
            # Bottom up
            ###########################
            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    A_h_0 = A_hat
                    A_0 = A
                    A_hat = torch.argmax(A_hat, dim=1)[:,None] if A_hat.shape[1] != 1 else A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)

            ############################
            # FeedForward loss variables
            ############################
            R = R_top 
            if 'CPC' in self.losses:
                # CPC between R:t and R:t+4
                if i >= min(self.cpc_steps)+1:
                    outputs['cpc_targets'].append(self.cpc_target_layer(R))
                    # print('targ', i)

                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        # print('preds', i, step)
                        outputs['cpc_preds'][step].append(self.cpc_pred_layer[step](R))

            ###########################
            # Gather loop variables
            ###########################
            outputs = self.get_loss_t(outputs, t, E_seq, A_0, A_h_0, extra, time_steps)
        
        ###########################
        # Calculate loss and process output
        ###########################
        outputs = self.get_loss(input, outputs, extra)

        return outputs

    def get_loss_t(self, outputs, i, E_seq, frame, A_hat, extra, timesteps):
        
        if i == 0:
            ############################
            # FeedForward loss variables
            ############################
            if 'CPC' in self.losses:
                cpc_targets = []
                cpc_preds = {step: [] for step in self.cpc_steps}
            
                outputs['cpc_targets'] = cpc_targets
                outputs['cpc_preds'] = cpc_preds
            
            ############################
            # Feedback loss variables
            ############################
            if 'CrossEntropy' in self.losses:
                outputs['ce_loss'] = []
            if 'FocalLoss' in self.losses:
                outputs['focal_loss'] = []
            if 'BinaryCrossEntropy' in self.losses:
                outputs['bce_loss'] = []
            if 'WeightedBinaryCrossEntropy' in self.losses:
                outputs['bce_loss'] = []
            if 'L1Loss' in self.losses:
                outputs['errors'] = []
            
            ############################
            # Evaluation
            ############################
            outputs['mse'] = []
            outputs['IoU'] = []
            outputs['prec'] = []
            outputs['recall'] = []
            outputs['f1s'] = []
            outputs['balacc'] = []
            outputs['frames'] = []

        else:
            ############################
            # Feedback loss variables
            ############################
            if 'CrossEntropy' in self.losses:
                pos = ((frame>=0.2)*1).sum()
                neg = ((frame<0.2)*1).sum()
                pos = pos/(pos+neg)
                neg = neg/(pos+neg)
                fb_loss = F.cross_entropy(A_hat.permute([0,2,3,1]).reshape([-1,2]), ((frame<0.2)*1).long().flatten(), weight=A_hat.new([pos, neg]))
                outputs['ce_loss'].append(fb_loss)
                
            if 'FocalLoss' in self.losses:
                fb_loss = losses.focal_loss(A_hat, (frame>=0.2)*1, gamma=2.0, reduce=True)
                outputs['focal_loss'].append(fb_loss)

            if 'BinaryCrossEntropy' in self.losses:
                fb_loss = F.binary_cross_entropy_with_logits(A_hat, frame, reduction='mean')
                outputs['bce_loss'].append(fb_loss)
            
            if 'WeightedBinaryCrossEntropy' in self.losses:    
                fb_loss = losses.weighted_bce_with_logits(A_hat, frame, reduction='mean')
                outputs['bce_loss'].append(fb_loss)

            if 'L1Loss' in self.losses:
                fb_loss = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                outputs['errors'].append(fb_loss)
            
            
            ############################
            # Auxilary losses
            ############################
            A_hat = A_hat.data #detach()
            frame = frame.data #detach()

            ############################
            # Evaluation
            ############################

            if 'FocalLoss' in self.losses or 'CrossEntropy' in self.losses:
                outputs['mse'].append(F.mse_loss(F.softmax(A_hat, dim=1)[:,1][:,None], frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                balacc, precision, recall, f1s = losses.acc_scores((frame>0.2).long(), A_hat)

                A_hat = torch.argmax(A_hat, dim=1)[:,None].float()
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                
            elif 'BinaryCrossEntropy' in self.losses or 'WeightedBinaryCrossEntropy' in self.losses:
                A_hat = torch.sigmoid(A_hat)
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))

                balacc, precision, recall, f1s = losses.metric_scores((frame>0.2).long().byte(), (A_hat>0.2).long().byte())
                balacc = balacc * 100
            else:
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                balacc, precision, recall, f1s = losses.metric_scores((frame>0.2).long().byte(), (A_hat>0.2).long().byte())
                balacc = balacc * 100

            outputs['prec'].append(precision)
            outputs['recall'].append(recall)
            outputs['f1s'].append(f1s)
            outputs['balacc'].append(balacc)

            if 'frames' in extra:
                outputs['frames'].append(A_hat)

        return outputs

    def get_loss(self, input, outputs, extra):

        output = {}
        total_loss = 0
        ############################
        # Feedback Loss
        ############################ 
        if 'CrossEntropy' in self.losses:
            output['ce_loss'] = torch.stack(outputs['ce_loss']).mean()
            total_loss += output['ce_loss'] * self.losses['CrossEntropy']
            
        if 'FocalLoss' in self.losses:
            output['focal_loss'] = torch.stack(outputs['focal_loss']).mean()
            total_loss += output['focal_loss'] * self.losses['FocalLoss']
       
        if 'BinaryCrossEntropy' in self.losses:
            output['bce_loss'] = torch.stack(outputs['bce_loss']).mean()
            total_loss += output['bce_loss'] * self.losses['BinaryCrossEntropy']

        if 'WeightedBinaryCrossEntropy' in self.losses:
            output['bce_loss'] = torch.stack(outputs['bce_loss']).mean()
            total_loss += output['bce_loss'] * self.losses['WeightedBinaryCrossEntropy']
            
        if 'L1Loss' in self.losses:
            output['errors'] = torch.stack(outputs['errors'], 2)*torch.Tensor([1]+[0.1]*(self.n_layers-1)).to(outputs['errors'][0].device)[None,:,None]
            output['errors'] = output['errors'].sum(1).mean()
            total_loss += output['errors'] * self.losses['L1Loss']

        ############################
        # Feedforward Loss
        ############################ 
        if 'CPC' in self.losses:
            cpc_loss = losses.cpc_loss(outputs['cpc_preds'], outputs['cpc_targets'])
            output['cpc_loss'] = cpc_loss
            total_loss += output['cpc_loss'] * self.losses['CPC']

        ############################
        # Auxilary Losses
        ############################ 
        output['total_loss'] = total_loss
        
        ############################
        # Evaluation
        ############################ 
        output['mse'] = torch.stack(outputs['mse'], 1).mean() - F.mse_loss(input[:,:,2:], input[:,:,1:-1])
        output['IoU'] = torch.stack(outputs['IoU'], 1).float().mean() #
        
        output['prec'] = torch.stack(outputs['prec'], 0).float().mean()
        output['recall'] = torch.stack(outputs['recall'], 0).float().mean()
        output['f1s'] = torch.stack(outputs['f1s'], 0).float().mean()
        output['balacc'] = torch.stack(outputs['balacc'], 0).float().mean()


        if 'frames' in extra:
            output['frames'] = torch.stack(outputs['frames'], 2)
        
        return output
    