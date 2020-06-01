import math
import torch
import torch.nn as nn
from torch.nn import init

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY

from slowfast.models import rnns
from slowfast.models import losses

from .rnns import (hConvGRUCell, 
                    ConvLSTMCell, 
                    ConvLSTMCell_C, 
                    ConvLSTMCell_CG1x1, 
                    ConvLSTMCell_CGpool, 
                    ConvLSTMCell_CG1x1_noI,
                    ConvLSTMCell_CG1x1_noF,
                    ConvLSTMCell_CG1x1_noO )

# import slowfast.models.losses as losses
# import slowfast.models.rnns as rnns



# import kornia

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
        
        # all configs:
        # layers and filters
        # using E+/- or just E
        # loss and weights 
        # eval 
        
        # cfg.PREDNET.LAYERS
        # cfg.PREDNET.LOSSES
        
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

                    # nn.init.xavier_uniform_(conv[2].weight)
                    # nn.init.normal_(conv[2].weight, std=0.001)
                    # nn.init.constant_(conv[2].bias, 0)
                    
                    # init.constant_(conv[2].bias[0], -0.1)
                    # init.constant_(conv[2].bias[1], 0.1)


                    # # original 
                    # conv = nn.Conv2d(self.r_channels[i], fan_out, 3, padding=1) #2 for focal loss
                    # init.xavier_normal_(conv.weight)
                    # init.constant_(conv.bias, 0)

                    # init.constant_(conv.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

                    # init.constant_(conv.bias[0], torch.log(torch.tensor((1 - 0.01) / 0.01)))
                    # init.constant_(conv.bias[1], 0)

                elif 'BinaryCrossEntropy' in self.losses:
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
                
                
            
            # self.spatial_transforms = {}
            # self.feature_transforms = {}
            # for step in self.cpc_steps:
            #     ws = SpatialTransformer(fan_in=fan_in)
            #     wf = nn.Conv2d(fan_in, fan_in, kernel_size=1)
            #     self.add_module('bu_warp_%d'%step, ws)
            #     self.add_module('bu_feat_%d'%step, wf)
            #     self.spatial_transforms[step] = ws
            #     self.feature_transforms[step] = wf

        # motion readout -> depends on specs in cfg
        # predicts velocity
        
        # main losses

        # ff loss -> cpc moco
        # fb loss -> CE or focal loss

        # auxilary losses

        # classification loss -> need to group all classes seen during one training 
        # momentary motion -> speed
        # long term motion -> acceleration
        # integration (accumulated distance)
        # beginning to end (difference)

        # eval

        # Encoding
        # momentary motion at least

        # pixel error (Acc)
        # IoU
        # mse difference

        # Extrapolation
        # pixel error / timestep (Acc)
        # IoU / timestep
        # mse difference / timestep

        # motion_labels= [
        #     'translate_v': 2, # [-1, 1]
        #     'translate_a': 2, # [-1, 1]
        #     'translate_distance': 1, # [0, +]
        #     'translate_difference': 2, # [-1, 1]
        #     'rotate_v': 1, # [-1, 1]
        #     'rotate_a': 1, # [-1, 1]
        #     'rotate_distance': 1, # [-, +]
        #     'rotate_difference': 1, # [-, +]
        #     'expand_v': 1, # [-1, 1]
        #     'expand_a': 1, # [-1, 1]
        #     'expand_distance': 1, # [-1, 1]
        #     'expand_difference': 1, # [-1, 1]
        # ]

        # fan_motion_labels = 
        # self.prediction_head = nn.Sequential(
        #     nn.AdaptiveMaxPool2d((1,1)),
        #     nn.Linear(R_channels[-1], motion_labels)
        # )

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
            # E_seq[l] = torch.zeros(batch_size, 2*self.a_channels[l], w, h).to(input.device)
            E_seq[l] = input.new_zeros([batch_size, 2*self.a_channels[l], w, h])
            # R_seq[l] = torch.zeros(batch_size, self.r_channels[l], w, h).to(input.device)
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
                
                # if l == 0:
                #     frame_prediction = A_hat
                #     if t>0:
                #         frame_error += torch.pow(A_hat - A, 2).detach().mean((1,2,3))
                
                if l == 0:
                    A_h_0 = A_hat
                    A_0 = A
                    A_hat = torch.argmax(A_hat, dim=1)[:,None] if A_hat.shape[1] != 1 else A_hat
                    # A_hat = F.softmax(A_hat, dim=1)[:,1][:,None] if A_hat.shape[1] != 1 else A_hat
                    
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)

            ###########################
            # Gather loop variables
            ###########################
            
            outputs = self.get_loss_t(outputs, t, R_top, E_seq, A_0, A_h_0, extra, time_steps)
        
        ###########################
        # Calculate loss and process output
        ###########################
        outputs = self.get_loss(input, outputs, extra)

        return outputs
    
    # def readout_motion(self, R):
    #     """ readout motion variables: velocity , traveled distance
    #     """
    #     return self.prediction_head(R)

    def get_loss_t(self, outputs, i, R, E_seq, frame, A_hat, extra, timesteps):
        
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
            if 'L1Loss' in self.losses:
                outputs['errors'] = []
            
            ############################
            # Evaluation
            ############################
            outputs['mse'] = []
            # outputs['Acc'] = []
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
                # print('target', frame.shape)
                # print('target', frame.min(), frame.max())
                # print('input', A_hat.shape)
                # pos = ((frame>=0.2)*1).sum()
                # neg = ((frame<0.2)*1).sum()
                # fb_loss = F.cross_entropy(A_hat, ((frame>=0.2)*1).long() weight=torch.Tensor([pos, neg]))
                fb_loss = losses.focal_loss(A_hat, (frame>=0.2)*1, gamma=2.0, reduce=True) # .view([frame.shape[0], -1]).mean(-1)
                
                # A_hat = torch.argmax(A_hat, dim=1)[:,None]
                # A_hat = F.softmax(A_hat, dim=1)[:,1][:,None]

                outputs['focal_loss'].append(fb_loss)
            if 'BinaryCrossEntropy' in self.losses:
                pos = ((frame>=0.2)*1)
                neg = ((frame<0.2)*1)
                n_pos = pos.sum()
                n_neg = neg.sum()
                n_pos = n_pos/(n_pos+n_neg)
                n_neg = n_neg/(n_pos+n_neg)
                mask = pos*n_neg + n_pos*neg            
                fb_loss = F.binary_cross_entropy_with_logits(A_hat, frame, reduction='none') * mask
                fb_loss = fb_loss.mean()

                outputs['bce_loss'].append(fb_loss)
            if 'L1Loss' in self.losses:
                fb_loss = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                outputs['errors'].append(fb_loss)
            
            ############################
            # FeedForward loss variables
            ############################
            if 'CPC' in self.losses:
                # CPC between R:t and R:t+4
                if i >= min(self.cpc_steps)+1:
                    outputs['cpc_targets'].append(self.cpc_target_layer(R))
                    # print('targ', i)

                for step in self.cpc_steps:
                    if i < timesteps-step: 
                        # print('preds', i, step)
                        outputs['cpc_preds'][step].append(self.cpc_pred_layer[step](R))
            
            ############################
            # Auxilary losses
            ############################

            # if 'smooth_l1_loss' in self.losses:
            #     # get_labels()
            #     ff_loss = F.smooth_l1_loss(A, frame)
            # outputs['ff_loss'] = ff_loss

            # outputs['mean_error'].append(mean_error)
            
            # print(A_hat[:,:10])
            # print(frame[:,:10])

            A_hat = A_hat.data #detach()
            frame = frame.data #detach()

            # print(A_hat.shape)
            ############################
            # Evaluation
            ############################

            if 'FocalLoss' in self.losses or 'CrossEntropy' in self.losses:
                outputs['mse'].append(F.mse_loss(F.softmax(A_hat, dim=1)[:,1][:,None], frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                # outputs['Acc'].append(losses.pixel_accuracy(A_hat, frame))
                balacc, precision, recall, f1s = losses.acc_scores((frame>0.2).long(), A_hat)

                A_hat = torch.argmax(A_hat, dim=1)[:,None].float()
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                
            elif 'BinaryCrossEntropy' in self.losses:
                A_hat = torch.sigmoid(A_hat)
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                # outputs['Acc'].append(losses.pixel_accuracy(A_hat, frame))
                outputs['IoU'].append(losses.intersectionAndUnion(A_hat, frame))
                balacc, precision, recall, f1s = losses.metric_scores((frame>0.2).long().byte(), (A_hat>0.2).long().byte())
                balacc = balacc * 100
            else:
                outputs['mse'].append(F.mse_loss(A_hat, frame, reduction='none').view([A_hat.shape[0], -1]).mean(-1))
                # outputs['Acc'].append(losses.pixel_accuracy(A_hat, frame))
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

        if 'L1Loss' in self.losses:
            output['errors'] = torch.stack(outputs['errors'], 2)*torch.Tensor([1]+[0.1]*(self.n_layers-1)).to(outputs['errors'][0].device)[None,:,None]
            output['errors'] = output['errors'].sum(1).mean()
            total_loss += output['errors'] * self.losses['L1Loss']

        ############################
        # Feedforward Loss
        ############################ 
        if 'CPC' in self.losses:
            cpc_targets = outputs['cpc_targets']
            
            cpc_preds = outputs['cpc_preds']
            cpc_loss = losses.cpc_loss(cpc_preds, cpc_targets)
                
            output['cpc_loss'] = cpc_loss
            total_loss += output['cpc_loss'] * self.losses['CPC']

            # for step in self.cpc_steps:
            #     if len(cpc_preds[step])>1:
            #         cpc_preds[step] = torch.stack(cpc_preds[step], -1).transpose(1,4)
            #         # .permute(1,0,3,4,2) #T B C H W -> B T H W C
            #         # logger.info(cpc_preds[step].shape)
            #         cpc_preds[step] = cpc_preds[step].reshape([-1,cpc_preds[step].shape[-1]]) # -> N C
            #         # logger.info(cpc_targets[:,step-min(self.cpc_steps):].shape)
            #         cpc_output = torch.matmul(cpc_targets[:,step-min(self.cpc_steps):].reshape([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

            #         labels = torch.cumsum(torch.ones_like(cpc_preds[step][:,0]).long(), 0) -1
            #         cpc_loss = cpc_loss + F.cross_entropy(cpc_output, labels)

        ############################
        # Auxilary Losses
        ############################ 
        # chosen losses from cfg ? how to integrate this information ?
        # motion labels from meta: translation, rotation, speed, acceleration
        # if not self.supervised:
        #     R = R.detach()
        # motion_outputs = self.readout_motion(R)

        output['total_loss'] = total_loss
        
        ############################
        # Evaluation
        ############################ 
        output['mse'] = torch.stack(outputs['mse'], 1).mean() - F.mse_loss(input[:,:,2:], input[:,:,1:-1])
        # output['Acc'] = torch.stack(outputs['Acc'], 1).float().mean() #
        output['IoU'] = torch.stack(outputs['IoU'], 1).float().mean() #
        
        output['prec'] = torch.stack(outputs['prec'], 0).float().mean()
        output['recall'] = torch.stack(outputs['recall'], 0).float().mean()
        output['f1s'] = torch.stack(outputs['f1s'], 0).float().mean()
        output['balacc'] = torch.stack(outputs['balacc'], 0).float().mean()


        if 'frames' in extra:
            output['frames'] = torch.stack(outputs['frames'], 2)
        
        return output
    

class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'


# https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81



def info(prefix, var):
    logger.info('-------{}----------'.format(prefix))
    logger.info('size: ', var.shape)
    logger.info('data type: ', type(var.data))
    logger.info(type(var))
