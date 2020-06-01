import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY
from .rnns import ConvGRUCell, ConvGRUCell_in, ConvGRUCell_out, ConvGRUCell_RNN

logger = logging.get_logger(__name__)


__all__ = [
    "PredNet_GRU",
]

@MODEL_REGISTRY.register()
class PredNet_GRU(nn.Module):
    def __init__(self, cfg): #R_channels=(3, 48, 96, 192), A_channels=(3, 48, 96, 192)
        super(PredNet_GRU, self).__init__()
        R_channels=(3, 48, 96, 192)
        A_channels=(3, 48, 96, 192)
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        # self.output_mode = output_mode

        # default_output_modes = ['prediction', 'error']
        # assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)
        
        if cfg.PREDNET.CELL =='GRU_in':
            rnn_cell = ConvGRUCell_in
        elif cfg.PREDNET.CELL =='GRU_out':
            rnn_cell = ConvGRUCell_out
        elif cfg.PREDNET.CELL =='GRU_rnn':
            rnn_cell = ConvGRUCell_RNN
        else:
            rnn_cell = ConvGRUCell

        for i in range(self.n_layers):
            cell = rnn_cell(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
                                3)
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], 3, padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input, extra=[], autoreg=False):
        
        if isinstance(input, list):
            input = input[0]
        

        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = torch.zeros(batch_size, 2*self.a_channels[l], w, h).to(input.device)
            R_seq[l] = torch.zeros(batch_size, self.r_channels[l], w, h).to(input.device)
            
            # E_seq[l] = input.new([batch_size, 2*self.a_channels[l], w, h])
            # R_seq[l] = input.new([batch_size, self.r_channels[l], w, h]).zeros_
            w = w//2
            h = h//2
        time_steps = input.size(2)
        total_error = []
        frames = []
        frame_error = 0
        for t in range(time_steps):
            A = input[:,:,t]
            A = A.float() #type(torch.cuda.FloatTensor)
            
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                E = E_seq[l]
                R = R_seq[l]
                if l == self.n_layers - 1:
                    R = cell(E, R)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R = cell(tmp, R)
                R_seq[l] = R


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
                    if t>0:
                        frame_error += torch.pow(A_hat - A, 2).detach().mean((1,2,3))
                
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg],1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
            if 'frames' in extra:
                frames.append(frame_prediction)
            mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
            # batch x n_layers
            if t>0:
                total_error.append(mean_error)
        output = {}
        output['pred_errors'] = torch.stack(total_error, 2)*torch.Tensor([1]+[0.1]*(self.n_layers-1)).to(mean_error.device)[None,:,None] # batch x n_layers x nt

        output['frame_errors'] = (frame_error/(time_steps-1)).mean() #output['pred_errors'][:,0].mean()
        output['pred_errors'] = output['pred_errors'].sum(1)
                
        if 'frames' in extra:
            output['frames'] = torch.stack(frames, 2)
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
