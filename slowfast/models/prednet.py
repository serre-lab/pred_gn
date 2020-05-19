import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY
from .rnns import hConvGRUCell

logger = logging.get_logger(__name__)


__all__ = [
    "PredNet",
]

@MODEL_REGISTRY.register()
class PredNet(nn.Module):
    def __init__(self, cfg): #R_channels=(3, 48, 96, 192), A_channels=(3, 48, 96, 192)
        super(PredNet, self).__init__()
        R_channels=(3, 48, 96, 192)
        A_channels=(3, 48, 96, 192)
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        # self.output_mode = output_mode

        # default_output_modes = ['prediction', 'error']
        # assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
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
        
        for t in range(time_steps):
            A = input[:,:,t]
            A = A.float() #type(torch.cuda.FloatTensor)
            
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


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
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


        output['frame_errors'] = output['pred_errors'][:,0].mean()
        output['pred_errors'] = output['pred_errors'].sum(1)
        
                
        if 'frames' in extra:
            output['frames'] = torch.stack(frames, 2)
        return output



@MODEL_REGISTRY.register()
class PredNet_hGRU(nn.Module):
    def __init__(self, cfg): #R_channels=(3, 48, 96, 192), A_channels=(3, 48, 96, 192)
        super(PredNet_hGRU, self).__init__()
        R_channels=(3, 48, 96, 192)
        A_channels=(3, 48, 96, 192)
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        # self.output_mode = output_mode

        # default_output_modes = ['prediction', 'error']
        # assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            # cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
            #                     (3, 3))
            
            # c_input = nn.Conv2d(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i], 1)
            if i == self.n_layers - 1:
                c_input = nn.Sequential(nn.Conv2d(2 * self.a_channels[i], self.r_channels[i], 1), nn.ReLU())
            
            else:
                c_input = nn.Sequential(nn.Conv2d(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i], 1), nn.ReLU())
            
            cell = hConvGRUCell(
                        input_size=self.r_channels[i],
                        hidden_size=self.r_channels[i],
                        kernel_size=3,
                        batchnorm=True, #'GN', # True
                        timesteps=cfg.DATA.NUM_FRAMES,
                        gala=False,
                        spatial_kernel=3,
                        less_softplus=False,
                        r=4,
                        init=nn.init.orthogonal_,
                        norm='GN',
                        bottom_layer=False)

            setattr(self, 'cell_input{}'.format(i), c_input)
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     for l in range(self.n_layers):
    #         cell = getattr(self, 'cell{}'.format(l))
    #         cell.reset_parameters()

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
        
        for t in range(time_steps):
            A = input[:,:,t]
            A = A.float() #type(torch.cuda.FloatTensor)
            
            for l in reversed(range(self.n_layers)):
                in_conv = getattr(self, 'cell_input{}'.format(l))
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    # hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    # hx = H_seq[l]
                if l == self.n_layers - 1:
                    tmp = in_conv(E)
                    R = cell(tmp, R, timestep=t)
                else:
                    # import pdb;pdb.set_trace()
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)

                    tmp = in_conv(tmp)
                    R = cell(tmp, R, timestep=t)
                R_seq[l] = R
                # H_seq[l] = hx


            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
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


        output['frame_errors'] = output['pred_errors'][:,0].mean()
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

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = nn.Parameter(torch.Tensor(
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = nn.Parameter(torch.Tensor(
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = nn.Parameter(torch.Tensor(
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = nn.Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        wx = F.conv2d(input, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)

        wxhc = wx + wh + torch.cat([wc[:, :2 * self.out_channels], 
                                    self.wc_blank.expand(wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), 
                                    wc[:, 2 * self.out_channels:]], 
                                    1)

        i = torch.sigmoid(wxhc[:, :self.out_channels])
        f = torch.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = torch.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = torch.sigmoid(wxhc[:, 3 * self.out_channels:])

        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, (h_1, c_1)


def info(prefix, var):
    logger.info('-------{}----------'.format(prefix))
    logger.info('size: ', var.shape)
    logger.info('data type: ', type(var.data))
    logger.info(type(var))
