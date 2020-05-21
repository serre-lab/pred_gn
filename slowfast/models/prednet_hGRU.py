import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .batch_norm import get_norm

import slowfast.utils.logging as logging

from .build import MODEL_REGISTRY
# from .rnns import hConvGRUCell

logger = logging.get_logger(__name__)


__all__ = [
    "PredNet_hGRU",
]


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
            if i == self.n_layers - 1:
                c_input = nn.Conv2d(2 * self.a_channels[i], self.r_channels[i], 1)
            
            else:
                c_input = nn.Conv2d(2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i], 1)
            
            cell = hConvGRUCell(
                        input_size=self.r_channels[i],
                        hidden_size=self.r_channels[i],
                        kernel_size=3,
                        batchnorm=(cfg.GN.RECURRENT_BN != ""), # True
                        timesteps=cfg.DATA.NUM_FRAMES,
                        gala=False,
                        spatial_kernel=3,
                        less_softplus=False,
                        r=4,
                        init=nn.init.orthogonal_,
                        norm=cfg.GN.RECURRENT_BN, #'GN',
                        )

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
            R_seq[l] = F.softplus(torch.zeros(batch_size, self.r_channels[l], w, h).to(input.device))
            
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
            spatial_kernel=3,
            less_softplus=False,
            r=4,
            init=nn.init.orthogonal_,
            grad_method='bptt',
            norm='GN',
            ):
        super(hConvGRUCell, self).__init__()
        self.gala = False
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.less_softplus = less_softplus
        if "GN" in norm and input_size<=4:
            norm = "IN"
        
        if self.gala:
            self.u0_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u0_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u0_spatial_gate_0 = nn.Parameter(
                torch.empty(
                    hidden_size // 2,
                    hidden_size,
                    kernel_size,
                    kernel_size))
            self.u0_spatial_bias = nn.Parameter(
                torch.empty((hidden_size // 2, 1, 1)))
            self.u0_spatial_gate_1 = nn.Parameter(
                torch.empty(
                    1,
                    hidden_size // 2,
                    kernel_size,
                    kernel_size))
            self.u0_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u0_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        if not self.less_softplus:
            self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
            self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        # Norm is harcoded to group norm
        if self.batchnorm:
            self.bn = nn.ModuleList(
                [get_norm(norm, hidden_size) for i in range(2)])

        init(self.w_gate_inh)
        init(self.w_gate_exc)

        if self.batchnorm:
            for bn in self.bn:
                nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        if not self.less_softplus:
            nn.init.constant_(self.w, 0.5)
            nn.init.constant_(self.kappa, 0.5)
        if self.timesteps == 1:
            init_timesteps = 2
        else:
            init_timesteps = self.timesteps
        if self.gala:
            # First init the kernels
            init(self.u0_channel_gate_0.weight)
            init(self.u0_channel_gate_1.weight)
            init(self.u0_spatial_gate_0)
            init(self.u0_spatial_gate_1)
            init(self.u1_gate.weight)

            # Now init biases
            nn.init.zeros_(self.u0_spatial_bias)
            nn.init.uniform_(self.u0_combine_bias, 1, init_timesteps - 1)
            self.u0_combine_bias.data.log()
            self.u1_gate.bias.data = -self.u0_combine_bias.data.squeeze()
        else:
            nn.init.xavier_uniform_(self.u0_gate.weight)
            nn.init.uniform_(self.u0_gate.bias.data, 1, init_timesteps - 1)
            self.u0_gate.bias.data.log()
            self.u1_gate.bias.data = -self.u0_gate.bias.data

    def forward(self, input_, h_, timestep=0, return_extra=[]):
        extra={}
        if self.gala:
            global_0 = F.softplus(self.u0_channel_gate_0(h_))
            global_1 = self.u0_channel_gate_1(global_0)
            local_0 = F.softplus(
                F.conv2d(
                    h_,
                    self.u0_spatial_gate_0,
                    padding=self.padding) + self.u0_spatial_bias)
            local_1 = F.conv2d(
                local_0,
                self.u0_spatial_gate_1,
                padding=self.padding)
            gate_act = global_1 * local_1 + self.u0_combine_bias
            g1_t = torch.sigmoid(gate_act)
        else:
            g1_t = torch.sigmoid(self.u0_gate(h_))
        if self.batchnorm:
            c0_t = self.bn[0](
                    F.conv2d(
                        h_ * g1_t,
                        self.w_gate_inh,
                        padding=self.padding))
        else:
            c0_t = F.conv2d(
                        h_ * g1_t,
                        self.w_gate_inh,
                        padding=self.padding)

        if self.less_softplus:
            supp = F.softplus(  # F.softplus(input_) moved outside
                input_ - c0_t * (self.alpha * h_ + self.mu))

        else:
            supp = F.softplus(F.softplus(input_) - F.softplus(c0_t * (self.alpha * h_ + self.mu)))

        g2_t = torch.sigmoid(self.u1_gate(supp))

        if self.batchnorm:
            excitation = F.softplus(self.bn[1](
                F.conv2d(
                    supp,
                    self.w_gate_exc,
                    padding=self.padding)))
        else:
            excitation = F.softplus(
                F.conv2d(
                    supp,
                    self.w_gate_exc,
                    padding=self.padding))
        if self.less_softplus:
            h_t = excitation
        else:
            h_t = F.softplus(
                self.kappa * (supp + excitation) + self.w * supp * excitation)
        op = (1 - g2_t) * h_ + g2_t * h_t
        if extra:
            return op, extra
        else:
            return op




def info(prefix, var):
    logger.info('-------{}----------'.format(prefix))
    logger.info('size: ', var.shape)
    logger.info('data type: ', type(var.data))
    logger.info(type(var))
