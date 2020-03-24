import torch
import torch.nn.functional as F
from torch import nn
from .batch_norm import get_norm
from torch.autograd import Function


class RBPFun(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(
                last_state,
                state_2nd_last,
                grad_outputs=neumann_v_prev,
                retain_graph=True,
                allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            normsv.append(normv.data.item())
            normsg.append(normg.data.item())
        return (None, neumann_g, None, None, None, None)


def CBP_penalty(last_state, prev_state, mu=0.9, compute_hessian=True, pen_type='l1'):
    """Handles RBP grads in the forward pass."""
    """Compute the constrained RBP penalty."""
    norm_1_vect = torch.ones_like(last_state)
    norm_1_vect.requires_grad = False
    vj_prod = torch.autograd.grad(
        last_state,
        prev_state,
        grad_outputs=[norm_1_vect],
        retain_graph=True,
        create_graph=compute_hessian,
        allow_unused=True)[0]
    vj_penalty = (vj_prod - mu).clamp(0) ** 2
    return vj_penalty.mean()  # Save memory with the mean

#########################################################################################################
#### New GN CBP     ###################################################################################
#########################################################################################################

class RBPFun(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad, max_steps=15, norm_cap=False):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]
        if norm_cap:
            normsv = []
            normsg = []
            normg = torch.norm(neumann_g_prev)
            normsg.append(normg.data.item())
            normsv.append(normg.data.item())
        if truncate_iter <= 0:
            normv = 1.
            steps = 0.  # TODO: Add to TB
            while normv > 1e-9 and steps < max_steps:
                neumann_v = torch.autograd.grad(
                    last_state,
                    state_2nd_last,
                    grad_outputs=neumann_v_prev,
                    retain_graph=True,
                    allow_unused=True)
                neumann_g = neumann_g_prev + neumann_v[0]
                normv = torch.norm(neumann_v[0])
                steps += 1
                neumann_v_prev = neumann_v
                neumann_g_prev = neumann_g
        else:
            for ii in range(truncate_iter):
                neumann_v = torch.autograd.grad(
                    last_state,
                    state_2nd_last,
                    grad_outputs=neumann_v_prev,
                    retain_graph=True,
                    allow_unused=True)
                neumann_g = neumann_g_prev + neumann_v[0]
                if norm_cap:
                    normv = torch.norm(neumann_v[0])
                    normg = torch.norm(neumann_g)
                    if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                        normsg.append(normg.data.item())
                        normsv.append(normv.data.item())
                        neumann_g = neumann_g_prev
                        break
                    normsv.append(normv.data.item())
                    normsg.append(normg.data.item())
                neumann_v_prev = neumann_v
                neumann_g_prev = neumann_g
        return (None, neumann_g, None, None, None, None)


def CBP_penalty(
        last_state,
        prev_state,
        tau=0.999,  # Changed 2/25/20 from 0.9
        compute_hessian=True,
        pen_type='l1'):
    """Handles RBP grads in the forward pass."""
    """Compute the constrained RBP penalty."""
    norm_1_vect = torch.ones_like(last_state)
    norm_1_vect.requires_grad = False
    vj_prod = torch.autograd.grad(
        last_state,
        prev_state,
        grad_outputs=[norm_1_vect],
        retain_graph=True,
        create_graph=compute_hessian,
        allow_unused=True)[0]
    vj_penalty = (vj_prod - tau).clamp(0) ** 2  # Squared to emphasize outliers
    return vj_penalty.sum()  # Save memory with the mean


#########################################################################################################
#### Currently Used     #################################################################################
#########################################################################################################


class hConvGRUCell_(nn.Module):
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
            norm='GN'):
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

#########################################################################################################

class tdConvGRUCell_(nn.Module):
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
            norm='GN'):
        
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

        prev_state2 = (1 - g2_t) * lower_ + g2_t * h2_t
        
        if not extra:
            return prev_state2
        else:
            return prev_state2, extra


#########################################################################################################
#### New GN Cells     ###################################################################################
#########################################################################################################


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
            norm='GN'):
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
        self.bn = nn.ModuleList(
            [get_norm(norm, hidden_size) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

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
        c0_t = self.bn[0](
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        if self.less_softplus:
            inhibition = F.softplus(  # F.softplus(input_) moved outside
                input_ - c0_t * (self.alpha * h_ + self.mu))
        else:
            inhibition = F.softplus(  # F.softplus(input_) moved outside
                F.softplus(input_) - F.softplus(c0_t * (self.alpha * h_ + self.mu)))
        if 'error' in return_extra:
            extra['error'] = inhibition

        g1_t = torch.sigmoid(self.u1_gate(inhibition))
        excitation = F.softplus(self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding)))
        if self.less_softplus:
            h_t = excitation
        else:
            h_t = F.softplus(
                self.kappa * (
                    inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g1_t) * h_ + g1_t * h_t
        if extra:
            return op, extra
        else:
            return op


#########################################################################################################

class tdConvGRUCell(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(
            self,
            fan_in,
            td_fan_in,
            diff_fan_in,
            kernel_size,
            gala=False,
            batchnorm=True,
            timesteps=1,
            init=nn.init.orthogonal_,
            grad_method='bptt',
            norm='SyncBN',
            spatial_transform=False,
            gn_remap=False):
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala
        self.gn_remap = gn_remap

        self.remap_0 = nn.Conv2d(td_fan_in, diff_fan_in, 1)
        self.remap_1 = nn.Conv2d(diff_fan_in, fan_in, 1)
        if self.gn_remap:
            self.gn_remap0 = get_norm(norm, diff_fan_in)
            self.gn_remap1 = get_norm(norm, fan_in)
        
        self.spatial_transform = spatial_transform
        if self.spatial_transform:
            self.warp = SpatialTransformer(fan_in)

        self.u1_gate = nn.Conv2d(fan_in, fan_in, 1)
        self.u2_gate = nn.Conv2d(fan_in, fan_in, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.w = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((fan_in, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(2)])

        # TODO: Alekh, why is orthogonal slow af
        init(self.w_gate_inh)
        init(self.w_gate_exc)

        init(self.u1_gate.weight)
        init(self.u2_gate.weight)

        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.mu, 1)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.kappa, 0.5)
        # if self.timesteps == 1:
        #     init_timesteps = 2
        # else:
        #     init_timesteps = self.timesteps
        nn.init.uniform_(self.u1_gate.bias.data, 1, self.timesteps - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data

    def forward(self, lower_, higher_, timestep=0, return_extra=[]):
        extra={}
        prev_state2 = F.interpolate(
            higher_,
            size = lower_.shape[2:],
            #scale_factor=2,
            mode="nearest")
        if self.gn_remap:
            prev_state2 = F.softplus(self.gn_remap0(self.remap_0(prev_state2)))
            prev_state2 = F.softplus(self.gn_remap1(self.remap_1(prev_state2)))
        else:
            prev_state2 = F.softplus(self.remap_0(prev_state2))
            prev_state2 = F.softplus(self.remap_1(prev_state2))
        
        if 'remap' in return_extra:
            extra['remap'] = prev_state2

        g1_t = torch.sigmoid(self.u1_gate(prev_state2))
        c1_t = self.bn[0](
            F.conv2d(
                prev_state2 * g1_t,
                self.w_gate_inh,
                padding=self.padding))

        inhibition = F.softplus(
            lower_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        if 'error' in return_extra:
            extra['error'] = inhibition

        g2_t = torch.sigmoid(self.u2_gate(inhibition))
        excitation = self.bn[1](
            F.conv2d(
                inhibition,
                self.w_gate_exc,
                padding=self.padding))
        h2_t = F.softplus(
            self.kappa * (
                inhibition + excitation) + self.w * inhibition * excitation)
        op = (1 - g2_t) * lower_ + g2_t * h2_t  # noqa Note: a previous version had higher_ in place of lower_
        
        if self.spatial_transform:
            op = self.warp(inhibition, op)
        if extra:
            return op, extra
        else:
            return op

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

#########################################################################################################
#### Unused Cells #######################################################################################
#########################################################################################################

class hConvGRUCell_old(nn.Module):
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
            r=4,
            grad_method='bptt',
            norm='SyncBN'):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.gala = gala

        if self.gala:
            self.u1_channel_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, 1)
            self.u1_channel_gate_1 = nn.Conv2d(
                hidden_size // r, hidden_size, 1, bias=False)
            self.u1_spatial_gate_0 = nn.Conv2d(
                hidden_size, hidden_size // r, spatial_kernel)
            self.u1_spatial_gate_1 = nn.Conv2d(
                hidden_size // r, 1, spatial_kernel, bias=False)
            self.u1_combine_bias = nn.Parameter(
                torch.empty((hidden_size, 1, 1)))
        else:
            self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
            nn.init.xavier_uniform_(self.u1_gate.weight)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
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

    def forward(self, input_, h_, timestep=0):
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
            F.conv2d(
                h_ * g1_t,
                self.w_gate_inh,
                padding=self.padding))
        next_state1 = F.softplus(
            input_ - F.softplus(c1_t * (self.alpha * h_ + self.mu)))
        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv2d(
                next_state1,
                self.w_gate_exc,
                padding=self.padding))
        h_ = (1 - g2_t) * h_ + g2_t * h2_t
        return h_

#########################################################################################################

class tdConvGRUCell_old(nn.Module):
    """
    Generate a TD cell
    """

    def __init__(
            self,
            fan_in,
            td_fan_in,
            diff_fan_in,
            kernel_size,
            batchnorm=True,
            timesteps=8,
            grad_method='bptt',
            norm='SyncBN'):
        super(tdConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.input_size = fan_in
        self.hidden_size = td_fan_in
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.grad_method = grad_method
        self.remap_0 = nn.Conv2d(td_fan_in, diff_fan_in, 1)
        self.remap_1 = nn.Conv2d(diff_fan_in, fan_in, 1)

        self.u1_gate = nn.Conv2d(fan_in, fan_in, 1)
        self.u2_gate = nn.Conv2d(fan_in, fan_in, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(
            torch.empty(fan_in, fan_in, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((fan_in, 1, 1)))
        self.mu = nn.Parameter(torch.empty((fan_in, 1, 1)))

        if norm == "":
            norm = 'SyncBN'

        # Norm is harcoded to group norm
        norm = 'GN'
        self.bn = nn.ModuleList(
            [get_norm(norm, fan_in) for i in range(2)])

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

    def forward(self, lower_, higher_, timestep=0):
        prev_state2 = F.interpolate(
            higher_,
            scale_factor=2,
            mode="nearest")
        prev_state2 = F.softplus(self.remap_0(prev_state2))
        prev_state2 = F.softplus(self.remap_1(prev_state2))

        g1_t = torch.sigmoid(self.u1_gate(prev_state2))
        c1_t = self.bn[0](
            F.conv2d(
                prev_state2 * g1_t,
                self.w_gate_inh,
                padding=self.padding))

        next_state1 = F.softplus(
            lower_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid(self.u2_gate(next_state1))
        h2_t = self.bn[1](
            F.conv2d(
                next_state1,
                self.w_gate_exc,
                padding=self.padding))

        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t
        return prev_state2