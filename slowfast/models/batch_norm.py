import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function
from torch.nn import functional as F

# from detectron2.utils import comm
from slowfast.utils.distributed import get_world_size

# from nn.modules.instancenorm import _InstanceNorm
# from nn.modules.batchnorm import _NormBase

import torch
import torch.nn.functional as F
from torch.nn.modules.instancenorm import _InstanceNorm

# import slowfast.utils.logging as logging

# logger = logging.get_logger(__name__)

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "BN2D": nn.BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm3d,
            "SyncBN2D": NaiveSyncBatchNorm2d,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(4, channels),
            "GNR": lambda channels: GroupNorm(4, channels, momentum=0.1, affine=True, track_running_stats=True),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.size(1).item() # .shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

# logger = logging.getLogger(__name__)

class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.size(1).item()

        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr =  vec.split(C)#torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias


class GroupNorm(_InstanceNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        
        
        super(GroupNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_groups = num_groups
        del self.running_mean
        del self.running_var

    # def _check_input_dim(self, input):
    #     raise NotImplementedError

    def reset_stats(self):
        self.reset=True

    def forward(self, input):
        # self._check_input_dim(input)
        x = input.reshape([input.size(0), self.num_groups, -1]) #input.size(1)//self.num_groups,-1])
        if self.reset:
            self.running_mean = x.mean((0,2)).detach()
            self.running_var = x.mean((0,2)).detach()
            self.num_batches_tracked.zero_()
            self.reset=False
        return F.instance_norm(
            x, self.running_mean, self.running_var, None, None,
            True, self.momentum, self.eps).reshape(input.shape)*self.weight[:,None,None] + self.bias[:,None,None]



# def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
#                   use_input_stats=True, momentum=0.1, eps=1e-5):
#     r"""Applies Group Normalization for channels in the same group in each data sample in a
#     batch.
#     See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
#     :class:`~torch.nn.GroupNorm3d` for details.
#     """
#     if not use_input_stats and (running_mean is None or running_var is None):
#         raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

#     b, c = input.size(0), input.size(1)
#     if weight is not None:
#         weight = weight.repeat(b)
#     if bias is not None:
#         bias = bias.repeat(b)

#     def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
#                        bias=None, use_input_stats=None, momentum=None, eps=None):
#         # Repeat stored stats and affine transform params if necessary
#         if running_mean is not None:
#             running_mean_orig = running_mean
#             running_mean = running_mean_orig.repeat(b)
#         if running_var is not None:
#             running_var_orig = running_var
#             running_var = running_var_orig.repeat(b)

#         #norm_shape = [1, b * c / group, group]
#         #print(norm_shape)
#         # Apply instance norm
#         input_reshaped = input.contiguous().view(1, int(b * c/group), group, *input.size()[2:])

#         out = F.batch_norm(
#             input_reshaped, running_mean, running_var, weight=weight, bias=bias,
#             training=use_input_stats, momentum=momentum, eps=eps)

#         # Reshape back
#         if running_mean is not None:
#             running_mean_orig.copy_(running_mean.view(b, int(c/group)).mean(0, keepdim=False))
#         if running_var is not None:
#             running_var_orig.copy_(running_var.view(b, int(c/group)).mean(0, keepdim=False))

#         return out.view(b, c, *input.size()[2:])
#     return _instance_norm(input, group, running_mean=running_mean,
#                           running_var=running_var, weight=weight, bias=bias,
#                           use_input_stats=use_input_stats, momentum=momentum,
#                           eps=eps)


# class _GroupNorm(_BatchNorm):
#     def __init__(self, num_features, num_groups=1, eps=1e-5, momentum=0.1,
#                  affine=False, track_running_stats=False):
#         self.num_groups = num_groups
#         self.track_running_stats = track_running_stats
#         super(_GroupNorm, self).__init__(int(num_features/num_groups), eps,
#                                          momentum, affine, track_running_stats)

#     def _check_input_dim(self, input):
#         return NotImplemented

#     def forward(self, input):
#         self._check_input_dim(input)

#         return group_norm(
#             input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
#             self.training or not self.track_running_stats, self.momentum, self.eps)


# class GroupNorm2d(_GroupNorm):
#     r"""Applies Group Normalization over a 4D input (a mini-batch of 2D inputs
#     with additional channel dimension) as described in the paper
#     https://arxiv.org/pdf/1803.08494.pdf
#     `Group Normalization`_ .
#     Args:
#         num_features: :math:`C` from an expected input of size
#             :math:`(N, C, H, W)`
#         num_groups:
#         eps: a value added to the denominator for numerical stability. Default: 1e-5
#         momentum: the value used for the running_mean and running_var computation. Default: 0.1
#         affine: a boolean value that when set to ``True``, this module has
#             learnable affine parameters. Default: ``True``
#         track_running_stats: a boolean value that when set to ``True``, this
#             module tracks the running mean and variance, and when set to ``False``,
#             this module does not track such statistics and always uses batch
#             statistics in both training and eval modes. Default: ``False``
#     Shape:
#         - Input: :math:`(N, C, H, W)`
#         - Output: :math:`(N, C, H, W)` (same shape as input)
#     Examples:
#         >>> # Without Learnable Parameters
#         >>> m = GroupNorm2d(100, 4)
#         >>> # With Learnable Parameters
#         >>> m = GroupNorm2d(100, 4, affine=True)
#         >>> input = torch.randn(20, 100, 35, 45)
#         >>> output = m(input)
#     """

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))


# class GroupNorm3d(_GroupNorm):
#     """
#         Assume the data format is (B, C, D, H, W)
#     """
#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'
#                              .format(input.dim()))