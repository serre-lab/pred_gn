#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


_LOSSES = {"cross_entropy": nn.CrossEntropyLoss, "bce": nn.BCELoss}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def pixel_accuracy(preds, label, reduce=False):
    bs = preds.shape[0]
    preds = preds.view([bs, -1])
    preds = (preds>0.5)*1
    label = label.view([bs, -1])
    label = (label>0.5)*1
    
    acc_sum = (1 * (preds == label)).sum(1) / label.shape[-1] 
    if reduce:
        acc_sum = acc_sum.mean()
    return acc_sum

def intersectionAndUnion(imPred, imLab, reduce=False):
    score = 0
    batch_size = imLab.shape[0]
    imLab = imLab.view([batch_size, -1])
    imPred = imPred.view([batch_size, -1])
    
    l = (imLab > 0.5)*1
    p = (imPred > 0.5)*1
    
    # print(l.shape)
    # print(p.shape)

    intersection = (p == l)*1
    union = (p != l)*1 + intersection
    
    score = intersection.new_zeros([batch_size])
    
    # print(intersection.type())
    # print(union.type())
    # import pdb; pdb.set_trace()
    for i in range(batch_size):
        u = union[i].sum().float()
        if u > 0:     
            score[i] = intersection[i].sum().float()/u
        
        else:
            score[i] = 0

    if reduce:
        score = score.mean()

    return score

def class_balanced_bce_with_logits(input_, target, reduction='mean'):
    #import pdb; pdb.set_trace()
    ones = (target == 1).to(target.device).float()
    zeros = (target == 0).to(target.device).float()
    n_ones = ones.sum()
    n_zeros = zeros.sum()
    combined = n_ones + n_zeros
    ones = ones * (n_zeros * 1. / combined)
    zeros = zeros * (n_ones * 1. / combined)
    weights = ones + zeros
    out = F.cross_entropy(input_, target, reduction='none')
    out = out * weights
    if reduction == 'mean':
        out = out.mean()
    return out


def wt_ce_with_logits(input_, target, data, reduction='mean'):
    
    #import pdb; pdb.set_trace()
    ones = (target == 1).to(target.device).float()
    zeros = (target == 0).to(target.device).float()
    n_ones = ones.sum()
    n_zeros = zeros.sum()
    combined = n_ones + n_zeros
    ones = ones * (n_zeros * 1. / combined)
    zeros = zeros * (n_ones * 1. / combined)
    weights = ones + zeros
    weights = weights + weights * ((data > 0.2).float() * 2)
    out = F.cross_entropy(input_, target, reduction='none')
    out = out * weights
    if reduction == 'mean':
        out = out.mean()
    return out

def weighted_bce_with_logits(input_, target, reduction='mean'):
    pos = ((target>=0.2)*1)
    neg = ((target<0.2)*1)
    n_pos = pos.sum()
    n_neg = neg.sum()
    n_pos = n_pos/(n_pos+n_neg)
    n_neg = n_neg/(n_pos+n_neg)
    mask = pos*n_neg + n_pos*neg            
    loss = F.binary_cross_entropy_with_logits(input_, target, reduction='none') * mask
    if reduction == 'mean':
        return loss.mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.reshape(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt

        if self.size_average: return loss.mean()
        else: return loss.sum()

def focal_loss(input, target, gamma=0, alpha=None, reduce=True):

    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
    target = target.reshape(-1, 1)
    
    logpt = F.log_softmax(input, dim=1)
    logpt = logpt.gather(1, target.long())
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if not isinstance(alpha, list): alpha = target.new([alpha, 1 - alpha])
        else: alpha = target.new(alpha)

        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        
        at = alpha.gather(0,target.long().data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1 - pt)**gamma * logpt
    
    loss = loss.view([loss.shape[0], -1]).mean(-1)

    if reduce: return loss.mean()
    else: return loss

def weighted_focal_loss(input, target, gamma=0, alpha=None, reduce=True):

    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        
    target = target.reshape(-1, 1)
    
    logpt = F.log_softmax(input, dim=1)
    logpt = logpt.gather(1, target.long())
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if not isinstance(alpha, list): alpha = target.new([alpha, 1 - alpha])
        else: alpha = target.new(alpha)

        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        
        at = alpha.gather(0,target.long().data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1 - pt)**gamma * logpt
    
    pos = ((target>=0.2)*1)
    neg = ((target<0.2)*1)
    n_pos = pos.sum()
    n_neg = neg.sum()
    n_pos = n_pos/(n_pos+n_neg)
    n_neg = n_neg/(n_pos+n_neg)
    mask = pos*n_neg + n_pos*neg

    loss = loss*mask
    
    loss = loss.view([loss.shape[0], -1]).mean(-1)
    
       

    if reduce: return loss.mean()
    else: return loss

    
def cpc_loss(cpc_preds, cpc_targets):
    cpc_loss = 0
    
    cpc_targets = torch.stack(cpc_targets,-1).transpose(1,4)

    cpc_steps = list(cpc_preds.keys())

    for step in cpc_steps:
        if len(cpc_preds[step])>1:
            
            cpc_preds[step] = torch.stack(cpc_preds[step], -1).transpose(1,4)
            
            # print(cpc_preds[step].shape, cpc_targets.shape)
            # .permute(1,0,3,4,2) #T B C H W -> B T H W C
            # logger.info(cpc_preds[step].shape)
            cpc_preds[step] = cpc_preds[step].reshape([-1,cpc_preds[step].shape[-1]]) # -> N C
            # logger.info(cpc_targets[:,step-min(self.cpc_steps):].shape)
            cpc_output = torch.matmul(cpc_targets[:,step-min(cpc_steps):].reshape([-1, cpc_preds[step].shape[-1]]), cpc_preds[step].t())

            labels = torch.cumsum(torch.ones_like(cpc_preds[step][:,0]).long(), 0) -1
            cpc_loss = cpc_loss + F.cross_entropy(cpc_output, labels)
    
    return cpc_loss

# def balanced_bce(input, target):
#     input = input.reshape([-1, 1])
#     target = target.reshape([-1, 1])
#     binary_cross_entropy_with_logits(input, target)


def metric_scores(target, pred):
    
    correct = pred.eq(target)
    tp = correct[target == 1].sum().float()
    tn = correct[target == 0].sum().float()

    P = target.sum()
    N = (target == 0).sum()
    tpfp = pred.sum()
    if tpfp == 0:
        tpfp = 1e-6
    recall = tp / P
    precision = tp / tpfp
    bacc = (tn / N + recall) / 2
    f1s = (2 * tp) / (P + tpfp)
    return bacc, precision, recall, f1s


def acc_scores(target, prediction):
    target = target.byte()
    _, pred = prediction.topk(1, 1, True, True)
    balacc, precision, recall, f1s = metric_scores(target, pred.byte())
    return balacc * 100, precision, recall, f1s
