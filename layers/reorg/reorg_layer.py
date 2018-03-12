#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:01:29 2018

@author: jlaplaza
"""

import torch
from torch.autograd import Function



class ReorgFunction(Function):
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        stride = self.stride

        bsize, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride * stride)  # noqa
        out = torch.FloatTensor(bsize, out_c, out_h, out_w)

        
        if x.is_cuda:
            out = out.cuda()
            out = x.view(bsize, out_c, out_h, out_w)
        else:
            out = x.view(bsize, out_c, out_h, out_w)
        
        
        
        

        return out


    def backward(self, grad_top):
        stride = self.stride
        bsize, c, h, w = grad_top.size()

        out_w, out_h, out_c = w * stride, h * stride, c / (stride * stride)
        grad_bottom = torch.FloatTensor(bsize, out_c, out_h, out_w)


        # rev_stride = 1. / stride    # reverse
        if grad_top.is_cuda:
            grad_bottom = grad_bottom.cuda()
            grad_bottom = grad_top.view(bsize, out_c, out_h, out_w)
        else:

            grad_bottom = grad_top.view(bsize, out_c, out_h, out_w)
            

        return grad_bottom


class ReorgLayer(torch.nn.Module):
    def __init__(self, stride):
        super(ReorgLayer, self).__init__()

        self.stride = stride

    def forward(self, x):
        x = ReorgFunction(self.stride)(x)
        return x