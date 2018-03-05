#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:46:50 2018

@author: jlaplaza
"""

def reorg_cpu(x_tensor, w, h, c, batch, stride, forward, out_tensor):

    # Grab the tensor
    

    # https://github.com/pjreddie/darknet/blob/master/src/blas.c
    out_c = c/(stride*stride);

    for b in batch:
        for k in c:
            for j in h:
                for i in w:
                    in_index  = i + w*(j + h*(k + c*b))
                    c2 = k % out_c
                    offset = k / out_c
                    w2 = i*stride + offset % stride
                    h2 = j*stride + offset / stride
                    out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b))
                    if (forward): 
                        out_tensor[out_index] = x_tensor[in_index]
                    else :
                        out_tensor[in_index] = x_tensor[out_index];

    return out_tensor