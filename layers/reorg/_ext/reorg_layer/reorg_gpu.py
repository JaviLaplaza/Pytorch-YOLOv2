#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:09:19 2018

@author: jlaplaza
"""

#include <THC/THC.h>
#include "reorg_cuda_kernel.h"

# extern THCState *state;

def reorg_cuda(x_tensor, w, h, c, batch, stride, forward, out_tensor):
    """
{
    float * x = THCudaTensor_data(state, x_tensor);
    float * out = THCudaTensor_data(state, out_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);
    reorg_ongpu(x, w, h, c, batch, stride, forward, out, stream);

    return 1;
}
"""