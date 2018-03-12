#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:25:08 2018

@author: jlaplaza
"""

from .nms.cpu_nms import cpu_nms
from .nms.gpu_nms import gpu_nms


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, use_gpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if use_gpu:
        return gpu_nms(dets, thresh)
    return cpu_nms(dets, thresh)