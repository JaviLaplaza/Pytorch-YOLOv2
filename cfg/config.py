#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:34:29 2018

@author: jlaplaza
"""

import os
import numpy as np
from .config_voc import * 


# input and output size
############################
multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        np.array([608, 608], dtype=np.int),
                        ]   # w, h
multi_scale_out_size = [multi_scale_inp_size[0] / 32,
                        multi_scale_inp_size[1] / 32,
                        multi_scale_inp_size[2] / 32,
                        multi_scale_inp_size[3] / 32,
                        multi_scale_inp_size[4] / 32,
                        multi_scale_inp_size[5] / 32,
                        multi_scale_inp_size[6] / 32,
                        multi_scale_inp_size[7] / 32,
                        multi_scale_inp_size[8] / 32,
                        multi_scale_inp_size[9] / 32,
                        ]   # w, h
inp_size = np.array([608, 608], dtype=np.int)   # w, h

out_size = inp_size / 32


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


#h5_fname = 'yolo-voc.weights.h5'
pretrained_fname = 'yolo.weights'


# detection config
############################
thresh = 0.6
iou_thresh = 0.6

# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
IM_DIR = os.path.join(ROOT_DIR, 'images')

trained_model = os.path.join(WEIGHTS_DIR, h5_fname)
pretrained_model = os.path.join(WEIGHTS_DIR, pretrained_fname)

