#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:58:30 2018

@author: jlaplaza
"""

from __future__ import absolute_import

import os
import cv2
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # adding ROOT_DIR to sys.path to handle the imports

import numpy as np

import torch

from network.yolo_network import YOLO
import cfg.config as cfg
import utils.network as net_utils
import utils.yolo as yolo_utils
from utils.timer import Timer


# required paths

#im_path = cfg.IM_DIR # path to the directory containing the images

# YOLO input
# x = Variable(torch.randn(1, 3, 608, 608))

def preprocess(frame):
    # return fname
    image = frame
    im_data = np.expand_dims(yolo_utils.preprocess_test((frame, None, cfg.inp_size))[0], 0)
    return image, im_data



def load_model(fname, model):
    import h5py
    h5f = h5py.File(fname, mode='r')
    #print("h5 contents = " + str(list(h5f.keys())))
    for k, v in list(model.state_dict().items()):
        
        #print("model contents = " + str(k))
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)
        
        
        
# YOLO forward  
model = YOLO() #creation of the network

load_model(cfg.trained_model, model) #loading pretrained weights into the network

if cfg.use_cuda:
    model.cuda()

model.eval()
print("Model loaded successfully.")
print("Setting Model to Evaluation Mode")

# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
# model.load_from_npz(cfg.pretrained_model, num_conv=18)


t_det = Timer()
t_total = Timer()
t_cap = Timer()
cap = cv2.VideoCapture("/dev/video1")
i=0

while(True):
    t_cap.tic()
    # Capture frame by frame
    ret, frame = cap.read()
    cap_time = t_cap.toc()
    
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t_total.tic()

    image, im_data = preprocess(frame)
    im_data = net_utils.np_to_variable(im_data, use_cuda=cfg.use_cuda, volatile=True).permute(0, 3, 1, 2)


    t_det.tic()
    bbox_pred, iou_pred, prob_pred = model(im_data)
    det_time = t_det.toc()

    # to numpy
    bbox_pred = bbox_pred.data.cpu().numpy()

    iou_pred = iou_pred.data.cpu().numpy()

    prob_pred = prob_pred.data.cpu().numpy()

    bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, cfg.thresh, cfg.iou_thresh)

    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg, cfg.thresh)
    total_time = t_total.toc()

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. *
                                  float(im2show.shape[1]) / im2show.shape[0]),
                              1000))
    """
    cv2.startWindowThread()
    cv2.imshow('test', im2show)
    """
    
    # Display the resulting frame
    cv2.imshow('frame',im2show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    total_time = t_total.toc()


    # wait_time = max(int(60 - total_time * 1000), 1)
    # cv2.waitKey(0)

    format_str = 'frame: %d, ' + '(detection: %.1f Hz, %.1f ms) ' + '(total: %.1f Hz, %.1f ms)'
    print((format_str % (i, 1./ det_time, det_time * 1000, 1./  total_time, total_time * 1000)))
    i += 1
    t_total.clear()
    t_det.clear()   

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


