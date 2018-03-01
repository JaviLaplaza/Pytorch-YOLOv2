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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # adding ROOT_DIR to sys.path to handle the imports

import numpy as np

import torch
from torch.autograd import Variable
from torch.multiprocessing import Pool


from network.yolo_network import YOLO
import cfg.config as cfg
import utils.network as net_utils
import utils.yolo as yolo_utils
from utils.timer import Timer

# required paths

im_path = cfg.IM_DIR # path to the directory containing the images




# YOLO input
x = Variable(torch.randn(1, 3, 608, 608))




def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
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

# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
# model.load_from_npz(cfg.pretrained_model, num_conv=18)



t_det = Timer()
t_total = Timer()


im_fnames = sorted((fname
                    for fname in os.listdir(im_path)
                    if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)

#pool = Pool(processes=1)

"""
#print("im_fnames next = " + str(next(im_fnames)))
interm = list(im_fnames)
print("im_fnames = " + str(interm))
print("len = " + str(len(interm)))
test1, test2 = preprocess(interm[1])
print("Test1 = " + str(test1))
print("Test2 = " + str(test2))
test = list(pool.imap(preprocess, im_fnames, chunksize=1))
print(test)
#print(next(pool.imap(preprocess, im_fnames, chunksize=1)))

"""




#for i, (image, im_data) in enumerate(pool.imap(preprocess, im_fnames, chunksize=1)):
t_total.tic()
interm = list(im_fnames)
image, im_data = preprocess(interm[0])
im_data = net_utils.np_to_variable(im_data, is_cuda=False, volatile=True).permute(0, 3, 1, 2)


t_det.tic()
bbox_pred, iou_pred, prob_pred = model(im_data)
det_time = t_det.toc()

# to numpy
bbox_pred = bbox_pred.data.cpu().numpy()

iou_pred = iou_pred.data.cpu().numpy()

prob_pred = prob_pred.data.cpu().numpy()



# print bbox_pred.shape, iou_pred.shape, prob_pred.shape

bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, cfg.thresh)
    

im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
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
    
im2show = im2show[...,::-1]

plt.imshow(im2show, aspect='auto')
plt.savefig('foo.jpg')
plt.show()

total_time = t_total.toc()


    # wait_time = max(int(60 - total_time * 1000), 1)
    # cv2.waitKey(0)

format_str = 'frame: %d, ' + '(detection: %.1f Hz, %.1f ms) ' + '(total: %.1f Hz, %.1f ms)'
print((format_str % (1, 1./ det_time, det_time * 1000, 1./  total_time, total_time * 1000)))

t_total.clear()
t_det.clear()