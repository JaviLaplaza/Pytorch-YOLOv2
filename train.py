#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:02:20 2018

@author: jlaplaza
"""

import os
import cv2
import numpy as np

import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from network.yolo_network import YOLO
import utils.yolo as yolo_utils
import utils.network as net_utils
import cfg.config as cfg
from datasets.transforms import Rescale, ToTensor
from utils.im_transform import imcv2_recolor

from datasets.caltech_dataset import CaltechDataset

from utils.timer import Timer
from random import randint
import matplotlib.pyplot as plt


# data loader
images_dir = os.path.join(cfg.ROOT_DIR, 'datasets/CALTECH/images_subset/')
annotations_dir = os.path.join(cfg.ROOT_DIR, 'datasets/CALTECH/')
dataset = CaltechDataset(images_dir, annotations_dir)
 #todas las imagenes del mismo batch deben tener el mismo tamaño, asi que hay que usar transformadas

# dst_size=cfg.inp_size)
print('Data loaded successfully.')
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

"""
for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched)
"""

model = YOLO()

# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
model.load_from_npz(cfg.pretrained_model, num_conv=18)
print("Model loaded successfully.")



"""
#Freeze all the layers of the network
for param in model.parameters():
    param.requires_grad = False 
"""


if cfg.use_cuda:
    model.cuda()

model.train()

print("Setting model to Training Mode.")





# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)



#batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = cfg.size_index
for i_batch, sample_batched in enumerate(dataloader):
    t.tic()
    # batch
    #batch = imdb.next_batch(size_index)
    im = sample_batched['images'].numpy()
    
    im = imcv2_recolor(im)
      
    im = np.resize(im, (1, cfg.multi_scale_inp_size[size_index][0], cfg.multi_scale_inp_size[size_index][1], 3))
    
    gt_boxes = sample_batched['gt_boxes']
    gt_classes = sample_batched['gt_classes']
    dontcare = sample_batched['dontcare']
    #orgin_im = sample_batched['origin_im']

    # forward
    im_data = net_utils.np_to_variable(im, use_cuda=cfg.use_cuda, volatile=False).permute(0, 3, 1, 2)


    model(im_data, gt_boxes, gt_classes, dontcare, size_index)
    
    

    # backward
    loss = model.loss
    bbox_loss += model.bbox_loss.data.cpu().numpy()[0]
    iou_loss += model.iou_loss.data.cpu().numpy()[0]
    cls_loss += model.cls_loss.data.cpu().numpy()[0]
    train_loss += loss.data.cpu().numpy()[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    if i_batch % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
               'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
               (i_batch, step_cnt, batch_size, train_loss, bbox_loss,
                iou_loss, cls_loss, duration,
                str(datetime.timedelta(seconds=int((batch_size - step_cnt) * duration))))))  # noqa

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()
        size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

    if i_batch > 0 and (i_batch % batch_size == 0):
        if i_batch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)

        save_name = os.path.join(cfg.train_output_dir,
                                 '{}_{}.h5'.format(cfg.exp_name, i_batch))
        net_utils.save_net(save_name, model)
        print(('save model: {}'.format(save_name)))
        step_cnt = 0

