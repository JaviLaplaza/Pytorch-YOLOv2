#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:12:58 2018

@author: jlaplaza
"""

import os
import json
import glob
import re
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from skimage import io

from torch.utils.data import Dataset


class CaltechDataset(Dataset):
    """Caltech Pedestrian Detection dataset."""
    

    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_dir = annotations_dir
        self.annotations_file = os.path.join(self.annotations_dir, 'annotations.json')
        self.annotations = json.load(open(self.annotations_file))
        
        self.images_dir = images_dir
        self.img_fns = defaultdict(dict)
        
        for fn in sorted(glob.glob(os.path.join(self.images_dir, '*.png'))):
            set_name = re.search('(set[0-9]+)', fn).groups()[0]
            self.img_fns[set_name] = defaultdict(dict)

        for fn in sorted(glob.glob(os.path.join(self.images_dir, '*.png'))):
            set_name = re.search('(set[0-9]+)', fn).groups()[0]
            video_name = re.search('(V[0-9]+)', fn).groups()[0]
            self.img_fns[set_name][video_name] = []


        for fn in sorted(glob.glob(os.path.join(self.images_dir, '*.png'))):
            set_name = re.search('(set[0-9]+)', fn).groups()[0]
            video_name = re.search('(V[0-9]+)', fn).groups()[0]
            n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
            self.img_fns[set_name][video_name].append((int(n_frame), fn))
        self.img_list = sorted(os.listdir(self.images_dir))
        
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_dir))
    
    def caltech_bbox_to_yolo_bbox(x, y, w, h):
        xbl = x
        ybl = y + h
        xtr = x + w
        ytr = y
        return xbl, ybl, xtr, ytr


    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir,
                                self.img_list[idx])
        image = cv2.imread(img_name)
        
        img_name_split = img_name.split('/')
        
        dataset_dir, img_dir, img = img_name_split[-3:]
        data_set, data_video, data_frame = img[0:len(img)-4].split('_') 
        

        if str(data_frame) in self.annotations[data_set][data_video]['frames']:
            data = self.annotations[data_set][data_video]['frames'][str(data_frame)]
            bboxes = np.zeros((len(data), 4), dtype=int)
            classes = np.zeros(len(data), dtype=int)
            for i, datum in enumerate(data):
                if datum['lbl'] == 'person' or datum['lbl'] == 'people':
                    classes[i] = 14
                    x, y, w, h = [int(v) for v in datum['pos']]
                    bboxes[i] = CaltechDataset.caltech_bbox_to_yolo_bbox(x, y, w, h)
                else:
                    classes[i] = 0
                    x, y, w, h = [int(v) for v in datum['pos']]
                    bboxes[i] = CaltechDataset.caltech_bbox_to_yolo_bbox(x, y, w, h)
        else:
            bboxes = np.zeros((1,4), dtype=int)
            classes = np.zeros(1, dtype=int)
                                    
                    
        
        sample = {'images': image, 'gt_boxes': bboxes, 'gt_classes': classes, 'dontcare': 0}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
        

if __name__ == '__main__':
    dataset = CaltechDataset('CALTECH/images_subset', 'CALTECH')
    
    def yolo_bbox_to_caltech_bbox(bbox):
        x = bbox[0]
        y = bbox[3]
        w = bbox[2] - bbox[0]
        h = bbox[1] - bbox[3]
        
        return [x, y, w, h]
    
    #print(dataset.img_fns['set00']['V000'])
    c = dataset[3]
    print(type(c['images']))
    
    for bbox in c['gt_boxes']:
        bbox = yolo_bbox_to_caltech_bbox(bbox)
        cv2.rectangle(c['images'], (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
    
    c['images'] = c['images'][...,::-1]
    plt.imshow(c['images'], aspect='auto')
    plt.show()
    dataset.close()
    