#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:26:05 2017

@author: jlaplaza
"""

import cv2

import torch
import numpy as np

from utils.nms_wrapper import nms
import cfg.config as cfg



# torch.manual_seed(1)



###############################################################################################################
##################### POSTPROCESS
###############################################################################################################

def yolo_to_bbox(bbox_pred, anchors, H, W):
    bsize = bbox_pred.shape[0]
    num_anchors = anchors.shape[0]
    bbox_out = np.zeros((bsize, int(H)*int(W), num_anchors, 4), dtype=float)

    for b in range(bsize):
        for row in range(int(H)):
            for col in range(int(W)):
                ind = int(row * W + col)
                for a in range(num_anchors):
                    cx = (bbox_pred[b, ind, a, 0] + col) / W
                    cy = (bbox_pred[b, ind, a, 1] + row) / H
                    bw = bbox_pred[b, ind, a, 2] * anchors[a][0] / W * 0.5
                    bh = bbox_pred[b, ind, a, 3] * anchors[a][1] / H * 0.5

                    bbox_out[b, ind, a, 0] = cx - bw
                    bbox_out[b, ind, a, 1] = cy - bh
                    bbox_out[b, ind, a, 2] = cx + bw
                    bbox_out[b, ind, a, 3] = cy + bh
    return bbox_out


def nms_detections(pred_boxes, scores, nms_thresh):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh, use_gpu=cfg.use_cuda)
    return keep

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes




def postprocess(bbox_pred, iou_pred, prob_pred, im_shape, cfg, thresh=0.6, iou_thresh = 0.6,
                size_index=9):
    """
    bbox_pred: (bsize, HxW, num_anchors, 4)
               ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    iou_pred: (bsize, HxW, num_anchors, 1)
    prob_pred: (bsize, HxW, num_anchors, num_classes)
    """

    # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    num_classes = cfg.num_classes
    anchors = cfg.anchors
    W, H = cfg.multi_scale_out_size[size_index]
    assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'  # noqa

    bbox_pred = yolo_to_bbox(bbox_pred, anchors, H, W)
    bbox_pred = np.reshape(bbox_pred, [-1, 4])
    bbox_pred[:, 0::2] *= float(im_shape[1])
    bbox_pred[:, 1::2] *= float(im_shape[0])
    bbox_pred = bbox_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, [-1])
    prob_pred = np.reshape(prob_pred, [-1, num_classes])

    cls_inds = np.argmax(prob_pred, axis=1)
    prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
    scores = iou_pred * prob_pred
    # scores = iou_pred

    # threshold
    keep = np.where(scores >= thresh)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # NMS
    keep = np.zeros(len(bbox_pred), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bbox_pred[inds]
        c_scores = scores[inds]
        c_keep = nms_detections(c_bboxes, c_scores, iou_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    # keep = nms_detections(bbox_pred, scores, 0.3)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # clip
    bbox_pred = clip_boxes(bbox_pred, im_shape)

    return bbox_pred, scores, cls_inds   
    
    
    
def preprocess_test(data, size_index=0):

    im, _, inp_size = data
    #inp_size = inp_size[size_index]
    if isinstance(im, str):
        im = cv2.imread(im)
    ori_im = np.copy(im)

    if inp_size is not None:
        w, h = inp_size
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 255.

    return im, [], [], [], ori_im    
    
    

    
###############################################################################################################
##################### DRAW RESULTS
###############################################################################################################    
    
    





def draw_detection(im, bboxes, scores, cls_inds, cfg, thr=0.6):
    # draw image
    colors = cfg.colors
    labels = cfg.label_names

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
