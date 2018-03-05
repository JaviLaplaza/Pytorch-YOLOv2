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


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
        box_confidence -- tensor of shape (19, 19, 5, 1), probability that there 
                          is some object for each 5 boxes predicted in 19x19 grid
        boxes -- tensor of shape (19, 19, 5, 4) containing (bx, by, bh, bw) for
                 each 5 boxes per cell
        box_class_probs -- tensor of shape (19, 19, 5, 80), containing the 
                            detection probabilities for each of the 80 classes
        threshold -- real value, if [highest class probability score < threshold],
                     then get rid of the corresponding box
        
    Returns:
        scores -- tensor of shape (None, ), containing the class probability
                  of selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w)
                 coordinates of selected boxes
        classes -- tensor of shape (None, ), containing the index of the class
                   detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected 
    boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10, ) if there are 
    10 boxes.
    """
    
    #Step 1: Compute box scores
    box_scores = box_confidence*box_class_probs #shape (19,19,5,80)
    
    #Step 2: Find the box_classes thanks to the max box_scores, keep track of 
    #the corresponding score
    box_class_scores, box_classes = torch.max(box_scores, -1, True) #shape (19,19,5,1)
 
    #Step 3: Create a filtering mask bases on "box_class_scores" by using 
    # "threshold". The mask should have the same dimension as box_class_scores, 
    # and be True for the boxes you want to keep
    filtering_mask = box_class_scores >= threshold
    #print("filtering_mask type = " + str(filtering_mask.type))
    
    #Step 4: Apply mask to scores, boxes and classes
    scores = torch.masked_select(box_class_scores, filtering_mask)
    boxes = torch.masked_select(boxes, filtering_mask)
    boxes = boxes.view(-1,4)
    classes = torch.masked_select(box_classes, filtering_mask)
    
    return scores, boxes, classes

"""
box_confidence = torch.normal(std=torch.randn([19,19,5,1]).fill_(4), means=torch.randn([19,19,5,1]).fill_(1))
boxes = torch.normal(std=torch.randn([19,19,5,4]).fill_(4), means=torch.randn([19,19,5,4]).fill_(1))
box_class_probs = torch.normal(std=torch.randn([19,19,5,1]).fill_(4), means=torch.randn([19,19,5,1]).fill_(1))
scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.size() = " + str(scores.size()))
print("boxes.size() = " + str(boxes.size()))
print("classes.size() = " + str(classes.size()))
"""    
    
    



def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() 
             that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used 
                     for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less
    than max_boxes. Note also that this function will transpose the shapes of 
    scores, boxes, classes. This is made for convenience.
    """
    
    # get the list of indices corresponding to boxes you keep
    order = torch.sort(scores, descending=True)[1].numpy()    
    x1 = boxes.numpy()[:,0]
    y1 = boxes.numpy()[:,1]
    x2 = boxes.numpy()[:,2]
    y2 = boxes.numpy()[:,3]
    
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    nms_indices = []
    while (len(nms_indices) < max_boxes):
        i = order[0]
        nms_indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]        
        
    
    # Use torch.index_select() to select only nms_indices from scores, boxes and classes
    scores = torch.index_select(scores, 0, torch.from_numpy(np.asarray(nms_indices)))
    boxes = torch.index_select(boxes, 0, torch.from_numpy(np.asarray(nms_indices)))
    classes = torch.index_select(classes, 0, torch.from_numpy(np.asarray(nms_indices)))

    return scores, boxes, classes
    


    
"""    
scores = torch.normal(std=torch.randn([54,]).fill_(4), means=torch.randn([54,]).fill_(1))
boxes = torch.normal(std=torch.randn([54,4]).fill_(4), means=torch.randn([54,4]).fill_(1))
classes = torch.normal(std=torch.randn([54,]).fill_(4), means=torch.randn([54,]).fill_(1))
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))
"""






def yolo_boxes_to_corners(box_xy, box_wh):
    x1 = box_xy[...,0] - box_wh[...,0]/2
    y1 = box_xy[...,1] - box_wh[...,1]/2
    x2 = box_xy[...,0] + box_wh[...,0]/2
    y2 = box_xy[...,1] + box_wh[...,1]/2
    boxes = torch.stack((x1, y1, x2, y2), -1)
    return boxes

def scale_boxes(boxes, image_shape):
    kx = image_shape[0]/608
    ky = image_shape[1]/608
    boxes[...,0] = boxes[...,0]*kx
    boxes[...,1] = boxes[...,1]*ky
    boxes[...,2] = boxes[...,2]*kx
    boxes[...,3] = boxes[...,3]*ky
    return boxes
    



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
        
    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use the functions to perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
        
    return scores, boxes, classes
"""    
yolo_outputs = (torch.normal(std=torch.randn([19, 19, 5, 1]).fill_(4), means=torch.randn([19, 19, 5, 1]).fill_(1)), torch.normal(std=torch.randn([19, 19, 5, 2]).fill_(4), means=torch.randn([19, 19, 5, 2]).fill_(1)), torch.normal(std=torch.randn([19, 19, 5, 2]).fill_(4), means=torch.randn([19, 19, 5, 2]).fill_(1)), torch.normal(std=torch.randn([19, 19, 5, 80]).fill_(4), means=torch.randn([19, 19, 5, 80]).fill_(1)))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape)) 
"""




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
    keep = nms(dets, nms_thresh)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
