#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:36:05 2018

@author: jlaplaza
"""


import numpy as np



DTYPE = np.float

"""
def extern from "math.h":
    double abs(double m)
    double log(double x)
"""

def bbox_overlaps(boxes, query_boxes):
    return bbox_overlaps_c(boxes, query_boxes)

def bbox_overlaps_c(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=DTYPE)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bbox_intersections(boxes, query_boxes):
    return bbox_intersections_c(boxes, query_boxes)


def bbox_intersections_c(boxes, query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K), dtype=DTYPE)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    intersec[n, k] = iw * ih / box_area
    return intersec


def bbox_ious(boxes, query_boxes):
    return bbox_ious_c(boxes, query_boxes)


def bbox_ious_c(boxes, query_boxes):
    """
    For each query box compute the IOU covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K), dtype=DTYPE)
    
    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    box_area = (
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1)
                    )
                    inter_area = iw * ih
                    intersec[n, k] = inter_area / (qbox_area + box_area - inter_area)
    return intersec


def anchor_intersections(anchors, query_boxes):
    return anchor_intersections_c(anchors, query_boxes)


def anchor_intersections_c(anchors, query_boxes):
    """
    For each query box compute the intersection ratio covered by anchors
    ----------
    Parameters
    ----------
    boxes: (N, 2) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K), dtype=DTYPE)

    for n in range(N):
        anchor_area = anchors[n, 0] * anchors[n, 1]
        for k in range(K):
            boxw = (query_boxes[k, 2] - query_boxes[k, 0] + 1)
            boxh = (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            iw = min(anchors[n, 0], boxw)
            ih = min(anchors[n, 1], boxh)
            inter_area = iw * ih
            intersec[n, k] = inter_area / (anchor_area + boxw * boxh - inter_area)

    return intersec


def bbox_intersections_self(boxes):
    return bbox_intersections_self_c(boxes)


def bbox_intersections_self_c(boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    Returns
    -------
    overlaps: (N, N) ndarray of intersec between boxes and query_boxes
    """
    N = boxes.shape[0]
    intersec = np.zeros((N, N), dtype=DTYPE)

    for k in range(N):
        box_area = (
            (boxes[k, 2] - boxes[k, 0] + 1) *
            (boxes[k, 3] - boxes[k, 1] + 1)
        )
        for n in range(k+1, N):
            iw = (
                min(boxes[n, 2], boxes[k, 2]) -
                max(boxes[n, 0], boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], boxes[k, 3]) -
                    max(boxes[n, 1], boxes[k, 1]) + 1
                )
                if ih > 0:
                    intersec[k, n] = iw * ih / box_area
    return intersec


def bbox_similarities(boxes, query_boxes):
    return bbox_similarities_c(boxes, query_boxes)

def bbox_similarities_c(boxes, query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float (dets)
    Returns
    -------
    overlaps: (N, K) ndarray of similarity scores between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    sims = np.zeros((N, K), dtype=DTYPE)

    for n in range(N):
        cx1 = (boxes[n, 0] + boxes[n, 2]) * 0.5
        cy1 = (boxes[n, 1] + boxes[n, 3]) * 0.5
        w1 = boxes[n, 2] - boxes[n, 0] + 1
        h1 = boxes[n, 3] - boxes[n, 1] + 1

        for k in range(K):
            cx2 = (query_boxes[k, 0] + query_boxes[k, 2]) * 0.5
            cy2 = (query_boxes[k, 1] + query_boxes[k, 3]) * 0.5
            w2 = query_boxes[k, 2] - query_boxes[k, 0] + 1
            h2 = query_boxes[k, 3] - query_boxes[k, 1] + 1

            loc_dist = abs(cx1 - cx2) / (w1 + w2) + abs(cy1 - cy2) / (h1 + h2)
            shape_dist = abs(w2 * h2 / (w1 * h1) - 1.0)

            sims[n, k] = -np.log(loc_dist + 0.001) - shape_dist * shape_dist + 1

    return sims

if __name__ == "__main__":
    a = np.array([[1, 3, 3, 1], [3, 5, 5, 3]])
    
    b = np.array([[2, 4, 4, 2], [2, 4, 4, 2]])
    print(bbox_ious(a,b))
    
    