import torch
from torch import nn

import numpy as np

def scale_anchor_boxes(anchor_boxes: list[list[tuple[float, float]]], output_scales=(13, 26, 52)):
    """
    Process anchor boxes to be relative to the output frame size.

    Parameters:
        anchor_boxes:
            Matrix of tuples of floats(0-1) representing the anchor box size
            ratio w.r.t. the original input image size.

        output_scales: list of ints or list of tuple(scale_h, scale_w) representing the output
        frame size, depending on the YOLO layer.

    Returns:
        Matrix of tuples of floats(0-output_scale_size) representing
        the anchor box size w.r.t. the output frame size.
    """
    anchor_boxes = torch.tensor(anchor_boxes)
    output_scales = torch.tensor(output_scales)
    if output_scales.ndim == 1:
        output_scales = output_scales.unsqueeze(1).repeat(1, 2)
    output_scales = output_scales.unsqueeze(1).repeat(1, anchor_boxes.shape[1], 1)

    return anchor_boxes * output_scales


def iou_wh(boxes1_wh, boxes2_wh):
    # Used to choose the best anchor box for a given bounding box
    intersection = torch.min(boxes1_wh, boxes2_wh).prod(axis=-1)
    union = boxes1_wh.prod(axis=-1) + boxes2_wh.prod(axis=-1) - intersection

    return intersection / union

def iou_boxes(boxes1, boxes2, box_format="midpoint"):
    # used to check how good pred box is wrt target box and calculate the loss
    # box_format: midpoint/corners
    """
    if box_format == "midpoint":
        boxes_n -> [x, y, w, h]
        bnx1 -> x - w/2
        bny1 -> y - h/2
        bnx2 -> x + w/2
        bny2 -> y + h/2

    elif box_format == "corners":
        boxes_n -> [bnx1, bny1, bnx2, bny2]

    x1 -> max(b1x1, b2x1)
    x2 -> min(b1x2, b2x2)
    y1 -> max(b1y1, b2y1)
    y2 -> min(b1y2, b2y2)

    Intersection -> Area(x1, y1, x2, y2)
    Union -> Area(b1) + Area(b2) - Intersection

    IOU -> Intersection / Union
    """
    if box_format == "midpoint":
        b1x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
        b1y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
        b1x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
        b1y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2
        b2x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
        b2y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
        b2x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
        b2y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2
    elif box_format == "corners":
        b1x1 = boxes1[..., 0:1]
        b1y1 = boxes1[..., 1:2]
        b1x2 = boxes1[..., 2:3]
        b1y2 = boxes1[..., 3:4]
        b2x1 = boxes2[..., 0:1]
        b2y1 = boxes2[..., 1:2]
        b2x2 = boxes2[..., 2:3]
        b2y2 = boxes2[..., 3:4]

    x1 = torch.max(b1x1, b2x1)
    x2 = torch.min(b1x2, b2x2)
    y1 = torch.max(b1y1, b2y1)
    y2 = torch.min(b1y2, b2y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = abs((b1x2 - b1x1) * (b1y2 - b1y1)) \
          + abs((b2x2 - b2x1) * (b2y2 - b2y1)) \
          - intersection

    return intersection / union

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    pass

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.functional.max_pool2d(nn.functional.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x