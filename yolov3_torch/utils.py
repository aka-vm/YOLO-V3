import torch
from torch import nn

import numpy as np

def anchor_box_convert(anchor_boxes: list[list[tuple[float, float]]], output_frame_size=((13, 13), (26, 26), (52, 52))):
    """
    Process anchor boxes to be relative to the output frame size.

    Parameters:
        anchor_boxes:
            Matrix of tuples of floats(0-1) representing the anchor box size
            ratio w.r.t. the original input image size.

        output_frame_size: list of tuples of ints representing the output
        frame size, depending on the YOLO layer.

    Returns:
        Matrix of tuples of floats(0-output_frame_size[i][j]) representing
        the anchor box size w.r.t. the output frame size.
    """
    anchor_boxes = np.array(anchor_boxes)

    output = []
    for i, frame_size in enumerate(output_frame_size):
        frame_size = frame_size
        anchor_box = anchor_boxes[i]
        anchor_box = anchor_box * frame_size
        output.append(anchor_box)

    return np.array(output)


def iou(boxes1_dims, boxes2_dims):
    # intersection = np.minimum(box1_dims, box2_dims).prod(axis=-1)
    intersection = torch.min(boxes1_dims, boxes2_dims).prod(axis=-1)
    # union = box1_dims.prod(axis=-1) + box2_dims.prod(axis=-1) - intersection
    union = boxes1_dims.prod(axis=-1) + boxes2_dims.prod(axis=-1) - intersection

    return intersection / union

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    pass

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.functional.max_pool2d(nn.functional.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x