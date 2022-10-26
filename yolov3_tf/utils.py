import numpy as np

# 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
PASCAL_ANCHORS = [
    [(0.279,0.216), (0.375,0.476), (0.897,0.784)],      # 116,90,  156,198,  373,326
    [(0.072,0.147), (0.149,0.108), (0.142,0.286)],      # 30,61,  62,45,  59,119
    [(0.024,0.031), (0.038,0.072), (0.079,0.055)],      # 10,13,  16,30,  33,23
]

def anchor_box_convert(anchor_boxes, output_frame_size=((13, 13), (26, 26), (52, 52))):
    anchor_boxes = np.array(anchor_boxes)

    output = []
    for i, frame_size in enumerate(output_frame_size):
        frame_size = frame_size
        anchor_box = anchor_boxes[i]
        anchor_box = anchor_box * frame_size
        output.append(anchor_box)

    return np.array(output)


def iou_area(box1_dims, box2_dims):
    intersection = np.minimum(box1_dims, box2_dims).prod(axis=-1)
    union = box1_dims.prod(axis=-1) + box2_dims.prod(axis=-1) - intersection

    return intersection / union