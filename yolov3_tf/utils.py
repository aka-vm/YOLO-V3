import numpy as np

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