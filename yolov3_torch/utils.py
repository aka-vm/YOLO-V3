import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from colorsys import hsv_to_rgb

import config

def format_anchor_boxes(
    anchor_boxes: list[list[tuple[float, float]]],
    output_scales: list
) -> torch.Tensor:
    """
    Process anchor boxes to be relative to the output frame size.

    Parameters:
        anchor_boxes:
            Matrix of tuples of floats(0-1) representing the anchor box size
            ratio w.r.t. the original input image size.
            e.g. -> config.ANCHORS
                [
                    [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
                    [(0.7, 0.8), (0.9, 0.1), (0.2, 0.3)],
                    [(0.4, 0.5), (0.6, 0.7), (0.8, 0.9)]
                ]

        output_scales:
            list of ints or list of tuple(scale_h, scale_w) representing the
            outputframe size, depending on the YOLO layer.

    Returns:
        Matrix of tuples of floats(0, output_scale_size) representing
        the anchor box size w.r.t. the output frame size.
        output[i][j] = anchor_boxes[i][j] * output_scales[i]
    """
    anchor_boxes = torch.tensor(anchor_boxes)
    output_scales = torch.tensor(output_scales)
    if output_scales.ndim == 1:
        output_scales = output_scales.unsqueeze(1).repeat(1, 2)
    output_scales = output_scales.unsqueeze(1).repeat(1, anchor_boxes.shape[1], 1)

    return anchor_boxes * output_scales

def iou_wh(
    boxes1_wh,
    boxes2_wh
) -> torch.Tensor:
    """
    Parameters:
        boxes1_wh: list of shape (N, 2) or (2, ).
        boxes2_wh: list of shape (N, 2) or (2, ).

    returns:
        Intersection over Union of the two boxes.
        shape: (N, ) or (1, )
    """
    # Used to choose the best anchor box for a given bounding box
    intersection = torch.min(boxes1_wh, boxes2_wh).prod(axis=-1)
    union = boxes1_wh.prod(axis=-1) + boxes2_wh.prod(axis=-1) - intersection

    return intersection / union

def iou_boxes(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    box_format="midpoint"
) -> torch.Tensor:
    # used to check how good pred box is wrt target box and calculate the loss
    # box_format: midpoint/corners
    """
    Find the Intersection over Union of two sets of boxes, takes exact coordinates into account.

    Parameters:
        boxes1: Tensor of shape (N, 4) for the format (x, y, w, h) or (x1, y1, x2, y2)
        boxes2: Tensor of shape (N, 4) for the format (x, y, w, h) or (x1, y1, x2, y2)

    Returns:
        Tensor of shape (N, 1) containing the IoU for each pair of boxes

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

def non_max_suppression(
    bboxes: list[list],
    base_prob_threshold: float,
    iou_threshold: float,
    box_format="midpoint"
) -> list[list]:
    """
    Parameters:
        bboxes: list of all bboxes, each bbox is a list of 6 elements.
            midpoint -> [prob, x, y, w, h, class]; corners -> [prob, x1, y1, x2, y2, class]
        base_prob_threshold: float -> threshold for class confidence, if lower than this, discard.
        iou_threshold: float -> threshold for iou, if higher than this, discard. means that its the same pred.
        box_format: str -> "midpoint" or "corners"

    Returns:
        bboxes after non-max suppression
    """
    # bbox -> bbox if bbox > base_prob_threshold

    bboxes = [bbox for bbox in bboxes if bbox[0] > base_prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    final_bboxes = []

    while bboxes:
        best_bbox = bboxes.pop(0)
    # For Each iter, this will remove all the bbox that have iou > iou_threshold with best_bbox;
    # iou > iou_threshold probabally means that they are the same bbox.

        bboxes = [
            bbox
            for bbox in bboxes
            if bbox[5] != best_bbox[5]
            or iou_boxes(
                torch.tensor(best_bbox),
                torch.tensor(bbox),
                box_format=box_format
            ) < iou_threshold
        ]

        final_bboxes.append(best_bbox)

    return final_bboxes

def cells_to_bboxes(
    predictions,
    anchors,
    S,
    is_preds=True
):
    """
    Model Output to a gernalized format for all the anchors.

    Parameters:
        predictions
            tensor of shape([N, 3, S, S, 6])
            [prob, x, y, w, h, class]
        anchors
            the anchors used for predictions tensor
            shape([3, 2]) [w, h]
        S: int
            The number of cells the image is divided into
        is_preds: bool
            Whether the input is model output or not

    Returns:
        bboxes: list[tensor]
            shape([N, num_anchors, S, S, 6]) [prob, x, y, w, h, class]

    """
    N = predictions.shape[0]
    num_anchors = len(anchors)
    # wrt the cell
    bboxes_coordinates = predictions[..., 1:5]  # last dim -> [x, y, w, h]
    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        bboxes_coordinates[..., 0:2] = torch.sigmoid(bboxes_coordinates[..., 0:2])          # x, y
        bboxes_coordinates[..., 2:4] = torch.exp(bboxes_coordinates[..., 2:4]) * anchors    # w, h
        scores      = torch.sigmoid(predictions[..., 0:1])
        best_class  = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores      = predictions[..., 0:1]
        best_class  = predictions[..., 5:6]

    cell_indices = torch.arange(S).repeat(N, num_anchors, S, 1).unsqueeze(-1).to(predictions.device)

    # wrt the image
    x = (bboxes_coordinates[..., 0:1] + cell_indices) / S
    y = (bboxes_coordinates[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / S
    w_h = bboxes_coordinates[..., 2:4] / S

    bboxes = torch.cat((scores, x, y, w_h, best_class), dim=-1).reshape(N, num_anchors * S * S, 6)
    return bboxes.tolist()

def plot_image(image, boxes):
    """
    image: image tensor
    boxes: list[tuple] -> shape([N, 6]) [prob, x, y, w, h, class]
    """
    class_labels = config.CLASS_LABELS
    num_classes = len(class_labels)
    if num_classes <= 20:
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]
    else:
        colors = generate_unique_colors(num_classes)

    image = np.array(image)
    h, w, _ = image.shape

    # create plot and draw image
    fig, plot = plt.subplots(1)
    plot.imshow(image)

    # draw the boxes
    for box in boxes:
        class_pred = box[5]
        box = box[1:5]
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (x1 * w, y1 * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none"
        )

        plot.add_patch(rect)
        plt.text(
            x1 * w,
            y1 * h,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()

def generate_unique_colors(n, saturation_range=(0.4, 1), lightness_range=(0.3, 1)):
    colors = []
    hue_range = 360 / n

    for i in range(n):
        hue = (i * hue_range) / 360.0
        saturation = random.uniform(saturation_range[0], saturation_range[1])
        lightness = random.uniform(lightness_range[0], lightness_range[1])

        rgb_color = hsv_to_rgb(hue, saturation, lightness)
        colors.append(rgb_color)

    return colors


def get_dataloaders(
    train_csv_path: str = None,
    test_csv_path: str = None,
    train_val_split: float=0.2,
    **kwargs
) -> dict[str, DataLoader]:
    from dataset import YOLODataset

    dataset_path = config.DATASET_PATH
    train_csv_path = train_csv_path or dataset_path / "train.csv"
    test_csv_path = test_csv_path or dataset_path / "test.csv"
    img_dir = config.IMAGES_DIR
    label_dir = config.LABELS_DIR

    anchors             = kwargs.get("anchors", config.ANCHORS)
    num_classes         = kwargs.get("num_classes", config.NUM_CLASSES)
    train_transforms    = kwargs.get("train_transforms", config.train_transforms)
    test_transforms     = kwargs.get("test_transforms", config.test_transforms)

    train_csv = pd.read_csv(train_csv_path)
    val_csv = train_csv.sample(frac=train_val_split)

    train_dataset = YOLODataset(
        train_csv,
        img_dir,
        label_dir,
        anchors,
        C=num_classes,
        transform=train_transforms,
        **kwargs
    )
    test_dataset = YOLODataset(
        test_csv_path,
        img_dir,
        label_dir,
        anchors,
        C=num_classes,
        transform=test_transforms,
        **kwargs
    )
    val_dataset = YOLODataset(
        val_csv,
        img_dir,
        label_dir,
        anchors,
        C=num_classes,
        transform=test_transforms,
        **kwargs
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    data_loaders = {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader
    }

    return data_loaders


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = nn.functional.max_pool2d(nn.functional.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x