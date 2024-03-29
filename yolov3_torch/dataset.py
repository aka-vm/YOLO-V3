"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import config
from utils import (
    cells_to_bboxes,
    iou_wh,
    format_anchor_boxes,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,   # Matrix of coordinate(0-1) tuples
        image_size=416,
        scales=(13, 26, 52),
        C=20,
        transform=None,
        ignore_iou_thresh = 0.5
    ):
        if not isinstance(csv_file, pd.DataFrame):
            csv_file = pd.read_csv(csv_file)
        self.annotations = csv_file
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.scales = scales
        self.C = C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])       # here it's 3 #! Modify later to make it general
        self.num_anchors = self.anchors.shape[0] # 9 for now
        self.num_anchors_per_scale = self.num_anchors // 3
        self.image_size = image_size
        self.ignore_iou_thresh = ignore_iou_thresh

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        label_path = self.label_dir / self.annotations.iloc[index, 1]
        # (x, y, w, h, c)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), -1, axis=1).tolist()
        img_path = self.img_dir / self.annotations.iloc[index, 0]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # taking 3 scaled predictions from the YOLOv3 model #! Modify later to make it general
        # targets shape - [tensor(3, S, S, 6),];
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.scales]
        #
        for box in bboxes:
            iou_anchors = iou_wh(torch.tensor(box[2:4]), self.anchors)
            anchors_rank = iou_anchors.argsort(descending=True, dim=0)
            x, y, w, h, class_label = box   # x, y, w, h are relative to the image size
    # It is not mentioned that we take atleast one (the best) anchor box for each scale or overall.
    # but I saw in some implementations that they take atleast one anchor box for each scale.
            scale_flag = [False, False, False]
            for i_anchor in anchors_rank:
                i_scale = int(i_anchor / self.num_anchors_per_scale)
                anchor_on_scale = i_anchor % self.num_anchors_per_scale
                # w_cell, h_cell = self.image_size / self.S[i_scale]
                scale = self.scales[i_scale]
                i, j = int(scale * y), int(scale * x)
                target_anchor_taken = targets[i_scale][anchor_on_scale, i, j, 0]
                if not target_anchor_taken and not scale_flag[i_scale]:
                    targets[i_scale][anchor_on_scale, i, j, 0] = 1

                    x_cell, y_cell = scale * x - j, scale * y - i
                    w_cell = w * scale
                    h_cell = h * scale
                    box_coordinates = torch.tensor([x_cell, y_cell, w_cell, h_cell])

                    targets[i_scale][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[i_scale][anchor_on_scale, i, j, 5] = int(class_label)
                    scale_flag[i_scale] = True

                elif not target_anchor_taken and iou_anchors[i_anchor] > self.ignore_iou_thresh:
    # It is written in the paper that if the anchor box is not best and has an IoU > 0.5 with the ground
    # truth box. Then we don't penalize the confidence score of that anchor box. But we don't assign any
    # objectness score to that anchor box either.
                    targets[i_scale][anchor_on_scale, i, j, 0] = -1 # will be ignored during loss calculation

        return image, tuple(targets)

def test(return_loader=False, plot=True):
    # init
    anchors = config.ANCHORS
    transform = config.test_transforms
    transform = config.train_transforms

    dataset_path = config.DATASET_PATH
    csv_file = dataset_path / "train.csv"
    csv_file = dataset_path / "8examples.csv"
    img_dir = config.IMAGES_DIR
    label_dir = config.LABELS_DIR
    annotations = pd.read_csv(csv_file)

    dataset = YOLODataset(
        csv_file,
        img_dir,
        label_dir,
        anchors=anchors,
        transform=transform
    )
    scales = (13, 26, 52)
    scaled_anchors = format_anchor_boxes(anchors, scales)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # init completed

    if return_loader:
        return data_loader, scaled_anchors,

    for img, bboxes in data_loader:    # bboxes is a tuple of 3 tensors
        boxes = []

        for i in range(bboxes[0].shape[1]):     # basically the number of different scales, like (13, 26, 52) in case of yoloV3
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(
                bboxes[i],
                is_preds=False,
                S=bboxes[i].shape[2],
                anchors=anchor
            )[0]
        boxes = nms(boxes, base_prob_threshold=0.7, iou_threshold=1, box_format="midpoint")

        objects = []
        for box in boxes:
            class_label = config.CLASS_LABELS[int(box[5])]
            objects.append(class_label)

        # print(objects)
        if plot:
            plot_image(img[0].permute(1, 2, 0).to("cpu"), boxes)
        break       # just for single testing



if __name__ == "__main__":
    test(plot=True)