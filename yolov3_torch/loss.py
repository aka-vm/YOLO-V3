"""
I assume you're familiar with the paper. If not, please read it first.

This was quite difficult to impliment.
For this I took reference from the paper and some previous implimentations by other people.

I think I have done it correctly how the paper says, but there are some places where things are not
clear in the paper. So, I took reference from other implimentations.

The loss function in the paper is makes less sense to me in some places. I understand they were seeing
a bigger picture but I'm not using `Open Images Dataset`. If I do something different I'll mention it.

If I ever get time, I will try to improve this part of the code. I know YoloV3 is not the best and is
very old, but I am doing this for the purpose of improving my paper implimentation and code skills.
Let's be honest, some people still impliment ANN from scratch, so I think this is fine.

! ALSO!
At the time of writing this code. I know It could have been done better.
But since this block depends on both the dataset and the model, I'm not going to change them just now.
"""

import torch
import torch.nn as nn

from utils import non_max_suppression, iou_boxes

class YoloLoss(nn.Module):
    """"
    The prediction is basically the model's barebone output tensor of shape([N, 3, S, S, 5+num_classes]).

    In the paper they did some opprations on it to get the final prediction. For now, I'll just use the
    barebone output and calculate the final predictions in the loss function. One of the main reason behind
    it is that instead of comparing ground truth with the final predictions, the authous invented the ground-
    truth coordinates.

    Let's call the model output as [t_obj, t_x, t_y, t_w, t_h, t_c_1, t_c_2, ...].
    and call final predictions as (p_obj, b_x, b_y, b_w, b_h, p_c_1, p_c2, ...).
    the ground truth is (obj, x, y, w, h, c_1, c_2, ...).

    Since there are 6 things we predict, we need multiple losses and cofficients for each of them.

    - obj(objectness): p_obj = sigmoid(t_obj)   ;logistic regression is used to predict it. If ground_truth 0, we ignore the rest.

    - w, h: b_w = pw*exp(t_w)          ; pw is the anchor box width
    - x, y: b_x = sigmoid(t_x)+cell_x  ; cell_x is the x coordinate of the cell

    - c_n: pc_n = sigmoid(t_c_n)       ; logistic regression is used to predict it.
        NOTE - softmax is not used, instead they used independent logistic clf. It helps in multi-labeling a bbox (e.g. rottwieler, dog, animal).
        but I'll use CrossEntropyLoss for now. Later I'll change it.

    Losses: I'll use mean reduction for all of them.
        - noobj_loss: bce(p_obj, 0)
            Binary Cross Entropy Loss.

        - obj_loss: bce(p_obj, iou(b_xywh, xywh))
            Binary Cross Entropy Loss; but instead of using 1 as target, we use IoU(target, prediction)
            This was not mentioned in the paper directly but in yolo9000 paper, something similar was done.
            The thing that is written makes no sense to me.

        - coord_loss: SquareError(b_xywh, xywh)
            I'll use MSE for this.

        - class_loss: CrossEntropyLoss(p_c, c)
            In the paper they used independent logistic clf and bce loss. But I'll use CrossEntropyLoss for now.

    """
    def __init__(self) -> None:
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box   = 5
        self.lambda_class = 1


    def forward(
        self,
        prediction,
        target,
        anchors,
        # device
    ):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Activation Layer; not what we call it but seems like a good name.
        prediction[...,0]   = self.sigmoid(prediction[..., 0])
        prediction[...,1:3] = self.sigmoid(prediction[..., 1:3])            # x,y coordinates
        prediction[...,3:5] = torch.exp   (prediction[..., 3:5]) * anchors  # width, height
        prediction[...,5:]  = self.sigmoid(prediction[..., 5:])             # class probabilities

        # Losses
        # No Object Loss: should me least
        noobj_loss = self.bce(
            prediction[..., 0][noobj],
            target    [..., 0][noobj]
            )

        # Object Loss
        ious = iou_boxes(
            prediction[..., 1: 5][obj],
            target    [..., 1: 5][obj]
        )
        object_loss = self.bce(
            prediction[..., 0:1][obj],
            ious
            )
        # Box Coordinates Loss
        box_loss = self.mse(
            prediction[..., 1:5][obj],
            target    [..., 1:5][obj]
            )
        # Class Loss #! check src
        class_loss = self.ce(
            prediction[..., 5:][obj],
            target    [..., 5:][obj],
        )


        return (
            self.lambda_noobj   * noobj_loss
            + self.lambda_obj   * object_loss
            + self.lambda_box   * box_loss
            + self.lambda_class * class_loss
        )
