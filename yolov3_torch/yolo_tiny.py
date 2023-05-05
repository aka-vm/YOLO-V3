import torch
import torch.nn as nn
from torchinfo import summary

import math

from utils import (
    anchor_box_convert,
    MaxPoolStride1
)
from config import (
    ANCHORS,
    NUM_CLASSES,
    CLASS_LABELS
)

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalize: bool=True,
        stride: int=1,
        padding = "same",
        **kwargs
    ):
        super().__init__()
        if padding == "same" and stride > 1:
            padding = math.ceil((kernel_size - stride) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not normalize, **kwargs)
        if normalize:
            self._bn = nn.BatchNorm2d(out_channels)
            self._lrelu = nn.LeakyReLU(0.1)
        self._normalize = normalize

    def forward(self, x):
        if self._normalize:
            return self._lrelu(self._bn(self.conv(x)))
        else:
            return self.conv(x)



class TinyYOLOv3(nn.Module):
    """
    Tiny YOLOv3 architecture.
    Ignore this for now.

    it's wrong.
    route 1 was suposed to be connecting #8 and #19, instead it connects #9 and #19.
    I should'nt have used _ConvMaxPollBlock.
    """

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        initial_filters: int=16,
        base_output_scale: tuple=(13, 13),
        anchors: list = None,
    ):
        """
        parameters:
            input_shape: shape of the input image
            num_classes: number of classes
        """
        super().__init__()
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._initial_filters = initial_filters
        self._base_output_scale = base_output_scale
        self._anchors = anchors

        self.layers = self._create_model()

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, self._OutputBlock):
                outputs.append(layer(x))
                continue

            x = layer(x)
            if i == 3:
                route_1 = x
            elif isinstance(layer, nn.Upsample):
                x = torch.cat((x, route_1), dim=1)

        return outputs

    def summary(self, verbose: int=1, **kwargs):
        """
        Prints the summary of the model.
        """
        summary(
            self,
            input_size=self._input_shape,
            batch_dim=0,
            col_names = ("input_size", "output_size", "num_params", "kernel_size"),
            verbose = verbose,
            **kwargs
            )
        # print
        # summary(self, input_size=(1, *self._input_shape))

    def _create_model(self):

        FILTERS = self._initial_filters
        layers = nn.ModuleList([
        # Darknet-19 starts
            self._ConvMaxPollBlock(in_channels=3, out_channels=FILTERS),
            self._ConvMaxPollBlock(in_channels=FILTERS, out_channels=FILTERS*2),
            self._ConvMaxPollBlock(in_channels=FILTERS*2, out_channels=FILTERS*4),
            self._ConvMaxPollBlock(in_channels=FILTERS*4, out_channels=FILTERS*8),      # route_1
            self._ConvMaxPollBlock(in_channels=FILTERS*8, out_channels=FILTERS*16),
            self._ConvMaxPollBlock(in_channels=FILTERS*16, out_channels=FILTERS*32, use_max_poll_stride=1),
            ConvBlock(in_channels=FILTERS*32, out_channels=FILTERS*64, kernel_size=3),
        # Darknet-19 ends
            ConvBlock(in_channels=FILTERS*64, out_channels=FILTERS*16, kernel_size=1),
            self._OutputBlock(in_channels=FILTERS*16, channels=FILTERS*32, num_classes=self._num_classes),

            ConvBlock(in_channels=FILTERS*16, out_channels=FILTERS*8, kernel_size=1),
            nn.Upsample(scale_factor=2), # Route 1
            # Join 1
            self._OutputBlock(in_channels=FILTERS*16, channels=FILTERS*16, num_classes=self._num_classes),
        ])

        return layers
    class _ConvMaxPollBlock(nn.Module):
        """
        Convolutional + MaxPool block for Tiny YOLOv3.
        parameters:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride for MaxPool
        """
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_max_poll_stride: int=2,
        ):
            MaxPoll = nn.MaxPool2d(kernel_size=2, stride=2) if use_max_poll_stride==2 else MaxPoolStride1()
            super().__init__()
            self._layers = nn.Sequential(
                ConvBlock(in_channels, out_channels, 3),
                MaxPoll
            )

        def forward(self, x):
            return self._layers(x)

    class _OutputBlock(nn.Module):
        """
        Output block for Tiny YOLOv3.
        parameters:
            in_channels: number of input channels
            channels: number of channels in the output of first ConvBlock
            num_classes: number of classes
        """
        def __init__(
            self,
            in_channels: int,
            channels: int,
            num_classes: int,
        ):
            super().__init__()
            self._layers = nn.Sequential(
                ConvBlock(in_channels, channels, 3),
                ConvBlock(
                    channels,
                    (5 + num_classes)*3,
                    kernel_size=1,
                    normalize=False
                )
            )
            self.num_classes = num_classes

        def forward(self, x):
            return (
                self._layers(x)
                .reshape(x.shape[0], 3, 5 + self.num_classes, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
            )


if __name__ == "__main__":
    num_classes = NUM_CLASSES
    class_labels = CLASS_LABELS
    IMAGE_SIZE = 416
    initial_filters = 16

    input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    model = TinyYOLOv3(input_shape, num_classes, initial_filters)
    x = torch.randn(2, *input_shape)
    out = model(x)

    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, 5 + num_classes)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, 5 + num_classes)
    print("TinyYOLOv3 Test passed.\n"+"-"*15)