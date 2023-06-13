"""
Pytorch implementation of YOLOv3 and its variants.
"""

import torch
import torch.nn as nn
from torchinfo import summary

import math

import config


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

class ResidualBlock(nn.Module):
    """
    Residual block for YOLOv3.
    parameters:
        in_channels: number of input channels
        out_channels: number of output channels
        reps: number of repetitions
    """

    def __init__(
        self,
        channels: int,
        repetes: int=1
    ):
        super().__init__()
        self._layers = nn.ModuleList()
        self.repetes = repetes

        for _ in range(repetes):
            self._layers.append(
                nn.Sequential(
                    ConvBlock(channels, channels//2, 1),
                    ConvBlock(channels//2, channels, 3)
                )
            )

    def forward(self, x):
        for layer in self._layers:
            x = x + layer(x)
        return x

class ConvPassBlock(nn.Module):
    """
    Convolutional pass block for YOLOv3. #! Will add the discription to readme later.
    parameters:
        in_channels: number of input channels
        out_channels: number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self._layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, 3),
            ConvBlock(out_channels, out_channels*2, 1),
            ConvBlock(out_channels*2, out_channels, 3),
            ConvBlock(out_channels, out_channels*2, 1),
            ConvBlock(out_channels*2, out_channels, 3),
        )

    def forward(self, x):
        return self._layers(x)

class OutputBlock(nn.Module):
    """
    Output block for YOLOv3.
    parameters:
        in_channels: number of input channels
        num_classes: number of classes
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self._layers = nn.Sequential(
            ConvBlock(in_channels, in_channels*2, 3),
            ConvBlock(
                in_channels*2,
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

class YOLOv3(nn.Module):
    """
    Base model for YOLOv3 architecture.
    """

    def __init__(
        self,
        input_shape: tuple[int] = (416, 416, 3),
        num_classes: int = 20,
        initial_filters: int = 32,
        base_output_scale: tuple[int] = (13, 13),
        anchors: list = None,
    ):
        super().__init__()
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._initial_filters = initial_filters
        self._base_output_scale = base_output_scale
        self._anchors = anchors

        self.layers = self._create_model()

    def forward(self, x):
        outputs = []
        route_layers = []

        for layer in self.layers:
            if isinstance(layer, OutputBlock):
                outputs.append(layer(x))
                continue

            x = layer(x)
            # if layer_name in ("residual_3", "residual_4"):
            if isinstance(layer, ResidualBlock) and layer.repetes == 8:
                route_layers.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_layers.pop()], dim=1)

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
        """
        Creates the model architecture.
        """
        FILTERS = self._initial_filters
        layers = nn.ModuleList([
        # Darknet-53 starts
            ConvBlock(3, FILTERS, 3),
            # Downsample 1
            ConvBlock(FILTERS, FILTERS*2, 3, stride=2),
            ResidualBlock(FILTERS*2, 1),
            # Downsample 2
            ConvBlock(FILTERS*2, FILTERS*4, 3, stride=2),
            ResidualBlock(FILTERS*4, 2),
            # Downsample 3
            ConvBlock(FILTERS*4, FILTERS*8, 3, stride=2),
            ResidualBlock(FILTERS*8, 8),
            # Downsample 4
            ConvBlock(FILTERS*8, FILTERS*16, 3, stride=2),
            ResidualBlock(FILTERS*16, 8), # Route 2
            # Downsample 5
            ConvBlock(FILTERS*16, FILTERS*32, 3, stride=2),
            ResidualBlock(FILTERS*32, 4), # Route 1
        # Darknet-53 ends

            ConvPassBlock(FILTERS*32, FILTERS*16),
            OutputBlock(FILTERS*16, self._num_classes),

            ConvBlock(FILTERS*16, FILTERS*8, 1),
            nn.Upsample(scale_factor=2), # Route 1
            # Join 1
            ConvPassBlock(FILTERS*24, FILTERS*8),
            OutputBlock(FILTERS*8, self._num_classes),

            ConvBlock(FILTERS*8, FILTERS*4, 1),
            nn.Upsample(scale_factor=2), # Route 2
            # Join 2
            ConvPassBlock(FILTERS*12, FILTERS*4),
            OutputBlock(FILTERS*4, self._num_classes),
        ])

        return layers

def test(
    num_classes: int = 20,
    class_labels: list = None,
    image_size: int = 416,
    initial_filters: int = 32,
    verbose: int = 0,
    return_model: bool = False,
):
    class_labels = config.CLASS_LABELS if not class_labels else class_labels

    input_shape = (3, image_size, image_size)
    model = YOLOv3(input_shape, num_classes, initial_filters)
    x = torch.randn(2, *input_shape)
    out = model(x)

    if return_model:
        targets = []
        anchors = torch.tensor(config.ANCHORS)
        anchors = anchors * \
            torch.reshape(torch.tensor(config.S), (3, 1, 1)).repeat(1, 3, 2)


        for i, out_ in enumerate(out):
            anchor_ = anchors[i].reshape(1, 3, 1, 1, 2)
            target = torch.zeros_like(out_[..., :6])
            # target = out_.clone()
            target[..., 0] = torch.sigmoid(target[..., 0]).round()
            target[..., 1:3] = torch.sigmoid(target[..., 1:3])
            target[..., 3:5] = torch.exp(target[..., 3:5]) * anchor_
            label = torch.argmax(out_[..., 5:], dim=-1)
            target[..., 5] = label

            targets.append(target)

        return model, out, targets, anchors

    assert out[0].shape == (2, 3, image_size//32, image_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, image_size//16, image_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, image_size//8, image_size//8, 5 + num_classes)
    print("YOLOv3 Test passed.\n"+"-"*15)
    model.summary(verbose=verbose)
    print("\n")

if __name__ == "__main__":
    test(verbose=1)
