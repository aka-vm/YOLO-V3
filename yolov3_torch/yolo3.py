"""
Pytorch implementation of YOLOv3 and its variants.
"""

import torch
import torch.nn as nn

# from .utils import (
#     anchor_box_convert
# )
# from .config import (
#     ANCHORS,
#     NUM_CLASSES,
#     CLASS_LABELS
# )


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalize: bool=True,
        stride: int=1,
        **kwargs
    ):
        super().__init__()
        _padding = "same"
        if stride == 2:
            _padding = 1

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=_padding, bias=not normalize, **kwargs)
        if normalize:
            self._bn = nn.BatchNorm2d(out_channels)
            self._lrelu = nn.LeakyReLU(0.1)
        self._normalize = normalize

    def forward(self, x):
        if self._normalize:
            return self._lrelu(self._bn(self.conv(x)))
        else:
            return self.conv(x)


class YOLOv3(nn.Module):
    """
    Base model for YOLOv3 architecture.
    """

    def __init__(
        self,
        input_shape: tuple = (416, 416, 3),
        num_classes: int = 20,
        initial_filters: int = 32,
        base_output_scale: tuple = (13, 13),
        anchors: list = None,
    ):
        super().__init__()
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._initial_filters = initial_filters
        self._base_output_scale = base_output_scale
        self._anchors = anchors
        self._layers = self._create_model()

    def forward(self, x):
        outputs = []
        route_layers = []

        for layer_name, layer in self._layers.items():
            if layer_name.startswith("output"):
                outputs.append(layer(x))
                continue

            x = layer(x)
            if layer_name in ("residual_3", "residual_4"):
                route_layers.append(x)
            elif layer_name.startswith("upsampling"):
                x = torch.cat([x, route_layers.pop()], dim=1)

        return outputs

    def _create_model(self):
        """
        Creates the model architecture.
        """
        FILTERS = self._initial_filters
        layers = dict(
        # Darknet-53 starts
            conv_1 = ConvBlock(3, FILTERS, 3),
            # Downsample 1
            downsample_1 = ConvBlock(FILTERS, FILTERS*2, 3, stride=2),
            residual_1 = self._ResidualBlock(FILTERS*2, 1),
            # Downsample 2
            downsample_2 = ConvBlock(FILTERS*2, FILTERS*4, 3, stride=2),
            residual_2 = self._ResidualBlock(FILTERS*4, 2),
            # Downsample 3
            downsample_3 = ConvBlock(FILTERS*4, FILTERS*8, 3, stride=2),
            residual_3 = self._ResidualBlock(FILTERS*8, 8),
            # Downsample 4
            downsample_4 = ConvBlock(FILTERS*8, FILTERS*16, 3, stride=2),
            residual_4 = self._ResidualBlock(FILTERS*16, 8), # Route 2
            # Downsample 5
            downsample_5 = ConvBlock(FILTERS*16, FILTERS*32, 3, stride=2),
            residual_5 = self._ResidualBlock(FILTERS*32, 4), # Route 1
        # Darknet-53 ends

            conv_pass_1 = self._ConvPassBlock(FILTERS*32, FILTERS*16),
            output_1 = self._OutputBlock(FILTERS*16, self._num_classes),

            conv_2 = ConvBlock(FILTERS*16, FILTERS*8, 1),
            upsampling_1 = nn.Upsample(scale_factor=2), # Route 1
            # Join 1
            conv_pass_2 = self._ConvPassBlock(FILTERS*24, FILTERS*8),
            output_2 = self._OutputBlock(FILTERS*8, self._num_classes),

            conv_3 = ConvBlock(FILTERS*8, FILTERS*4, 1),
            upsampling_2 = nn.Upsample(scale_factor=2), # Route 2
            # Join 2
            conv_pass_3 = self._ConvPassBlock(FILTERS*12, FILTERS*4),
            output_3 = self._OutputBlock(FILTERS*4, self._num_classes),
        )

        return layers

    class _ResidualBlock(nn.Module):
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

    class _ConvPassBlock(nn.Module):
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

    class _OutputBlock(nn.Module):
        """
        Output block for YOLOv3.
        parameters:
            in_channels: number of input channels
            out_channels: number of output channels
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

class TinyYOLOv3:
    """
    Tiny YOLOv3 architecture.
    """

if __name__ == "__main__":
    num_classes = 20
    # class_labels = CLASS_LABELS
    IMAGE_SIZE = 416
    initial_filters = 32

    input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    model = YOLOv3(input_shape, num_classes, initial_filters)
    x = torch.randn(2, *input_shape)
    out = model(x)

    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, 5 + num_classes)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, 5 + num_classes)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, 5 + num_classes)

    print("Test passed.")


