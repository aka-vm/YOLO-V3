import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        normalize=True,
        downsample=False,
        **kwargs
    ):
        pass

    def forward(self, x):
        pass




class ResidualBlock(nn.Module):

    def __init__(
        self,
        filters,
        repeats=1,
        **kwargs
        ):
        pass

    def forward(self, x):
        pass


class ConvPassBlock(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class OutputBlock(nn.Module):

        def __init__(self):
            pass

        def forward(self, x):
            pass


class YoloV3(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass



if __name__ == '__main__':
    num_classes = 20
    IMAGE_H, IMAGE_W = 416, 416
    initial_filters = 32

    raise NotImplementedError("Implement the model")

    model = YoloV3(

    )

    inputs = torch.randn((2, 3, IMAGE_H, IMAGE_W))
    op = model(inputs)

    assert op[0].shape == (2, 3, IMAGE_H//32, IMAGE_W//32, num_classes + 5)
    assert op[1].shape == (2, 3, IMAGE_H//16, IMAGE_W//16, num_classes + 5)
    assert op[2].shape == (2, 3, IMAGE_H//8, IMAGE_W//8, num_classes + 5)


