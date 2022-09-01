from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor


def conv3x3(
            in_planes: int, 
            out_planes: int, 
            stride: int = 1, 
            groups: int = 1, 
            dilation: int = 1, 
            bias: bool = False
        ) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if norm_layer == nn.Identity:
            bias = True
        else:
            bias = False
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.relu = nn.ReLU6(True)
        # self.relu = nn.SiLU(inplace=True)
        # self.relu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes)
        if inplanes == planes * self.expansion and stride == 1:
            self.downsample = None
        else:
            conv = conv1x1 if stride == 1 else conv3x3
            self.downsample = nn.Sequential(
                conv(inplanes, planes * self.expansion, bias=bias, stride=stride),
                norm_layer(planes * self.expansion)
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TensorFormatter(nn.Module):
    """Helper to format, apply operation, format back tensor.

    Class to format tensors of shape B x D x C_i x H x W into B*D x C_i x H x W,
    apply an operation, and reshape back into B x D x C_o x H x W.

    Used for multidepth - batching feature extraction on source images"""

    def __init__(self):
        super().__init__()

        self.batch_size = None
        self.depth_chns = None

    def _expand_batch_with_channels(self, x):
        if x.dim() != 5:
            raise ValueError('TensorFormatter expects tensors with 5 dimensions, '
                             'not {}!'.format(len(x.shape)))
        self.batch_size, self.depth_chns, chns, height, width = x.shape
        x = x.view(self.batch_size * self.depth_chns, chns, height, width)
        return x

    def _reduce_batch_to_channels(self, x):
        if self.batch_size is None or self.depth_chns is None:
            raise ValueError('Cannot  call _reduce_batch_to_channels without first calling'
                             '_expand_batch_with_channels!')
        _, chns, height, width = x.shape
        x = x.view(self.batch_size, self.depth_chns, chns, height, width)
        return x

    def forward(self, x, apply_func):
        x = self._expand_batch_with_channels(x)
        x = apply_func(x)
        x = self._reduce_batch_to_channels(x)
        return x
