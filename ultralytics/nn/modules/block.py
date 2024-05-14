# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.init import constant_
from ultralytics.utils import LOGGER

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock
from .dcn.ops_dcnv3.modules.dcnv3 import _is_power_of_2, CenterFeatureScaleModule
from .dcn.ops_dcnv3.functions import DCNv3Function, dcnv3_core_pytorch
try:
    from .dcn.DCNv4_op.DCNv4.modules import DCNv4
    use_dcnv4 = True
except ModuleNotFoundError:
    use_dcnv4 = False


__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'BottleneckDCNv3', 'BottleneckDCNv4', 'RepNBottleneck',
           'Proto', 'RepC3', 'RepNCSP', 'RepConvN', 'DCNv3SingleConv', 'DCNv3C2f', 'DCNv4C2f', 'PyramidUpfusion',
           'PyramidDownfusion', 'PyramidUpfusionv2', 'PyramidDownfusionv2', 'PyramidfusionP4', 'MultGLPAN', 'MultGLFPN',
           'MultGLPANc', 'MultGLFPNc', 'RepNCSPELAN4', 'ADown', 'DySample')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2)
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0)
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0)
        self.stem3 = Conv(cm * 2, cm, 3, 2)
        self.stem4 = Conv(cm, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class DCNv3C2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=4, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckDCNv3(self.c, self.c, shortcut, g, k=(3, 3)) for _ in range(n))


class DCNv4C2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=4, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(BottleneckDCNv4(self.c, self.c, shortcut, g, k=(3, 3)) for _ in range(n))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class BottleneckDCNv3(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=4, k=(3, 3)):
        super().__init__()
        self.cv1 = Conv(c1, c2, k[0], 1)
        self.cv2 = DCNv3SingleConv(c2, k[1], group=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckDCNv4(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=4, k=(3, 3)):
        super().__init__()
        self.cv1 = Conv(c1, c2, k[0], 1)
        if use_dcnv4:
            self.cv2 = DCNv4(c2, k[1], group=g, without_pointwise=True)
        else:
            self.cv2 = DCNv3SingleConv(c2, k[1], group=g)
            LOGGER.warning('WARNING ⚠ : Unable to use DCNv4, use DCNv3 instead')
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def normal_init(module, mean=0, std=1.0, bias=.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.contiguous().view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class PyramidUpfusion(nn.Module):
    """
        (p3,p4,p5) fusion;
        (p4，p5) feature sample towards p3 level;
        Early experimental version: up-sample bilinear, down-sample avgpool
    """
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_c[0], out_c, act=act) if in_c[0] != out_c else nn.Identity()
        self.conv2 = Conv(in_c[1], out_c, act=act) if in_c[1] != out_c else nn.Identity()
        self.conv3 = Conv(in_c[2], out_c, act=act) if in_c[2] != out_c else nn.Identity()
        self.convf = Conv(out_c * 3, out_c, act=act)
        self.sample = partial(F.interpolate, mode='bilinear', align_corners=False)

    def forward(self, x):
        _, _, h, w = x[0].shape
        x0 = self.conv1(x[0])
        x1 = self.conv2(self.sample(x[1], (h, w)))
        x2 = self.conv3(self.sample(x[2], (h, w)))
        return self.convf(torch.cat([x0, x1, x2], dim=1))


class PyramidDownfusion(nn.Module):
    """
        (p3,p4,p5) fusion;
        (p3，p4) feature sample towards p5 level;
        Early experimental version: up-sample bilinear, down-sample avgpool
    """
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_c[0], out_c, act=act) if in_c[0] != out_c else nn.Identity()
        self.conv2 = Conv(in_c[1], out_c, act=act) if in_c[1] != out_c else nn.Identity()
        self.conv3 = Conv(in_c[2], out_c, act=act) if in_c[2] != out_c else nn.Identity()
        self.convf = Conv(out_c * 3, out_c, act=act)
        self.sample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        _, _, h, w = x[2].shape
        x0 = self.conv1(self.sample(x[0], (h, w)))
        x1 = self.conv2(self.sample(x[1], (h, w)))
        x2 = self.conv3(x[2])
        return self.convf(torch.cat([x0, x1, x2], dim=1))


class PyramidUpfusionv2(nn.Module):
    """
        (p3,p4,p5) fusion;
        (p4，p5) feature upsample towards p3 level;
        Final version;
    """
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_c[0], out_c, act=act)
        self.conv2 = Conv(in_c[1], out_c, act=act)
        self.conv3 = Conv(in_c[2], out_c, act=act)
        self.convf = Conv(out_c * 3, out_c, act=act)
        self.sample1 = DySample(out_c, 2, 'lp', groups=4)
        self.sample2 = DySample(out_c, 4, 'lp', groups=4)

    def forward(self, x):
        x0 = self.conv1(x[0])
        x1 = self.sample1(self.conv2(x[1]))
        x2 = self.sample2(self.conv3(x[2]))
        return self.convf(torch.cat([x0, x1, x2], dim=1))


class PyramidDownfusionv2(nn.Module):
    """
        (p3,p4,p5) fusion;
        (p3，p4) feature downsample towards p5 level;
        Final version;
    """
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_c[0], out_c, act=act)
        self.conv2 = Conv(in_c[1], out_c, act=act)
        self.conv3 = Conv(in_c[2], out_c, act=act)
        self.convf = Conv(out_c * 3, out_c, act=act)
        self.sample1 = DCNv3SingleConv(out_c, 3, 4)
        self.sample2 = DCNv3SingleConv(out_c, 3, 2)

    def forward(self, x):
        x0 = self.sample1(self.conv1(x[0]))
        x1 = self.sample2(self.conv2(x[1]))
        x2 = self.conv3(x[2])
        return self.convf(torch.cat([x0, x1, x2], dim=1))


class PyramidfusionP4(nn.Module):
    """
        (p3,p4,p5) fusion;
        (p3，p5) feature downsample towards p4 level;
        Final version;
    """
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_c[0], out_c, act=act)
        self.conv2 = Conv(in_c[1], out_c, act=act)
        self.conv3 = Conv(in_c[2], out_c, act=act)
        self.convf = Conv(out_c * 3, out_c, act=act)
        self.sample1 = DCNv3SingleConv(out_c, 3, 2)
        self.sample2 = DySample(out_c, 2, 'lp', groups=4)

    def forward(self, x):
        x0 = self.sample1(self.conv1(x[0]))
        x1 = self.conv2(x[1])
        x2 = self.sample2(self.conv3(x[2]))
        return self.convf(torch.cat([x0, x1, x2], dim=1))


class MultGLPAN(nn.Module):
    """
    x is a list for [x_local, x_global, Option(x_low)]
    if at P4 or P5 level, flag will be True and x has the feature from the low level[x_low]
    Early experimental version: up-sample bilinear, down-sample avgpool
    """
    def __init__(self, in_channel, out_channel, flag=False):
        super().__init__()
        self.flag = flag
        if flag:
            self.local_conv = Conv(in_channel[0]+in_channel[2], out_channel)
        else:
            self.local_conv = Conv(in_channel[0], out_channel)
        self.global_conv = Conv(in_channel[1], out_channel, act=nn.Sigmoid())

    def forward(self, x):
        if self.flag:
            x_local, x_global, x_low = x
            _, _, h, w = x_local.shape
            x_global = nn.functional.adaptive_avg_pool2d(x_global, (h, w))
            x_low = nn.functional.adaptive_avg_pool2d(x_low, (h, w))
            x_local = torch.cat([x_local, x_low], dim=1)
        else:
            x_local, x_global = x
            _, _, h, w = x_local.shape
            if x_global.shape[2] != h:
                x_global = nn.functional.adaptive_avg_pool2d(x_global, (h, w))
        x_local = self.local_conv(x_local)
        x_global = self.global_conv(x_global)
        return x_global + (x_local * x_global)


class MultGLFPN(nn.Module):
    """
    x is a list for [x_local, x_global, Option(x_high)]
    if at P3 or P4 level, flag will be True and x has the feature from the high level[x_high]
    Early experimental version: up-sample bilinear, down-sample avgpool
    """
    def __init__(self, in_channel, out_channel, flag=False):
        super().__init__()
        self.flag = flag
        if flag:
            self.local_conv = Conv(in_channel[0]+in_channel[2], out_channel)
        else:
            self.local_conv = Conv(in_channel[0], out_channel)
        self.global_conv = Conv(in_channel[1], out_channel, act=nn.Sigmoid())

    def forward(self, x):
        if self.flag:
            x_local, x_global, x_high = x
            _, _, h, w = x_local.shape
            x_global = F.interpolate(x_global, (h, w), mode='bilinear', align_corners=False)
            x_high = F.interpolate(x_high, (h, w), mode='bilinear', align_corners=False)
            x_local = torch.cat([x_local, x_high], dim=1)
        else:
            x_local, x_global = x
            _, _, h, w = x_local.shape
            if x_global.shape[2] != h:
                x_global = F.interpolate(x_global, (h, w), mode='bilinear', align_corners=False)
        x_local = self.local_conv(x_local)
        x_global = self.global_conv(x_global)
        return x_global + (x_local * x_global)


class MultGLPANc(nn.Module):
    """
    x is a list for [x_local, x_global, Option(x_low)]
    if at P4 or P5 level, flag will be True and x has the feature from the low level[x_low]
    Final version;
    """
    def __init__(self, in_channel, out_channel, global_scale, flag=False, local_res=True):
        super().__init__()
        self.flag = flag
        self.local_res = local_res
        if flag:
            self.low_sample = DCNv3SingleConv(in_channel[2], 3, 2)
            self.local_conv = Conv(in_channel[0]+in_channel[2], out_channel)
        else:
            self.local_conv = Conv(in_channel[0], out_channel)
        assert global_scale in [-4, -2, 1, 2, 4]
        if global_scale == -4:
            self.global_sample = nn.Sequential(DCNv3SingleConv(in_channel[1], 3, 2),
                                               Conv(in_channel[1], in_channel[1], 3, 2))
        elif global_scale == -2:
            self.global_sample = DCNv3SingleConv(in_channel[1], 3, 2)
        elif global_scale in [2, 4]:
            self.global_sample = DySample(in_channel[1], global_scale)
        self.global_conv = Conv(in_channel[1], out_channel)

    def forward(self, x):
        if self.flag:
            x_local, x_global, x_low = x
            x_low = self.low_sample(x_low)
            x_local = torch.cat([x_local, x_low], dim=1)
        else:
            x_local, x_global = x
        x_local = self.local_conv(x_local)
        if hasattr(self, 'global_sample'):
            x_global = self.global_conv(self.global_sample(x_global))
        else:
            x_global = self.global_conv(x_global)
        if self.local_res:
            return x_local + (x_local * x_global)
        else:
            return x_global + (x_local * x_global)


class MultGLFPNc(nn.Module):
    """
    x is a list for [x_local, x_global, Option(x_high)]
    if at P3 or P4 level, flag will be True and x has the feature from the high level[x_high]
    Final version;
    """
    def __init__(self, in_channel, out_channel, global_scale, flag=False, local_res=True):
        super().__init__()
        self.flag = flag
        self.local_res = local_res
        if flag:
            self.high_sample = DySample(in_channel[2], 2)
            self.local_conv = Conv(in_channel[0]+in_channel[2], out_channel)
        else:
            self.local_conv = Conv(in_channel[0], out_channel)
        assert global_scale in [-4, -2, 1, 2, 4]
        if global_scale == -4:
            self.global_sample = nn.Sequential(DCNv3SingleConv(in_channel[1], 3, 2),
                                               Conv(in_channel[1], in_channel[1], 3, 2))
        elif global_scale == -2:
            self.global_sample = DCNv3SingleConv(in_channel[1], 3, 2)
        elif global_scale in [2, 4]:
            self.global_sample = DySample(in_channel[1], global_scale)
        self.global_conv = Conv(in_channel[1], out_channel)

    def forward(self, x):
        if self.flag:
            x_local, x_global, x_high = x
            x_high = self.high_sample(x_high)
            x_local = torch.cat([x_local, x_high], dim=1)
        else:
            x_local, x_global = x
        x_local = self.local_conv(x_local)
        if hasattr(self, 'global_sample'):
            x_global = self.global_conv(self.global_sample(x_global))
        else:
            x_global = self.global_conv(x_global)
        if self.local_res:
            return x_local + (x_local * x_global)
        else:
            return x_global + (x_local * x_global)


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class DCNv3SingleConv(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            center_feature_scale=False,
            remove_center=False,
    ):
        """
        DCNv3 Single Conv
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        # self.pad = pad
        self.pad = kernel_size // 2
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = Conv(channels, channels, dw_kernel_size, g=channels)
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)

    def forward(self, x):
        """
        :param x                    (B, C, H, W)
        :return output              (B, C, H, W)
        """
        B, _, H, W = x.shape
        dtype = x.dtype
        # (B, C, H, W)
        x1 = self.dw_conv(x).permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        # x1 x: (B, H, W, C)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(B, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(B, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x.contiguous(), offset.contiguous(), mask.contiguous(),
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        # x = dcnv3_core_pytorch(
        #     x, offset, mask,
        #     self.kernel_size, self.kernel_size,
        #     self.stride, self.stride,
        #     self.pad, self.pad,
        #     self.dilation, self.dilation,
        #     self.group, self.group_channels,
        #     self.offset_scale, self.remove_center)
        return x.permute(0, 3, 1, 2)