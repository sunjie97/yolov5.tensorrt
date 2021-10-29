import math 
import warnings
import torch
from torch._C import device
import torch.nn as nn


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding=padding if padding else kernel_size//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        c_ = int(c2 * expansion)
        self.conv1 = Conv(c1, c_, kernel_size=1)
        self.conv2 = Conv(c_, c2, kernel_size=3, groups=groups)

        self.shortcut = shortcut and c1 == c2 

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class CSPBottleneck(nn.Module):
    """ CSP Bottleneck with 3 convolutions """
    def __init__(self, c1, c2, depths=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        c_ = int(c2 * expansion)
        self.conv1 = Conv(c1, c_, kernel_size=1)
        self.conv2 = Conv(c1, c_, kernel_size=1)
        self.conv3 = Conv(2 * c_, c2, kernel_size=1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(c_, c_, shortcut, groups, expansion=1.) for _ in range(depths)])

    def forward(self, x):

        x = torch.cat([self.bottlenecks(self.conv1(x)), self.conv2(x)], dim=1)
        x = self.conv3(x)

        return x 


class SPPFast(nn.Module):
    """ Spatial Pyramid Pooling - Fast layer for YOLOv5 by Glenn Jocher """
    def __init__(self, c1, c2, kernel_size=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()

        c_ = c1 // 2 
        self.conv1 = Conv(c1, c_, kernel_size=1)
        self.conv2 = Conv(c_ * 4, c2, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv1(x)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
             
            y1 = self.pool(x)
            y2 = self.pool(y1)

            return self.conv2(torch.cat([x, y1, y2, self.pool(y2)], 1))


class Concat(nn.Module):
    """ Concatenate a list of tensors along dimension """
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class PANet(nn.Module):

    def __init__(self, in_channels, depth_multiple=0.33):
        super().__init__()

        depth = max(round(3 * depth_multiple), 1) 

        self.p5_upconv = Conv(in_channels[2], in_channels[1], kernel_size=1)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p4_upcat = nn.Sequential(
            Concat(),
            CSPBottleneck(in_channels[1]*2, in_channels[1], depths=depth, shortcut=False)
        )
        self.p4_upconv = Conv(in_channels[1], in_channels[0], kernel_size=1)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p3_upcat = nn.Sequential(
            Concat(),
            CSPBottleneck(in_channels[0]*2, in_channels[0], depths=depth, shortcut=False)
        )

        self.p3_downconv = Conv(in_channels[0], in_channels[0], kernel_size=3, stride=2)
        self.p4_downcat = nn.Sequential(
            Concat(),
            CSPBottleneck(in_channels[0]*2, in_channels[1], depths=depth, shortcut=False)
        )
        self.p4_downconv = Conv(in_channels[1], in_channels[1], kernel_size=3, stride=2)
        self.p5_downcat = nn.Sequential(
            Concat(),
            CSPBottleneck(in_channels[1]*2, in_channels[2], depths=depth, shortcut=False)
        )

    def forward(self, xs):
        x0, x1, x2 = xs  

        p5 = self.p5_upconv(x2)
        p5_up = self.p5_upsample(p5)
        p4 = self.p4_upcat([p5_up, x1])
        p4 = self.p4_upconv(p4)
        p4_up = self.p4_upsample(p4)
        p3 = self.p3_upcat([p4_up, x0])
        p3_down = self.p3_downconv(p3)
        p4 = self.p4_downcat([p3_down, p4])
        p4_down = self.p4_downconv(p4)
        p5 = self.p5_downcat([p4_down, p5])

        return p3, p4, p5


class Head(nn.Module):

    def __init__(self, num_classes=80, anchors=(), ch=()):
        super().__init__()

        self.num_classes = num_classes
        self.out_channels = int(num_classes) + 5 
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2 

        self.grid = [torch.zeros(1)] * self.num_layers 
        self.anchor_grid = [torch.zeros(1)] * self.num_layers 
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_layers, -1, 2))
        self.stride = torch.tensor([8, 16, 32], dtype=torch.float32, device=self.anchors.device)
        self.anchors /= self.stride.view(-1, 1, 1)
        
        self.outs = nn.ModuleList(nn.Conv2d(c, self.out_channels * self.num_anchors, kernel_size=1) for c in ch)

    def forward(self, x):
        x = list(x)

        z = [] 
        for i in range(self.num_layers):
            x[i] = self.outs[i](x[i])

            bs, _, ny, nx = x[i].shape  
            x[i] = x[i].view(bs, self.num_anchors, self.out_channels, ny, nx).permute(0, 1, 3, 4, 2).contiguous() 

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  
            y = torch.cat([xy, wh, y[..., 4:]], -1)

            z.append(y.view(-1, int(y.size(1)*y.size(2)*y.size(3)), self.out_channels))

        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20, i=0):

        device = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(device), torch.arange(nx).to(device)])
        grid = torch.stack([xv, yv], 2).expand((1, self.num_anchors, ny, nx, 2)).float().to(device)
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.num_anchors, 1, 1, 2))\
            .expand((1, self.num_anchors, ny, nx, 2)).float().to(device)

        return grid, anchor_grid 
