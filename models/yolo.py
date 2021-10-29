import torch
import torch.nn as nn
from .layers import * 


class CSPDarknet(nn.Module):
    def __init__(self, width_multiple=0.5, depth_multiple=0.33):
        super().__init__()
        
        feature_channels = [64, 128, 256, 512, 1024]
        depths = [3, 6, 9, 3]

        self.feature_channels = [make_divisible(c * width_multiple, 8) for c in feature_channels]
        self.depths = [max(round(depth * depth_multiple), 1) for depth in depths]

        self.conv1 = Conv(3, self.feature_channels[0], kernel_size=6, stride=2, padding=2)

        self.stage1 = nn.Sequential(
            Conv(self.feature_channels[0], self.feature_channels[1], kernel_size=3, stride=2),
            CSPBottleneck(self.feature_channels[1], self.feature_channels[1], depths=self.depths[0]),
            Conv(self.feature_channels[1], self.feature_channels[2], kernel_size=3, stride=2),
            CSPBottleneck(self.feature_channels[2], self.feature_channels[2], depths=self.depths[1])
        )

        self.stage2 = nn.Sequential(
            Conv(self.feature_channels[2], self.feature_channels[3], kernel_size=3, stride=2),
            CSPBottleneck(self.feature_channels[3], self.feature_channels[3], depths=self.depths[2])
        )

        self.stage3 = nn.Sequential(
            Conv(self.feature_channels[3], self.feature_channels[4], kernel_size=3, stride=2),
            CSPBottleneck(self.feature_channels[4], self.feature_channels[4], depths=self.depths[3])
        )

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)

        return outs


class Yolo(nn.Module):
    def __init__(self, anchors, num_classes, width_multiple, depth_multiple):
        super().__init__()

        self.backbone = CSPDarknet(width_multiple=width_multiple, depth_multiple=depth_multiple)
        feature_channels = self.backbone.feature_channels
        self.spp = SPPFast(feature_channels[-1], feature_channels[-1])

        self.panet = PANet(in_channels=feature_channels[-3:], depth_multiple=depth_multiple)

        self.head = Head(num_classes=num_classes, anchors=anchors, ch=feature_channels[-3:])

    def forward(self, x):
        xs = self.backbone(x)
        xs[2] = self.spp(xs[2])

        xs = self.panet(xs)

        out = self.head(xs)

        return out 
