import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
#from einops import rearrange
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.init import trunc_normal_

from itertools import repeat
import collections.abc
from swin_unet import SwinTransformerSys


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


class SENet(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SENet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)  # Dimension reduction
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)  # Restore dimensions
        self.sigmoid = nn.Sigmoid()  # Compress weights to between 0 and 1

    def forward(self, x):
        b, c, _, _ = x.size()  # Get batch size and number of channels
        y = self.global_avg_pool(x).view(b, c)  # Shape: (batch_size, in_channels)
        y = self.relu(self.fc1(y))  # Shape: (batch_size, in_channels // reduction)
        y = self.fc2(y)  # Shape: (batch_size, in_channels)
        y = self.sigmoid(y).view(b, c, 1, 1)  # Shape: (batch_size, in_channels, 1, 1)
        return x * y  # Apply channel-wise weights


class CoTCN(nn.Module):
    """
    Global-Local Fusion Net (CoTCN)
    """
    def __init__(self, in_ch=3, out_ch=1, fusion_dim=24):
        super(CoTCN, self).__init__()
        self.unet =  UNet(in_ch=in_ch, out_ch=fusion_dim)
        self.swin = SwinTransformerSys(img_size=(720, 1440), patch_size=4, in_chans=in_ch, num_classes=fusion_dim, ape=True,
                 embed_dim=96, depths=[2, 2, 2], depths_decoder=[1, 2, 2], num_heads=[3, 6, 12], drop_rate=0.2, attn_drop_rate=0.2,
                 window_size=5)
        self.se_module = SENet(in_channels=fusion_dim*2)
        self.last_conv = nn.Conv2d(fusion_dim*2, out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x_local = self.unet(x)
        x_global = self.swin(x)
        x = torch.concat([x_local, x_global], axis=1)
        x = self.se_module(x)
        out = self.last_conv(x)
    
        return out

