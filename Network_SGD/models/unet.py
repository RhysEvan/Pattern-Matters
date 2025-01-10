# Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet
# A sigmoid function and a prepadding module were added to the UNet


import torch
import torch.nn as nn
import torch.nn.functional as func


class UNet(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, prepadding=4, feature_base=64,
                 expansive_path=False, kernel_size=3, scale_factor=2, pool_dilation=1, apply_sigmoid=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.prepadding = prepadding
        self.feature_base = feature_base
        self.expansive_path = expansive_path
        factor = 2 if bilinear else 1

        self.pad = nn.ReflectionPad2d(prepadding)
        self.inc = DoubleConv(n_channels, feature_base, kernel_size=kernel_size)
        self.down1 = Down(feature_base, feature_base*(scale_factor), kernel_size, scale_factor, pool_dilation)
        self.down2 = Down(feature_base*(scale_factor), feature_base*(scale_factor**2), kernel_size, scale_factor, pool_dilation)
        self.down3 = Down(feature_base*(scale_factor**2), feature_base*(scale_factor**3), kernel_size, scale_factor, pool_dilation)
        self.down4 = Down(feature_base*(scale_factor**3), feature_base*(scale_factor**4), kernel_size, scale_factor, pool_dilation)
        
        if self.expansive_path == True:
            self.down5 = Down(feature_base*(scale_factor**4), feature_base*(scale_factor**5), kernel_size, scale_factor, pool_dilation)
            self.up5 = Up(feature_base*(scale_factor**5), feature_base*(scale_factor**4) // factor, bilinear, kernel_size, scale_factor)
        
        self.up4 = Up(feature_base*(scale_factor**4), feature_base*(scale_factor**3) // factor, bilinear, kernel_size, scale_factor)
        self.up3 = Up(feature_base*(scale_factor**3), feature_base*(scale_factor**2) // factor, bilinear, kernel_size, scale_factor)
        self.up2 = Up(feature_base*(scale_factor**2), feature_base*(scale_factor) // factor, bilinear, kernel_size, scale_factor)
        self.up1 = Up(feature_base*(scale_factor), feature_base, bilinear, kernel_size, scale_factor)
        self.outc = OutConv(feature_base, n_classes)
        self.sigmoid = nn.Sigmoid() if apply_sigmoid else None

    def forward(self, x):
        x = self.pad(x)
        x = x1 = self.inc(x)
        x = x2 = self.down1(x)
        x = x3 = self.down2(x)
        x = x4 = self.down3(x)
        x = x5 = self.down4(x)
        
        if self.expansive_path == True:
            x = self.down5(x)
            x = self.up5(x, x5)

        x = self.up4(x, x4)    
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        
        # Apply sigmoid to return values between 0 and 1
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        return x
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=(kernel_size-1)//2, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=(kernel_size-1)//2, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, pool_dilation=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=scale_factor, dilation=pool_dilation),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, kernel_size=3, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // scale_factor, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)
            self.conv = DoubleConv(in_channels//scale_factor*2, out_channels, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = func.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

