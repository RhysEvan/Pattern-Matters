from models.unet import DoubleConv, OutConv, Down


import torch
import torch.nn as nn
import torch.nn.functional as func

class NestedUNet(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, prepadding=4, feature_base=64):
        super(NestedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.prepadding = prepadding
        self.feature_base = feature_base
        factor = 2 if bilinear else 1

        self.pad = nn.ReflectionPad2d(prepadding)
        self.inc = DoubleConv(n_channels, feature_base)

        self.down1 = Down(feature_base * 1, feature_base * 2)
        self.down2 = Down(feature_base * 2, feature_base * 4)
        self.down3 = Down(feature_base * 4, feature_base * 8)
        self.down4 = Down(feature_base * 8, feature_base * 16)

        self.up1 = nn.ModuleList([
            Up(feature_base * 2, feature_base // factor, bilinear),
            Up(feature_base * 2, feature_base // factor, bilinear, skip_connections = 1),
            Up(feature_base * 2, feature_base // factor, bilinear, skip_connections = 2),
            Up(feature_base * 2, feature_base // factor, bilinear, skip_connections = 3)
        ])
        
        self.up2 = nn.ModuleList([
            Up(feature_base * 4, feature_base * 2 // factor, bilinear),
            Up(feature_base * 4, feature_base * 2 // factor, bilinear, skip_connections = 1),
            Up(feature_base * 4, feature_base * 2 // factor, bilinear, skip_connections = 2)
        ])

        self.up3 = nn.ModuleList([
            Up(feature_base * 8, feature_base * 4 // factor, bilinear),
            Up(feature_base * 8, feature_base * 4 // factor, bilinear, skip_connections = 1)
        ])

        self.up4 = Up(feature_base * 16, feature_base * 8 // factor, bilinear)

        self.outc = OutConv(feature_base, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)

        x0_0 = self.inc(x)

        x1_0 = self.down1(x0_0)
        x0_1 = self.up1[0](x1_0, x0_0)

        x2_0 = self.down2(x1_0)
        x1_1 = self.up2[0](x2_0, x1_0)
        x0_2 = self.up1[1](x1_1, x0_0, x0_1)

        x3_0 = self.down3(x2_0)
        x2_1 = self.up3[0](x3_0, x2_0)
        x1_2 = self.up2[1](x2_1, x1_0, x1_1)
        x0_3 = self.up1[2](x1_2, x0_0, x0_1, x0_2)

        x4_0 = self.down4(x3_0)
        x3_1 = self.up4(x4_0, x3_0)
        x2_2 = self.up3[1](x3_1, x2_0, x2_1)
        x1_3 = self.up2[2](x2_2, x1_0, x1_1, x1_2)
        x0_4 = self.up1[3](x1_3, x0_0, x0_1, x0_2, x0_3)

        output = self.outc(x0_4)
        
        # Apply sigmoid to return values between 0 and 1
        output = self.sigmoid(output)
        return output
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, skip_connections=0):
        super().__init__()

        skip_channels = skip_connections*out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, *xn):
        x1 = self.up(x1)
        # input is CHW
        diffY = xn[0].size()[2] - x1.size()[2]
        diffX = xn[0].size()[3] - x1.size()[3]

        x1 = func.pad(x1, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
        x = torch.cat([*xn, x1], dim=1)
        return self.conv(x)