from models.unet import DoubleConv, OutConv, Down, Up

import torch.nn as nn

class YNet(nn.Module):
    
    def __init__(self, n_channels=1, n_classes1=1, n_classes2=1, bilinear=False, prepadding=4, feature_base=64,
                 expansive_path=True, kernel_size=3, scale_factor=2):
        super(YNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes1 = n_classes1  # Number of output channels for decoder 1
        self.n_classes2 = n_classes2  # Number of output channels for decoder 2
        self.bilinear = bilinear
        self.prepadding = prepadding
        self.feature_base = feature_base
        self.expansive_path = expansive_path
        factor = 2 if bilinear else 1

        self.pad = nn.ReflectionPad2d(prepadding)
        self.inc = DoubleConv(n_channels, feature_base, kernel_size=kernel_size)
        self.down1 = Down(feature_base, feature_base*(scale_factor), kernel_size, scale_factor)
        self.down2 = Down(feature_base*(scale_factor), feature_base*(scale_factor**2), kernel_size, scale_factor)
        self.down3 = Down(feature_base*(scale_factor**2), feature_base*(scale_factor**3), kernel_size, scale_factor)
        
        if self.expansive_path == True:
            self.down4 = Down(feature_base*(scale_factor**3), feature_base*(scale_factor**4), kernel_size=3, scale_factor=scale_factor)
        
        # First Decoder Path (Decoder 1)
        if self.expansive_path == True:
            self.up4_1 = Up(feature_base*(scale_factor**4), feature_base*(scale_factor**3) // factor, bilinear, kernel_size, scale_factor)
        self.up3_1 = Up(feature_base*(scale_factor**3), feature_base*(scale_factor**2) // factor, bilinear, kernel_size, scale_factor)
        self.up2_1 = Up(feature_base*(scale_factor**2), feature_base*(scale_factor) // factor, bilinear, kernel_size, scale_factor)
        self.up1_1 = Up(feature_base*(scale_factor), feature_base, bilinear, kernel_size, scale_factor)
        self.outc1 = OutConv(feature_base, n_classes1)  # Output for the first decoder path

        # Second Decoder Path (Decoder 2)
        if self.expansive_path == True:
            self.up4_2 = Up(feature_base*(scale_factor**4), feature_base*(scale_factor**3) // factor, bilinear, kernel_size, scale_factor)
        self.up3_2 = Up(feature_base*(scale_factor**3), feature_base*(scale_factor**2) // factor, bilinear, kernel_size, scale_factor)
        self.up2_2 = Up(feature_base*(scale_factor**2), feature_base*(scale_factor) // factor, bilinear, kernel_size, scale_factor)
        self.up1_2 = Up(feature_base*(scale_factor), feature_base, bilinear, kernel_size, scale_factor)
        self.outc2 = OutConv(feature_base, n_classes2)  # Output for the second decoder path

    def forward(self, x):
        x = self.pad(x)
        x = x1 = self.inc(x)
        x = x2 = self.down1(x)
        x = x3 = self.down2(x)
        x = x4 = self.down3(x)
        
        if self.expansive_path == True:
            x = x5 = self.down4(x)
        
        # Decoder 1
        if self.expansive_path == True:
            x1_1 = self.up4_1(x5, x4)
        else:
            x1_1 = x4
        x1_1 = self.up3_1(x1_1, x3)
        x1_1 = self.up2_1(x1_1, x2)
        x1_1 = self.up1_1(x1_1, x1)
        output1 = self.outc1(x1_1)
        
        # Decoder 2
        if self.expansive_path == True:
            x1_2 = self.up4_2(x5, x4)
        else:
            x1_2 = x4
        x1_2 = self.up3_2(x1_2, x3)
        x1_2 = self.up2_2(x1_2, x2)
        x1_2 = self.up1_2(x1_2, x1)
        output2 = self.outc2(x1_2)

        return output1, output2  # Two outputs, one from each decoder path