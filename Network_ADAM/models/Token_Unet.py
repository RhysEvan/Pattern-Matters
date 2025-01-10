# Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet
# A sigmoid function and a prepadding module were added to the UNet


import torch
import torch.nn as nn
import torch.nn.functional as func


class Unpadder(nn.Module):
    
    def __init__(self, amount):
        super(Unpadder, self).__init__()
        self.amount = amount

    def forward(self, x):
        return x[:, :, self.amount:-self.amount, self.amount:-self.amount]


class ImagePatcher(nn.Module):
    def __init__(self, patch_size=160, img_size=(480, 640), in_channels=1):
        super(ImagePatcher, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Number of patches in each dimension
        self.num_patches_h = self.img_size[0] // self.patch_size
        self.num_patches_w = self.img_size[1] // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

    def forward(self, x):
        # Ensure input x has the shape (batch_size, in_channels, height, width)
        batch_size = x.size(0)  # Get batch size

        # Break input into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape patches: (batch_size, in_channels, num_patches_h, num_patches_w, patch_size, patch_size)

        patches = patches.contiguous().view(batch_size, self.in_channels, self.num_patches_h, self.num_patches_w, self.patch_size, self.patch_size)

        # Rearranging dimensions to have patches in the last dimension
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (batch_size, num_patches_h, num_patches_w, in_channels, patch_size, patch_size)
        
        # Reshape to (batch_size * num_patches, in_channels, patch_size, patch_size)
        patches = patches.contiguous().view(-1, self.in_channels, self.patch_size, self.patch_size)

        return patches

class Token_UNet(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, prepadding=4, feature_base=64,
                 expansive_path=True, kernel_size=3, scale_factor=2, patch_size=160, img_size=(480, 640)):
        super(Vit_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.prepadding = prepadding
        self.feature_base = feature_base
        self.expansive_path = expansive_path
        self.patch_size = patch_size

        # Initialize ImagePatcher
        self.patcher = ImagePatcher(patch_size=patch_size, img_size=img_size, in_channels=n_channels)

        factor = 2 if bilinear else 1

        self.pad = nn.ReflectionPad2d(prepadding)
        self.unpad = Unpadder(prepadding)
        self.inc = DoubleConv(n_channels, feature_base, kernel_size=kernel_size)
        self.down1 = Down(feature_base, feature_base*(scale_factor), kernel_size, scale_factor)
        self.down2 = Down(feature_base*(scale_factor), feature_base*(scale_factor**2), kernel_size, scale_factor)
        self.down3 = Down(feature_base*(scale_factor**2), feature_base*(scale_factor**3), kernel_size, scale_factor)
        
        if self.expansive_path:
            self.down4 = Down(feature_base*(scale_factor**3), feature_base*(scale_factor**4), kernel_size=3, scale_factor=scale_factor)
            self.up4 = Up(feature_base*(scale_factor**4), feature_base*(scale_factor**3) // factor, bilinear, kernel_size, scale_factor)
        
        self.up3 = Up(feature_base*(scale_factor**3), feature_base*(scale_factor**2) // factor, bilinear, kernel_size, scale_factor)
        self.up2 = Up(feature_base*(scale_factor**2), feature_base*(scale_factor) // factor, bilinear, kernel_size, scale_factor)
        self.up1 = Up(feature_base*(scale_factor), feature_base, bilinear, kernel_size, scale_factor)
        self.outc = OutConv(feature_base, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Patch division
        patches = self.patcher(x)  # Shape: (batch_size * num_patches, in_channels, patch_size, patch_size)

        # Step 2: Process patches in parallel through the U-Net
        # We are not iterating over individual patches, but processing the whole batch of patches
        patches = self.pad(patches)  # Apply padding to all patches
        x1 = self.inc(patches)       # Initial double conv on all patches
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.expansive_path:
            x5 = self.down4(x4)
            x = self.up4(x5, x4)
        else:
            x = x4

        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out_patches = self.outc(x)  # Final output conv on all patches
        
        # Step 3: Apply sigmoid to the output patches
        out_patches = self.sigmoid(out_patches)  # Shape: (batch_size * num_patches, n_classes, patch_size, patch_size)
        out_patches = self.unpad(out_patches)

        # Step 4: Reassemble patches
        batch_size = x.size(0) // (self.patcher.num_patches_h * self.patcher.num_patches_w)
        num_patches_h = self.patcher.num_patches_h
        num_patches_w = self.patcher.num_patches_w

        # Reshape and permute to reconstruct the full image
        out_image = out_patches.view(batch_size, num_patches_h, num_patches_w, self.n_classes, self.patch_size, self.patch_size)
        out_image = out_image.permute(0, 3, 1, 4, 2, 5)  # (batch_size, n_classes, num_patches_h, patch_size, num_patches_w, patch_size)

        # Final reshape to merge patches back into original image dimensions
        out_image = out_image.contiguous().view(batch_size, self.n_classes, num_patches_h * self.patch_size, num_patches_w * self.patch_size)

        return out_image



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

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=scale_factor),
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

