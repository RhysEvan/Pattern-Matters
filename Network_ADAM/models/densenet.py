# Source: https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """
        A basic block used in DenseNet. It consists of BatchNorm, ReLU, and a Conv2D layer.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            dropRate (float): Dropout rate (default: 0.0).
        """
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        """
        A bottleneck block used in DenseNet. It reduces the number of channels before the expensive 3x3 convolution.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels (growth rate).
            dropRate (float): Dropout rate (default: 0.0).
        """
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    """
        A transition block used in DenseNet. It reduces the size of feature maps and the number of channels.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels after transition.
            dropRate (float): Dropout rate (default: 0.0).
        """
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    """
        A dense block consisting of multiple convolutional layers where each layer is connected to every other layer.

        Args:
            nb_layers (int): Number of layers in the block.
            in_planes (int): Number of input channels.
            growth_rate (int): Number of output channels for each layer.
            block (nn.Module): Type of block to use (BasicBlock or BottleneckBlock).
            dropRate (float): Dropout rate (default: 0.0).
        """
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = self._make_layers(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layers(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        """
        DenseNet architecture for image classification.

        Args:
            depth (int): Depth of the network (total number of layers).
            num_classes (int): Number of output classes.
            growth_rate (int): Number of output channels per block (default: 12).
            reduction (float): Compression factor at transition layers (default: 0.5).
            bottleneck (bool): Whether to use bottleneck blocks (default: True).
            dropRate (float): Dropout rate (default: 0.0).
        """
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

class DenseNetFCN(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, prepadding=2):
        """
        Fully Convolutional DenseNet (FCN) for image segmentation or pixel-wise classification.

        Args:
            depth (int): Depth of the network (total number of layers).
            num_classes (int): Number of output classes for pixel-wise classification.
            growth_rate (int): Number of output channels per block (default: 12).
            reduction (float): Compression factor at transition layers (default: 0.5).
            bottleneck (bool): Whether to use bottleneck blocks (default: True).
            dropRate (float): Dropout rate (default: 0.0).
            prepadding (int): Amount of reflection padding to apply before the initial convolution (default: 0).
        """
        super(DenseNetFCN, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n // 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)

        # Pre-padding
        self.prepadding = prepadding
        self.pad = nn.ReflectionPad2d(prepadding)

        # Initial convolution
        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense blocks with transition layers
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = in_planes + n * growth_rate
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = in_planes + n * growth_rate
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = in_planes + n * growth_rate

        # BatchNorm and ReLU before upsampling
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Final 1x1 convolution to reduce to num_classes output channels (for pixel-wise classification)
        self.final_conv = nn.Conv2d(in_planes, num_classes, kernel_size=1, stride=1, padding=0)

        # Upsampling layer to match the input size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Pre-padding
        x = self.pad(x)

        # Initial convolution and dense blocks with transitions
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)

        # BatchNorm and ReLU
        x = self.relu(self.bn1(x))

        # Upsample back to input resolution
        x = self.upsample(x)

        # Final 1x1 conv to reduce to num_classes
        x = self.final_conv(x)

        # Apply sigmoid to return values between 0 and 1
        x = self.sigmoid(x)

        return x
