import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=1, n_features=64, hidden_layers=8, prepadding=4, kernel_size=3, **kwargs):
        super(CNN, self).__init__()

        self.prepadding = prepadding
        
        self.pad = nn.ReflectionPad2d(prepadding)
        self.inc = Conv(n_channels, n_features, kernel_size, **kwargs)
        self.midc = nn.Sequential(
            *(Conv(n_features, n_features, kernel_size, **kwargs) 
              for _ in range(hidden_layers)))
        self.outc = Conv(n_features, n_classes, kernel_size, **kwargs)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.pad(x)
        x = self.inc(x)
        x = self.midc(x)
        x = self.outc(x)
        
        # Apply sigmoid to return values between 0 and 1
        x = self.sigmoid(x)
        return x
        
        
        
class Conv(nn.Module):
    """convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)
  