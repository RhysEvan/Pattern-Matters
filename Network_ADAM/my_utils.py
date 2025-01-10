import torch
import torch.nn as nn
from piqa import SSIM
from torch.utils.data import Dataset
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, lambda1=0.7, lambda2=0.3, window_size_input = 11):
        super(CombinedLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.RMSE = RMSELoss()
        self.SSIM = SSIMLoss(window_size=window_size_input)

    def forward(self, predicted_depth, ground_truth_depth):
        rmse_loss = self.RMSE(predicted_depth, ground_truth_depth)
        ssim_loss = self.SSIM(predicted_depth, ground_truth_depth)
        return self.lambda1 * rmse_loss + self.lambda2 * ssim_loss

class SSIMLoss(SSIM):
    
    def __init__(self, window_size = 11):
        super(SSIMLoss, self).__init__(window_size=window_size, n_channels=1)

    def forward(self, x, y):
        return 1.00 - super().forward(x, y)


class RMSELoss(nn.Module):
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(RMSELoss,self).__init__()
        self.criterion = nn.MSELoss(size_average, reduce, reduction)

    def forward(self, x, y):
        loss = torch.sqrt(self.criterion(x, y))
        return loss


# Custom unpadder

class Unpadder(nn.Module):
    
    def __init__(self, amount):
        super(Unpadder, self).__init__()
        self.amount = amount

    def forward(self, x):
        return x[:, :, self.amount:-self.amount, self.amount:-self.amount]


# Matlab data converter

class MatDataset(Dataset):
    def __init__(self, mat_data: dict, transform=None):
        # Assign data and labels
        self.data = torch.Tensor(mat_data['x'][:,:,:128]).permute(2, 0, 1).unsqueeze(dim=1)
        self.labels = torch.Tensor(mat_data['y'][:,:,:128]).permute(2, 0, 1).unsqueeze(dim=1)

        # Assign transformer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.data[idx], self.labels[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample