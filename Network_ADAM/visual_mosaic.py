import os

import torch
import torch.nn.functional as F
import torch.utils as utils

import matplotlib.pyplot as plt

from my_utils import Unpadder, SSIMLoss, RMSELoss

from models.unet import UNet
from models.densenet import DenseNetFCN
from Network_ADAM.models.Token_Unet import Vit_UNet
from models.VisionTransformer import DepthEstimationTransformer

def save_figures(n: int, save_path: str, err_max: float = None):

    os.makedirs(save_path, exist_ok=True)

    # Set the network in evaluation mode (dropout layers are inactive)
    network.eval()

    with torch.no_grad():  # Disable gradient computation for testing
        # Get one batch of test data
        data, target = next(iter(test_loader))
        print(data.shape)
        print(target.shape)
        # data = F.interpolate(data, size=(256, 256), mode='bilinear', align_corners=False)
        # target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)
        # print(data.shape)
        # print(target.shape)
        # Move data and target tensors to the specified device
        data, target = data.to(device), target.to(device)
        # Forward pass to obtain predictions
        pred = network(data)
        # Remove padding from predictions
        # pred = unpad(pred)
        
    for idx in range(n):
        generate_figure(data[idx], target[idx], pred[idx], err_max=err_max)
        plt.savefig(f"{save_path}/figure-{idx+1:>03}")
        plt.close()


def generate_figure(data: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, err_max: float = None):

    # Initialise loss functions
    ssim_loss_func = SSIMLoss(window_size=11).to(device)
    rmse_loss_func = RMSELoss().to(device)

    # Calculate losses
    ssim_loss = float(ssim_loss_func(pred.unsqueeze(0), target.unsqueeze(0)))
    rmse_loss = float(rmse_loss_func(pred.unsqueeze(0), target.unsqueeze(0)))
    
    data, target, pred = data.to("cpu"), target.to("cpu"), pred.to("cpu")

    # Calculate error map
    err = pred - target
    err_max = err_max or min(max(err.max(), -err.min()), 0.1)

    # Create subplots
    fig, ax = plt.subplots(2, 2, layout="constrained")
    fig.suptitle(f"{ssim_loss=:.6f}, {rmse_loss=:.6f}")
    # Create input image figure
    ax[0, 0].imshow(data.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax[0, 0].set_title('Input image')
    # Create output image figure
    im = ax[0, 1].imshow(pred.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax[0, 1].set_title('Output image')
    plt.colorbar(im, ax=ax[0, 1])
    # Create target figure
    ax[1, 0].imshow(target.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax[1, 0].set_title('Target')
    # Create error map figure
    im = ax[1, 1].imshow(err.squeeze(), cmap='bwr', vmin=-err_max, vmax=err_max)
    ax[1, 1].set_title('Error map')
    plt.colorbar(im, ax=ax[1, 1])
    
    # Set figure size
    fig.set_size_inches(8, 7, forward=True)



data_file = "./data/lines_16p_data_5.pth"
labels_file = "./data/general_labels_5.pth"

data = torch.load(data_file)
labels = torch.load(labels_file)

dataset = utils.data.TensorDataset(data, labels)

test_loader = utils.data.DataLoader(dataset, batch_size=24)

# img_size = (128, 128)
# patch_size = 16             # 4         # 32        # 16        # 8
# embed_dim = 768            # 256       # 1024      # 768       # 512
# depth = 12                  # 12        # 12        # 12        # 12
# num_heads = 12              # 4         # 16        # 12        # 8
# mlp_dim = 3072  
# out_channels = 1

# network = DepthEstimationTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
#         num_heads=num_heads, mlp_dim=mlp_dim, out_channels=out_channels)
# network.load_state_dict(torch.load("./output/trained_models/adam_vit_128/ADAM_lines_16p_1_16_VIT.pth", weights_only=True))

patch_size = 160
input_resolution = (480,640)

network = Vit_UNet(prepadding=2, feature_base=16, patch_size=patch_size, img_size=input_resolution)
network.load_state_dict(torch.load("./output/trained_models/Unet_token/ADAM_lines_16p_5_ViT_Unet.pth", weights_only=True))

unpad = Unpadder(amount=network.prepadding)
device = "cuda:0"

network.to(device)
save_figures(n=24, save_path="UNet_Token")