from models.VisionTransformer import DepthEstimationTransformer
from Network_ADAM.trainer_vit import Trainer
from my_utils import CombinedLoss, RMSELoss

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

def training_initiation(name, idx, img_size, patch_size, embed_dim, depth, num_heads, mlp_dim):
    full_name = f"ADAM_{name}_{idx}_{patch_size}_VIT"
    
    out_channels = 1
    # Settings
    seed = idx*(idx+1)
    use_all_devices = False
    data_file = f"data/{name}_data_{idx}.pth"
    labels_file = f"data/general_labels_{idx}.pth"
    network_path = f"output/trained_models/{full_name}.pth"
    figures_path = f"output/figures/{full_name}"
    error_fig_path = f"output/error_figures/{full_name}"
    loss_file = f"output/loss/{full_name}_rmse.txt"
    lr_file = f"output/{'_'.join(full_name.split('_')[:-1])}_lr.txt"
    perf_file = f"output/perf/Unet_3x3_kernel.txt"
    
    n_epochs = 24
    batch_size_train = 24
    batch_size_test = 24

    train_set_ratio = 0.8
    device = "cuda"

    train_network = True

    # Skip already trained networks
    if os.path.exists(network_path) and train_network:
        print(f"Skipping already trained network: {full_name}")
        return

    # Create model
    network =  DepthEstimationTransformer(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, mlp_dim=mlp_dim, out_channels=out_channels
    ).to(device)


    # Load data
    data = torch.load(data_file)
    labels = torch.load(labels_file)
    print("Data loaded")

    # Setup dataset and dataloaders
    print("Initializing dataloaders")
    dataset = utils.data.TensorDataset(data, labels)
    train_size = int(train_set_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Use random_split to split the dataset into training and testing sets
    train_set, test_set = utils.data.random_split(dataset, [train_size, test_size])

    train_loader = utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True) # Data is shuffled at each epoch
    test_loader = utils.data.DataLoader(test_set, batch_size=batch_size_test)
    # Loss function and optimizer
    train_loss_func = CombinedLoss(window_size_input=11).to(device)
    test_loss_func = RMSELoss().to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-4)


    noise_factor = 1
    noise_module = None

    # Initialise the trainer
    trainer = Trainer(
            network, train_loss_func, test_loss_func, optimizer, train_loader, test_loader, device,
            noise_module, noise_factor, name=full_name)
    trainer.create_log()
    trainer.log_append(f"{seed = }", print_comment=True)

    # Create file to write loss values to
    if loss_file:
        os.makedirs(os.path.dirname(loss_file), exist_ok=True)
        open_mode = "w" if train_network else "a"
        with open(loss_file, open_mode) as f:
            pass    # Clear file if it already exists

    # Perform training and testing in multiple epochs
    if train_network:
        trainer.train_and_test(n_epochs=n_epochs, loss_file=loss_file)

    # Save the trained network to a file
    if train_network:
        trainer.log_append("Saving model", print_comment=True)
        trainer.save_network_state(network_path)



# Test script
if __name__ == "__main__":
    # Parameters
    img_size = (128, 128)
    patch_size = 16             # 4         # 32        # 16        # 8
    embed_dim = 768            # 256       # 1024      # 768       # 512
    depth = 12                  # 12        # 12        # 12        # 12
    num_heads = 12              # 4         # 16        # 12        # 8
    mlp_dim = 3072              # 1024      # 4096      # 3072      # 2048

    out_channels = 1
    # Script settings
    # n_periods = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24]
    n_periods = [16]
    # n_periods = [2, 4, 5, 8, 9, 12, 15, 16, 18, 24]
    pattern = "lines"

    for _, i in enumerate(n_periods):

        name = f"{pattern}_{i}p"

        start_set = 1
        end_set = 4

        # Loop through sets
        for idx in range(start_set, end_set+1):


            training_initiation(name, idx, img_size, patch_size, embed_dim, depth, num_heads, mlp_dim)
