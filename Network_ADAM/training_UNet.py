from models.unet import UNet
from trainer import Trainer
from my_utils import SSIMLoss, RMSELoss, CombinedLoss

import os
import torch
import torchvision
import torch.utils as utils

def unet_train(name, idx):

    full_name = f"avg_pool_{name}_{idx}_UNet"

    # Settings
    seed = idx*(idx+1)
    use_all_devices = False
    data_file = f"data/{name}_data_{idx}.pth"
    labels_file = f"data/general_labels_{idx}.pth"
    network_path = f"output/trained_models/{full_name}_lr_e-3.pth"
    figures_path = f"output/figures/{full_name}"
    error_fig_path = f"output/error_figures/{full_name}"
    loss_file = f"output/loss/{full_name}_rmse_lr_e-3.txt"
    lr_file = f"output/{'_'.join(full_name.split('_')[:-1])}_lr.txt"
    perf_file = f"output/perf/Unet_3x3_kernel.txt"

    # Define hyperparameters
    n_epochs = 24
    dataset_size = 2**14
    batch_size_train = 128
    batch_size_test = 128
    train_set_ratio = 0.8
    learning_rate = 0.03
    min_lr = 10**-4
    max_lr = 6*10**-2
    momentum = 0.8
    step_size = dataset_size*train_set_ratio/batch_size_train/2

    # # Enable or disable tasks below
    # Resize data to new_size
    resize_data = False
    # Train & test the network (loss appended to loss file) and save it to a file              
    train_network = True
    # Load & evaluate a previously trained network (loss appended to loss file)                
    evaluate_network = not train_network
    # Add noise to the training data before evaluation            
    add_noise = False
    # Generate and save figures comparing network output to labels               
    save_figures = False
    # Identify datapoints where network performs poorly and generate figures                
    find_large_errors = False

    # Skip already trained networks
    if os.path.exists(network_path) and train_network:
        print(f"Skipping already trained network: {full_name}")
        return

    # Set manual seed and disable nondeterministic algorithms
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)

    # Check for GPU Availability
    if torch.cuda.is_available():
        print("CUDA is available, working on GPU")
        device = torch.device("cuda")
    else:
        print("CUDA is not available, working on CPU")
        device = torch.device("cpu")

    # Initialise the network
    print("Initialising network")
    network = UNet(prepadding=2, feature_base=16)

    # Send network to GPU device
    device_count = torch.cuda.device_count()
    if device_count > 1 and use_all_devices:
        print(f"Using {device_count} available GPU's.")
        network = torch.nn.DataParallel(network)     # Use multiple GPU
    network = network.to(device)

    # Set the loss function
    # train_loss_func = SSIMLoss(window_size=11).to(device)
    train_loss_func = CombinedLoss(window_size_input=11).to(device)
    test_loss_func = RMSELoss().to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Load data
    data = torch.load(data_file)
    labels = torch.load(labels_file)
    print("Data loaded")

    # Transform data
    if resize_data:
        new_size = 81
        resizer = torchvision.transforms.Resize(new_size, antialias=True)
        data = resizer(data)
        labels = resizer(labels)

    # Setup dataset and dataloaders
    print("Initializing dataloaders")
    dataset = utils.data.TensorDataset(data, labels)
    train_size = int(train_set_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Use random_split to split the dataset into training and testing sets
    train_set, test_set = utils.data.random_split(dataset, [train_size, test_size])

    train_loader = utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True) # Data is shuffled at each epoch
    test_loader = utils.data.DataLoader(test_set, batch_size=batch_size_test)

    # Create shot noise module
    if add_noise:
        # noise_module = torch.distributions.Normal(loc=0, scale=0.01)
        noise_rate = 160
        noise_factor = 4095
        noise_module = torch.distributions.Poisson(rate=noise_rate)
        noise_fig_path = f"shot_noise_figures/{full_name}_{noise_rate}"
    else:
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

    # Create file to write performance to
    if perf_file and evaluate_network:
        os.makedirs(os.path.dirname(perf_file), exist_ok=True)
        with open(perf_file, "a") as f:
            pass    # Clear file if it already exists

    # Load the trained network state
    if evaluate_network:
        print("Loading network state dict")
        trainer.restore_network_state(network_path)

    # Perform training and testing in multiple epochs
    if train_network:
        trainer.train_and_test(n_epochs=n_epochs, loss_file=loss_file)

    # Evaluate trained network
    if evaluate_network:
        trainer.test(add_noise=add_noise, loss_file=loss_file, perf_file=perf_file)

    # Save the trained network to a file
    if train_network:
        trainer.log_append("Saving model", print_comment=True)
        trainer.save_network_state(network_path)

    # Generate and save figures
    if save_figures:
        trainer.log_append("Generating figures", print_comment=True)
        trainer.save_figures(n=24, save_path=figures_path)

    # Find large errors
    if find_large_errors:
        trainer.find_large_errors(limit=24, save_path=error_fig_path, threshold=0.02)

if __name__ == "__main__":

     # Script settings
    n_periods = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24]
    # n_periods = [2, 4, 5, 8, 9, 12, 15, 16, 18, 24]
    pattern = "lines"

    for _, i in enumerate(n_periods):

        name = f"{pattern}_{i}p"

        start_set = 1
        end_set = 4

        # Loop through sets
        for idx in range(start_set, end_set+1):

            unet_train(name, idx)
