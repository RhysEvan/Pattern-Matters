from models.ynet import YNet
from trainer import Trainer
from my_utils import SSIMLoss, RMSELoss

import os
import torch
import torchvision
import torch.utils as utils

def ynet_train(name, idx):

    full_name = f"YNet_{name}_{idx}"

    # Settings
    seed = idx*(idx+1)
    use_all_devices = False
    data_file = f"data/{name}_data_{idx}.pth"
    nums_file = f"data/{name}_nums_{idx}.pth"
    dens_file = f"data/{name}_dens_{idx}.pth"
    network_path = f"output/trained_models/{full_name}.pth"
    figures_path = f"output/figures/{full_name}"
    error_fig_path = f"output/error_figures/{full_name}"
    loss_file = f"output/loss/{full_name}_rmse.txt"
    lr_file = f"output/{name}_lr.txt"
    perf_file = f"output/perf/Unet_3x3_kernel.txt"

    # Define hyperparameters
    n_epochs = 24
    dataset_size = 2**14
    batch_size_train = 64
    batch_size_test = 64
    train_set_ratio = 0.8
    learning_rate = 0.03
    min_lr = 10**-5
    max_lr = 4*10**-2
    momentum = 0.8
    step_size = dataset_size*train_set_ratio/batch_size_train/2

    # # Enable or disable tasks below
    # Resize data to new_size
    resize_data = False
    # Train & test the network (loss appended to loss file) and save it to a file              
    train_network = True
    # Load & evaluate a previously trained network (loss appended to loss file)                
    evaluate_network = not train_network
    # Add ambient light to the training data before training
    add_ambient_light = True
    # Add noise to the training data before evaluation            
    add_noise = False
    # Perform a LR range test                   
    test_lr_range = False
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
        device = torch.device("cuda:2")
    else:
        print("CUDA is not available, working on CPU")
        device = torch.device("cpu")

    # Initialise the network
    print("Initialising network")
    network = YNet(prepadding=2, feature_base=16)

    # Send network to GPU device
    device_count = torch.cuda.device_count()
    if device_count > 1 and use_all_devices:
        print(f"Using {device_count} available GPU's.")
        network = torch.nn.DataParallel(network)     # Use multiple GPU
    network = network.to(device)

    # Set the loss function
    train_loss_func = RMSELoss().to(device)
    test_loss_func = RMSELoss().to(device)

    # Create an optimizer for training the neural network using Stochastic Gradient Descent (SGD)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Initiate a scheduler to dynamically update the learning rate and momentum
    scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size,
            mode = "exp_range", gamma=0.9995, cycle_momentum=True)

    # Load data
    data = torch.load(data_file, weights_only=True)
    nums = torch.load(nums_file, weights_only=True)
    dens = torch.load(dens_file, weights_only=True)
    print("Data loaded")


    # Add ambient light to the input data
    if add_ambient_light:
        sigma = 0.001

        Axy = 0.1 + (sigma**0.5)*torch.randn(*data.shape[-2:])
        Bxy = 0.9 + (sigma**0.5)*torch.randn(*data.shape[-2:])
        
        data = Axy + Bxy*data
        nums = Bxy*nums
        dens = Bxy*dens    

    # Transform data
    if resize_data:
        new_size = 81
        resizer = torchvision.transforms.Resize(new_size, antialias=True)
        data = resizer(data)
        nums = resizer(nums)
        dens = resizer(dens)
        print(f"Resized data to {data.shape}")

    # Setup dataset and dataloaders
    print("Initializing dataloaders")
    dataset = utils.data.TensorDataset(data, nums, dens)
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
            scheduler, noise_module, noise_factor, name=full_name)
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

    # Perform LR range test
    if test_lr_range:
        trainer.lr_range_test(n_epochs=24, min_lr=min_lr, max_lr=max_lr, loss_file=loss_file, lr_file=lr_file, network_path=network_path)

    # Generate and save figures
    if save_figures and idx == 1:
        trainer.log_append("Generating figures", print_comment=True)
        trainer.save_figures(n=24, save_path=figures_path)

    # Find large errors
    if find_large_errors and idx == 1:
        trainer.find_large_errors(limit=24, save_path=error_fig_path, threshold=0.02)

if __name__ == "__main__":

    # Script settings
    n_periods = 8
    pattern = "lines"

    name = f"{pattern}_{n_periods}p"

    start_set = 1
    end_set = 4

    # Loop through sets
    for idx in range(start_set, end_set+1):

        ynet_train(name, idx)
