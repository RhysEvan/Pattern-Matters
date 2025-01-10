import os
import time
import torch
import matplotlib.pyplot as plt

from my_utils import Unpadder
from my_utils import SSIMLoss, RMSELoss


CLEAR = '\x1b[2K'

class Trainer:

    def __init__(self, network, train_loss_func, test_loss_func, optimizer, train_loader, test_loader, device,
                 scheduler = None, noise_module = None, noise_factor: float = 1, name: str = None):
        # Initialize the class with the network, data loaders and optimizer
        self.network = network
        self.train_loss_func = train_loss_func
        self.test_loss_func = test_loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_file = None
        self.name = name or "unnamed-network"
        self.prepadding = network.prepadding or 0
        self.prepad = torch.nn.ReflectionPad2d(network.prepadding)
        self.unpad = Unpadder(amount=network.prepadding)
        self.noise_module = noise_module
        self.noise_factor = noise_factor


    def train(self, epoch: int, log_int = 1, lr_file: str = None, freeze_scheduler=False, verbose=True):
        self.network.train()  # Set the network in training mode

        avg_loss = 0  # Initialize the train loss
        avg_lr = 0
        length = len(self.train_loader)
        next_perc = 0
        # Iterate through the training data in batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack the batch (data, target1, target2, ...)
            data, *targets = batch

            # Move data and target tensors to the specified device
            data = data.to(self.device)
            targets = [target.to(self.device) for target in targets]

            # Reset the gradients to zero
            self.optimizer.zero_grad()

            # Forward pass to obtain predictions
            preds = self.network(data)
            preds = (preds,) if isinstance(preds, torch.Tensor) else preds
            assert len(preds) == len(targets)

            # Apply reflection padding to the labels
            targets = [self.prepad(target) for target in targets]

            # Calculate the loss between the predictions and the actual target values
            loss = sum(self.train_loss_func(pred, target)
                       for pred, target in zip(preds, targets)
                       )/len(preds)

            # Update the average loss
            avg_loss += loss/length

            # Backpropagation: Compute the gradients
            loss.backward()

            # Update the model's parameters (weights) based on the computed gradients
            self.optimizer.step()

            # Update learning rate and momentum with cyclic scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR) and not freeze_scheduler:
                    self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                avg_lr += current_lr/length
            else:
                for param_group in self.optimizer.param_groups:
                    current_lr = param_group['lr']
                    avg_lr += current_lr/length
            
            # Output intermediate results at the log_interval
            if 100*batch_idx//length >= next_perc:
                current = batch_idx*len(data)
                total = len(self.train_loader.dataset)
                percentage = 100*batch_idx//length
                next_perc = percentage + log_int
                # Print training progress, including epoch, current batch's progress, and loss
                print(
                    f'Train Epoch: {epoch} [{current}/{total} ({percentage:.0f}%)] Loss: {loss.item():.4f} Current LR: {current_lr:.4f}',
                    end="\r")     
        print(CLEAR, end="\r")

        if lr_file is not None:
            os.makedirs(os.path.dirname(lr_file), exist_ok=True)
            with open(lr_file, 'a') as f:
                f.write(f"{avg_lr:.6f}\n")

        self.log_append(f'Training epoch {epoch} complete. Avg LR: {avg_lr:.4f}', print_comment=verbose)
        

    def test(self, log_int=5, add_noise: bool = False, loss_file: str = None, perf_file: str = None, verbose = True):
        # Set the network in evaluation mode (dropout layers are inactive)
        self.network.eval()

        avg_loss = 0  # Initialize the test loss
        length = len(self.test_loader)
        next_perc = 0

        if perf_file:
            tic = time.perf_counter()

        with torch.no_grad():  # Disable gradient computation for testing
            for batch_idx, batch in enumerate(self.test_loader):
                # Unpack the batch (data, target1, target2, ...)
                data, *targets = batch

                # Move data and target tensors to the specified device
                data = data.to(self.device)
                targets = [target.to(self.device) for target in targets]

                # Add noise
                if add_noise:
                    data = self._add_noise(data)

                # Forward pass to obtain predictions
                preds = self.network(data)
                preds = (preds,) if isinstance(preds, torch.Tensor) else preds

                # Remove padding from predictions
                preds = [self.unpad(pred) for pred in preds]
                loss = sum(self.test_loss_func(pred, target)
                           for pred, target in zip(preds, targets)
                           )/len(preds)
                
                # Update the average loss
                avg_loss += loss/length
        
                # Output intermediate results at the log_interval        
                if 100*batch_idx//length >= next_perc:
                    current = batch_idx*len(data)
                    total = len(self.test_loader.dataset)
                    percentage = 100*batch_idx//length
                    next_perc = percentage + log_int
                    
                    # Print testing progress, including epoch, current batch's progress, and loss
                    print(
                        f'Test set - [{current}/{total} ({percentage:.0f}%)] Loss: {loss.item():.4f}',
                        end="\r")
        print(CLEAR, end="\r")

        # Total number of examples in the test dataset
        total = len(self.test_loader.dataset)

        # Print the test results, including the average loss and accuracy
        self.log_append(f'Test complete: Avg. loss = {avg_loss:.6f}', print_comment=verbose)

        # Add loss value to loss file
        if loss_file:
            with open(loss_file, 'a') as f:
                f.write(f"{avg_loss:.6f}\n")

        if perf_file:
            toc = time.perf_counter()
            perf_time = toc - tic
            self.log_append(f"Test completed in {perf_time:.2f} seconds", print_comment=verbose)
            with open(perf_file, 'a') as f:
                    f.write(f"{perf_time:.4f}\n")

        return avg_loss
    
        
    def train_and_test(self, n_epochs=24, prev_epochs=0, loss_file: str = None, lr_file: str = None, figures_path: str = None, verbose = True):
        # Loop through the specified number of epochs and perform training and testing in each epoch
        for epoch in range(1+prev_epochs, n_epochs+1):
            
            if bool(figures_path):
                self.save_figures(n=12, save_path=figures_path+f"epoch_{epoch}", err_max=0.05)

            # Perform training epoch
            self.train(epoch, lr_file=lr_file, verbose=verbose)
            # Perform test
            self.test(loss_file=loss_file, verbose=verbose)
            # Update the learning rate (for stepLR)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                self.scheduler.step()


    def save_network_state(self, network_path: str):
        os.makedirs(os.path.dirname(network_path), exist_ok=True)
        state_dict = self.network.state_dict()
        torch.save(state_dict, network_path)


    def restore_network_state(self, network_path: str):
        state_dict = torch.load(network_path,  map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict)
            

    def lr_range_test(self, n_epochs: int = 12, min_lr=1e-4, max_lr=1e-1, network_path: str = None,
                       loss_file: str = None, lr_file: str = None, verbose = True):
        # Initialise scheduler
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=n_epochs,
            mode = "triangular", cycle_momentum=False)
        for epoch in range(1, n_epochs+1):
            if network_path is not None:
                self.restore_network_state(network_path)
            self.train(epoch, lr_file=lr_file, freeze_scheduler=True, verbose=verbose)
            self.scheduler.step()
            self.test(loss_file=loss_file, verbose=verbose)
            

    def save_figures(self, n: int, save_path: str, err_max: float = None, add_noise: bool = False):

        os.makedirs(save_path, exist_ok=True)

        # Set the network in evaluation mode (dropout layers are inactive)
        self.network.eval()

        with torch.no_grad():  # Disable gradient computation for testing
            # Get one batch of test data
            data, target = next(iter(self.test_loader))
            # Move data and target tensors to the specified device
            data, target = data.to(self.device), target.to(self.device)
            # Add noise
            if add_noise:
                data = self._add_noise(data)
            # Forward pass to obtain predictions
            pred = self.network(data)
            # Remove padding from predictions
            pred = self.unpad(pred)
            
        for idx in range(n):
            for feat in range(pred.shape[1]):
                self._generate_figure(data[idx][0], target[idx][feat], pred[idx][feat], err_max=err_max)
                plt.savefig(f"{save_path}/figure-{idx+1:>03}-{feat+1}")
                plt.close()


    def _generate_figure(self, data: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, err_max: float = None):

        # Calculate losses
        train_loss = float(self.train_loss_func(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0)))
        test_loss = float(self.test_loss_func(pred.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0)))
        
        data, target, pred = data.to("cpu"), target.to("cpu"), pred.to("cpu")

        # Calculate error map
        err = pred - target
        err_max = err_max or min(max(err.max(), -err.min()), 0.1)

        # Create subplots
        fig, ax = plt.subplots(2, 2, layout="constrained")
        fig.suptitle(
            f"{type(self.train_loss_func).__name__}: {train_loss:.6f}, {type(self.test_loss_func).__name__}: {test_loss:.6f}")
        # Create input image figure
        ax[0, 0].imshow(data, cmap='gray', vmin=0, vmax=1)
        ax[0, 0].set_title('Input image')
        # Create output image figure
        im = ax[0, 1].imshow(pred, cmap='gray', vmin=-1, vmax=1)
        ax[0, 1].set_title('Output image')
        plt.colorbar(im, ax=ax[0, 1])
        # Create target figure
        ax[1, 0].imshow(target, cmap='gray', vmin=-1, vmax=1)
        ax[1, 0].set_title('Target')
        # Create error map figure
        im = ax[1, 1].imshow(err, cmap='bwr', vmin=-err_max, vmax=err_max)
        ax[1, 1].set_title('Error map')
        plt.colorbar(im, ax=ax[1, 1])

        # Set tickmarks for each axis
        x_size = data.size()[-2]
        y_size = data.size()[-1]
        xticks = range(0, x_size+1, x_size//4)
        yticks = range(0, y_size+1, y_size//4)
        plt.setp(ax, xticks=xticks, yticks=yticks)
        # Set figure size
        fig.set_size_inches(8, 7, forward=True)

    
    def _add_noise(self, data):
        noise = self.noise_module.sample(data.size()).to(self.device)
        data = data + noise/self.noise_factor
        return data
        
        
    def find_large_errors(self, limit: int, save_path: str, threshold: float = 0.01, log_int = 5):

        os.makedirs(save_path, exist_ok=True)

        # Set the network in evaluation mode (dropout layers are inactive)
        self.network.eval()

        length = len(self.test_loader)
        next_perc = 0
        large_errors_found = 0

        with torch.no_grad():  # Disable gradient computation for testing
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # Move data and target tensors to the specified device
                data, target = data.to(self.device), target.to(self.device)
                # Forward pass to obtain predictions
                pred = self.network(data)
                # Remove padding from predictions
                pred = self.unpad(pred)
                # Generate loss function with reduction disabled
                loss_func = type(self.test_loss_func)(reduction = "none")
                # Find the mean loss value for every datapoint
                loss = loss_func(pred, target).mean(dim=(1, 2, 3))
                # Find the maximum loss value
                arg = torch.argmax(loss)
                if loss[arg] > threshold:
                    batch_size = len(data)
                    data_idx = batch_size*batch_idx + arg
                    for feat in range(pred.shape[1]):
                        self._generate_figure(data[arg][0], target[arg][feat], pred[arg][feat])
                        plt.savefig(f"{save_path}/error_figure-{data_idx+1:>04}-{feat+1}")
                        plt.close()
                    large_errors_found += 1
                
                    if large_errors_found == limit:
                        break

                # Output intermediate results at the log_interval        
                if 100*batch_idx//length >= next_perc:
                    current = batch_idx*len(data)
                    total = len(self.test_loader.dataset)
                    percentage = 100*batch_idx//length
                    next_perc = percentage + log_int
                    
                    # Print testing progress, including epoch, current batch's progress, and loss
                    print(
                        f'Test set - [{current}/{total} ({percentage:.0f}%)] Large errors found: {large_errors_found}',
                        end="\r")
        print(CLEAR, end="\r")


    def create_log(self, log_dir: str = "output/logs"):
        """Create a new log file with a uniqe name"""
        # Get current time
        t = time.localtime()
        date_str = time.strftime("%Y-%m-%d", t)
        index = 1
        os.makedirs(log_dir, exist_ok=True)
        while index < 100:
            # Create a unique log file
            try:
                path = os.path.join(log_dir, f"{date_str}_{self.name}_{index}.log")
                with open(path, "x") as f:
                    self.log_file = f.name
                break
            except FileExistsError:
                pass
            index += 1
            if index == 100:
                raise RuntimeError("Failed to create log file.")
        self.log_append(f"New log file created for {self.name}", print_comment = True)
            
        
    def log_append(self, comment: str, log_file: str = None, include_time: bool = True, print_comment: bool = False):
        """Append an entry to the log file"""
        if print_comment:
            print(comment)

        log_file = log_file or self.log_file
        if not bool(log_file):
            return      # Do nothing (no log file available)
        
        with open(log_file, 'a') as log:
            t = time.localtime()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", t)
            prefix = f"[{time_str}]\t" if include_time else "\t"
            comment = str(comment).replace('\n','\n\t')
            log.write(f"{prefix}{comment}\n")    
