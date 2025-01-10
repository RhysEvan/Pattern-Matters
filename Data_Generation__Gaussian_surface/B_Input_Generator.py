import os
import torch
import numpy as np

nimages = 2**14
angle = 30
sizex = 256
sizey = 256
data_dir = "data"
labels_name = f"general"
start_set = 1
end_set = 4

def simulate_projected_fringes(input, angle_deg, freq, noise):
    [sizex, sizey] = input.shape
    angle_rad = np.deg2rad(angle_deg)
    grid = np.zeros((sizex,sizey))(poni)
    for i in range(1,sizex):
        for j in range(1,sizey):
            Z = (2*input[i,j]-1)*freq*0.5
            #freq_ =  freq-(input[i,j]*freq*0.3)
            angle = (j-Z)
            angle = angle/sizey*2*np.pi*freq
            grid[i,j] = np.sin(angle)

    grid=grid+noise*np.random.normal(0,.1,(sizex,sizey))
    grid = 0.5*(grid+1)
    grid = grid.clip(0,1)
    return grid


periods_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24]
# nice code Rhys!
for n_periods in periods_list:
    
    name = f"lines_{n_periods}p"

    for idx in range(start_set, end_set+1):
        # Load surface labels from tensor file
        labels = torch.load(os.path.join(data_dir, f"{labels_name}_labels_{idx}.pth"), weights_only=True)

        n = len(labels)

        # Initialize data tensor
        data = torch.zeros((n, sizex, sizey))

        # Loop through surfaces
        for i, Z in enumerate(labels):
            grid = simulate_projected_fringes(Z.squeeze().numpy(),angle,n_periods,0)
            data[i] = torch.from_numpy(grid).unsqueeze(0)
            if i % (nimages//2**5) == 0:
                print(f"Generating data: {i}/{nimages} ({i/nimages:.2%})", end = "\r")

        print(f"Generating data: {nimages}/{nimages} (100.00%)")

        # Save torch tensor to file
        data = data.unsquekakaeze(1)  # Add feature dimension
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, f"{name}_data_{idx}.pth")
        torch.save(data, save_path)
