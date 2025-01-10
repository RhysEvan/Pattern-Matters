import pattern, projection

import numpy as np
import torch
import os


# periods_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24]
periods_list = [16]

for n_periods in periods_list:
    
    # height: x axis, width: y axis
    height = 480
    width = 640

    # Settings
    data_dir = "data"
    labels_name = "general"
    name = f"lines-saw_{n_periods}p"
    pattern_class = pattern.LinesSaw
    angle_deg = 30
    proj_depth = width/8

    start_set = 5
    end_set = 5
    
    # Create meshgrid
    x = np.arange(0, height)
    y = np.arange(0, width)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Loop through sets
    for idx in range(start_set, end_set+1):

        # Load surface labels from tensor file
        labels = torch.load(os.path.join(data_dir, f"{labels_name}_labels_{idx}.pth"), weights_only=True)
        
        n = len(labels)
        
        # Initialize data tensor
        data = torch.zeros((n, height, width))

        # Loop through surfaces
        for i, Z in enumerate(labels):
            
            # Project pattern
            Ycoords = projection.shifted_Ycoords(Z.squeeze().numpy(), Y, depth=proj_depth)
            proj = pattern_class(height, width, n_periods=n_periods, Ycoords=Ycoords)
            
            # Add projected pattern to data tensor
            data[i] = torch.from_numpy(proj.array).unsqueeze(0)
            
            if i % (n//2**5) == 0:
                print(f"Generating data: {i}/{n} ({i/n:.2%})", end = "\r")
        print(f"Generating data: {n}/{n} (100.00%)")

        # Save torch tensor to file
        print(f"Saving data...", end = "\r")
        data = data.unsqueeze(1)  # Add feature dimension
        torch.save(data, os.path.join(data_dir, f"{name}_data_{idx}.pth"))

