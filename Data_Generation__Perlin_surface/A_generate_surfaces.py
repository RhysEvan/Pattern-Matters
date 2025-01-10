import surface, projection

import numpy as np
import torch
import torchvision
import os


# Script settings
# height: x axis, width: y axis
height = 480
width = 640
proj_depth = width/8
data_dir = "data"
labels_name = "general"
resize_data = False

# Number of surfaces to generate per dataset
n = 2**14

# Dataset index range
start_set = 5
end_set = 5

# Beta distribution parameters
mean = 0.6
var = 0.25**2
nu = mean*(1-mean)/var - 1
alpha = mean*nu
beta = (1-mean)*nu

for idx in range(start_set, end_set+1):

    labels = torch.zeros(n, height, width)

    for i in range(n):
        
        # Randomise frequencies
        freq_x = np.random.choice([1,2,3,5,7])
        freq_y = np.random.choice([1,2,3,5,7])
        
        max_depth = 0
        while max_depth*0.85 < proj_depth:       # Regenerate surface when max_depth is too low
            # Generate Perlin noise surface
            surf = surface.Perlin(height, width, freq=(freq_x, freq_y), linear_segments=True)
            # Randomise distance according to beta distribution
            dist = np.random.beta(a=alpha,b=beta)
            # Rescale surface to random distance               
            surf.redistribute((1 - dist)/2, (1 + dist)/2)
            # Invert the surface 50% of the time
            surf = 1-surf if np.random.random() > 0.5 else surf     
            # Calculate the maximum depth of this surface according to the gradient
            max_depth = projection.max_depth(surf)                  

        # Add surface to labels tensor
        labels[i] = torch.from_numpy(surf.array).unsqueeze(0)
        
        if i % (n//2**5) == 0:
            print(f"Generating data: {i}/{n} ({i/n:.2%})", end = "\r")
    print(f"Generating data: {n}/{n} (100.00%)")
    
    # Resize the surface through bilinear interpolation
    if resize_data:
        resize_factor = 1
        resizer = torchvision.transforms.Resize((height*resize_factor, width*resize_factor), antialias=True)
        labels = resizer(labels)
        print("Resized labels to", labels.shape)

    # Save torch tensor to file
    labels = labels.unsqueeze(1)  # Add feature dimension
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f"{labels_name}_labels_{idx}.pth")
    torch.save(labels, save_path)

