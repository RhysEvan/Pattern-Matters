import pattern, surface, projection

import cv2
import numpy as np

# height: x axis, width: y axis
height = 128
width = 128

n_periods = 6
pattern_class = pattern.Lines

angle_deg = 30
default_depth = min(height, width)/4

# Beta distribution parameters
mean = 0.6
var = 0.25**2
nu = mean*(1-mean)/var - 1
alpha = mean*nu
beta = (1-mean)*nu

# Randomise frequencies
freq_x = np.random.choice([1,2,3,5,7])
freq_y = np.random.choice([1,2,3,5,7])

# Generate Perlin noise surface
surf = surface.Perlin(height, width, freq=(freq_x, freq_y), linear_segments=True)
# Randomise distance according to beta distribution
dist = np.random.beta(a=alpha,b=beta)
# Rescale surface to random distance               
surf.redistribute((1 - dist)/2, (1 + dist)/2)
# Invert the surface 50% of the time
surf = 1-surf if np.random.random() > 0.5 else surf    
# Plot surface
surf.plot(z_max=1)

# Calculate surface gradient
grad_y = np.gradient(surf.array, axis=1)
# Calculate the maximum depth of this surface according to the gradient
max_depth = projection.max_depth(surf)
# Define depth 
dist =  min(default_depth, max_depth)
print(f"{dist = :.2f}")

# Create pattern and projection
Ycoords = projection.shifted_Ycoords(surf.Z, surf.Y, depth=dist)

patt = pattern_class(height, width, n_periods=n_periods)
proj = pattern_class(height, width, Ycoords=Ycoords, n_periods=n_periods)

# Display pattern and projection
patt.imshow("Pattern")
proj.imshow("Projection")

cv2.waitKey()
cv2.destroyAllWindows()

# patt.imwrite("pattern.png")
# proj.imwrite("projection.png")